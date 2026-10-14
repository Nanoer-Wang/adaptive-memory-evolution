"""Repair explicit addressing before testing duplicate-write savings.

Stage one only uses the existing qa1 split. A separate, frozen transfer stage
will consume qa6; its results must not select any checkpoint or policy.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,copy,hashlib
import numpy as np
import torch
from src.memory.addressed_codec import AddressedCodec
from scripts.bounded_codec_study import prepare,tensorize,score
from src.provenance import snapshot_source,runtime_info

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/addressed-memory-v1'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    if OUT.exists(): raise SystemExit('Frozen run exists; will not overwrite.')
    OUT.mkdir(parents=True);torch.set_num_threads(1)
    rows,entities,values,hashes=prepare()
    splits={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    tensors={s:tensorize(r,4,6) for s,r in splits.items()}
    signature,archive=snapshot_source()
    save('protocol.json',dict(signature=signature,source_archive=archive,runtime=runtime_info(),data_hashes=hashes,
        entities=entities,values=values,split_sizes={s:len(r) for s,r in splits.items()},
        kinds=['input_flow','gru','ltc'],slot_widths=[2,8],seeds=[11,23,42],epochs=200,patience=40,lr=.003,batch=128,
        checkpoint='best validation cross entropy; test never used for selection',
        hypothesis='Explicit entity-addressed writes/reads repair binding; compare shared state at same float32 payload, not same parameter count.',
        memory='4*slot_width*4 bytes; no history, clock or event cache; shared weights and ontology separately counted.',
        scope='qa1 development diagnostic, previously inspected; NOT blind generalization. No artificial channel loss.',
        followup='Freeze checkpoints before new public qa6 transfer. Always-write and exact duplicate guard; native and deterministic fourfold repeats clearly separated. No liquid-specific skipping claim.',
        acceptance='Foundational goal >=99% qa1 validation accuracy; report all capacities/seeds even if goal not reached.'))
    save('examples.json',rows);results=[];predictions=[]
    for kind in ('input_flow','gru','ltc'):
        for width in (2,8):
            for seed in (11,23,42):
                torch.manual_seed(seed);model=AddressedCodec(kind,width)
                optimizer=torch.optim.Adam(model.parameters(),lr=.003)
                best=float('inf');best_state=None;stale=0;history=[];start=time.perf_counter()
                for epoch in range(200):
                    model.train();losses=[]
                    for batch in torch.randperm(len(splits['train'])).split(128):
                        x,v,q,y=(t[batch] for t in tensors['train'])
                        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x,v,q),y)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
                    model.eval();vn,va=score(model,tensors['validation'])
                    history.append(dict(epoch=epoch+1,train_nll=float(np.mean(losses)),val_nll=vn,val_accuracy=va))
                    if vn<best:best=vn;best_state=copy.deepcopy(model.state_dict());stale=0
                    else:stale+=1
                    if stale>=40:break
                elapsed=time.perf_counter()-start;model.load_state_dict(best_state);model.eval()
                name=f'{kind}-{width}-{seed}';torch.save(best_state,OUT/f'{name}.pt');save(f'{name}-epochs.json',history)
                vn,va=score(model,tensors['validation']);tn,ta=score(model,tensors['test'])
                with torch.no_grad():
                    batched=model(*tensors['test'][:3]).argmax(-1).tolist()
                    for i,row in enumerate(splits['test']):
                        state=model.initial()
                        for owner,value in row['events']:
                            state=model.write(state,torch.tensor([owner]),torch.eye(6)[value:value+1])
                        payload=model.pack(state);assert len(payload)==width*4*4
                        pred=int(model.read(model.unpack(payload),torch.tensor([row['query']])).argmax(-1))
                        assert pred==batched[i]
                        predictions.append(dict(id=row['id'],story=row['story'],method=name,label=row['label'],prediction=pred))
                results.append(dict(method=name,kind=kind,slot_width=width,seed=seed,state_bytes=width*16,
                    weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),val_accuracy=va,val_nll=vn,
                    test_accuracy=ta,test_nll=tn,epochs=len(history),train_seconds=elapsed,
                    checkpoint_sha256=hashlib.sha256((OUT/f'{name}.pt').read_bytes()).hexdigest()))
                save('summary.json',results)
                print(name,f'val={va:.4f} test={ta:.4f} {elapsed:.1f}s',flush=True)
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    save('complete.json',dict(models=18,predictions=len(predictions),state_only_replay_passed=True,api_requests=0))


if __name__=='__main__':main()

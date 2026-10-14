"""First capacity gate: public qa1, fixed-state codec versus exact table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, hashlib, time, copy
import numpy as np
import torch
from src.data.babi_events import stories, read_key
from src.memory.version_ledger import Evidence
from src.memory.bounded_codec import BoundedCodec, pack_state, unpack_state
from src.provenance import snapshot_source, runtime_info, rss_mb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/bounded-codec-v2'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare():
    sources = {s: ROOT / f'data/babi/qa1_single-supporting-fact_{s}.txt' for s in ('train', 'test')}
    corpus = {s: stories(p, s) for s, p in sources.items()}
    entities = sorted({e.owner for story in corpus['train'] for e in story if isinstance(e, Evidence)})
    values = sorted({e.value for story in corpus['train'] for e in story if isinstance(e, Evidence)})
    rows = []
    for origin, group in corpus.items():
        for sid, story in enumerate(group):
            digest = hashlib.sha256('\n'.join(e.quote if isinstance(e, Evidence) else e['question'] for e in story).encode()).hexdigest()
            split = 'test' if origin == 'test' else ('validation' if int(digest[:8], 16) % 5 == 0 else 'train')
            events = []
            for e in story:
                if isinstance(e, Evidence):
                    events.append((entities.index(e.owner), values.index(e.value)))
                else:
                    rows.append(dict(id=f'{origin}:{sid}:{e["line"]}', story=f'{origin}:{sid}', split=split,
                                     events=list(events), query=entities.index(read_key(e['question'])[0]), label=values.index(e['answer'])))
    return rows, entities, values, {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in sources.items()}


def tensorize(rows, ne, nv):
    x = torch.zeros(len(rows), max(len(r['events']) for r in rows), ne + nv)
    valid = torch.zeros(x.shape[:2], dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, (owner, value) in enumerate(row['events']):
            x[i, j, owner] = 1; x[i, j, ne + value] = 1; valid[i, j] = True
    q = torch.nn.functional.one_hot(torch.tensor([r['query'] for r in rows]), ne).float()
    return x, valid, q, torch.tensor([r['label'] for r in rows])


def score(model, tensors):
    with torch.no_grad():
        logits = model(*tensors[:3])
        return float(torch.nn.functional.cross_entropy(logits, tensors[3])), float((logits.argmax(-1) == tensors[3]).float().mean())


def latency(model, ne, nv):
    # One request, fixed rotating event workload, pre-encoded inputs; no parsing/LLM.
    events = torch.eye(ne + nv)[:ne]
    events[:, ne] = 1
    q = torch.eye(ne)[0:1]
    writes, reads = [], []
    state = torch.zeros(1, model.width)
    with torch.no_grad():
        for i in range(600):
            start = time.perf_counter_ns(); state = model.write(state, events[i % ne:i % ne + 1]); end = time.perf_counter_ns()
            _ = model.read(state, q); done = time.perf_counter_ns()
            if i >= 100:
                writes.append((end-start)/1000); reads.append((done-end)/1000)
    return dict(write_us_p50=float(np.median(writes)), write_us_p95=float(np.percentile(writes,95)),
                read_us_p50=float(np.median(reads)), read_us_p95=float(np.percentile(reads,95)),
                raw_write_us=writes, raw_read_us=reads)


def main():
    if OUT.exists():
        raise SystemExit('Run directory exists; preserve frozen results.')
    OUT.mkdir(parents=True)
    torch.set_num_threads(1)
    rows, entities, values, hashes = prepare(); ne, nv = len(entities), len(values)
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    tensors = {s: tensorize(r, ne, nv) for s, r in splits.items()}
    signature, archive = snapshot_source()
    save('protocol.json', dict(source=signature, archive=archive, runtime=runtime_info(), hashes=hashes,
         entities=entities, values=values, shared_dictionary_utf8_bytes=len(json.dumps([entities,values]).encode()),
         examples={s:len(r) for s,r in splits.items()}, seeds=[11,23,42], widths=[8,32],
         epochs=200, early_stop_patience=40, batch=128, lr=.003, checkpoint='minimum validation cross entropy',
         pilot='v1 retained: 20 epochs left train/validation losses still falling. v2 raises common training ceiling; test is exploratory, not blind.',
         scope='Public qa1 diagnostic, previously inspected test; not independent blind benchmark. No observation noise or world forecasting.',
         capacity='float32 hidden payload only; shared weights separately counted; exact table uint8 per entity, 255 unknown. No history reader.',
         timing='CPU single thread eager batch1 pre-encoded microbenchmark; excludes parsing, token generation and serialization; no GPU or API.',
         acceptance='Efficiency advantage requires test accuracy within 1 percentage point of exact table, and lower measured resource use. No posthoc threshold tuning.'))
    save('examples.json', rows)
    results = []; predictions = []; timing = {}
    # Strong exact typed sufficient statistic, not an oracle: consumes only events.
    for row in splits['test']:
        state = bytearray([255] * ne)
        for owner, value in row['events']: state[owner] = value
        restored = bytes(state)
        pred = restored[row['query']]
        predictions.append(dict(id=row['id'], story=row['story'], method='exact_table', label=row['label'], prediction=pred))
    state = bytearray([255] * ne); w=[]; r=[]
    for i in range(600):
        owner=i%ne; value=i%nv
        a=time.perf_counter_ns(); state[owner]=value; b=time.perf_counter_ns(); _=state[owner]; c=time.perf_counter_ns()
        if i>=100: w.append((b-a)/1000); r.append((c-b)/1000)
    # Bulk amortization avoids a zero median from clock resolution for tiny table ops.
    bulk_w=[]; bulk_r=[]
    for _ in range(30):
        a=time.perf_counter_ns()
        for j in range(10000): state[j%ne]=j%nv
        b=time.perf_counter_ns()
        for j in range(10000): unused=state[j%ne]
        c=time.perf_counter_ns()
        bulk_w.append((b-a)/1e7); bulk_r.append((c-b)/1e7)
    timing['exact_table']=dict(raw_write_us=w,raw_read_us=r,bulk_write_us=bulk_w,bulk_read_us=bulk_r,
                               write_us_p50=float(np.median(bulk_w)),read_us_p50=float(np.median(bulk_r)),
                               write_us_p95=float(np.percentile(bulk_w,95)),read_us_p95=float(np.percentile(bulk_r,95)))
    results.append(dict(method='exact_table', state_bytes=ne, weight_bytes=0, accuracy=np.mean([p['prediction']==p['label'] for p in predictions]),
                        **{k:v for k,v in timing['exact_table'].items() if k.endswith(('p50','p95'))}))
    for kind in ('input_flow','gru','ltc'):
        for width in (8,32):
            for seed in (11,23,42):
                torch.manual_seed(seed); model=BoundedCodec(kind,width,ne,nv)
                optimizer=torch.optim.Adam(model.parameters(),lr=.003)
                history=[]; best=float('inf'); best_state=None; stale=0; start=time.perf_counter()
                for epoch in range(200):
                    model.train(); order=torch.randperm(len(splits['train'])); train_loss=[]
                    for batch in order.split(128):
                        x,v,q,y=(t[batch] for t in tensors['train'])
                        optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x,v,q),y)
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); train_loss.append(float(loss.detach()))
                    model.eval(); vn,va=score(model,tensors['validation'])
                    history.append(dict(epoch=epoch+1,train_nll=float(np.mean(train_loss)),val_nll=vn,val_accuracy=va))
                    if vn<best:
                        best=vn; best_state=copy.deepcopy(model.state_dict()); stale=0
                    else: stale+=1
                    if stale>=40: break
                elapsed=time.perf_counter()-start; model.load_state_dict(best_state); model.eval()
                name=f'{kind}-{width}-{seed}'; torch.save(best_state,OUT/f'{name}.pt'); save(f'{name}-epochs.json',history)
                test_nll,test_acc=score(model,tensors['test'])
                # Independent sequential replay then serialize. Reader receives ONLY
                # restored state and query. Compare against batched prediction.
                with torch.no_grad():
                    batched=model(*tensors['test'][:3]).argmax(-1).tolist()
                    for index,row in enumerate(splits['test']):
                        state=torch.zeros(1,width)
                        for owner,value in row['events']:
                            event=torch.zeros(1,ne+nv); event[0,owner]=1; event[0,ne+value]=1
                            state=model.write(state,event)
                        payload=pack_state(state); assert len(payload)==4*width
                        restored=unpack_state(payload,width)
                        q=torch.nn.functional.one_hot(torch.tensor([row['query']]),ne).float()
                        probs=model.read(restored,q).softmax(-1)[0]; pred=int(probs.argmax())
                        assert pred==batched[index], 'State-only replay disagrees'
                        predictions.append(dict(id=row['id'],story=row['story'],method=name,label=row['label'],prediction=pred,
                                                probabilities=probs.tolist(),events=len(row['events'])))
                timing[name]=latency(model,ne,nv)
                result=dict(method=name,kind=kind,width=width,seed=seed,state_bytes=4*width,
                            weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),accuracy=test_acc,nll=test_nll,
                            train_seconds=elapsed,epochs=len(history),checkpoint_file_bytes=(OUT/f'{name}.pt').stat().st_size,process_rss_mb=rss_mb(),
                            **{k:v for k,v in timing[name].items() if not k.startswith('raw')})
                results.append(result); save('summary.json',results); save('timing.json',timing)
                print(name, round(test_acc,4), f'{elapsed:.1f}s training',flush=True)
    with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in predictions: f.write(json.dumps(row)+'\n')
    save('complete.json',dict(models=18,predictions=len(predictions),state_only_replay_passed=True,api_requests=0))


if __name__=='__main__': main()

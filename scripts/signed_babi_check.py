"""Native public signed-evidence operator check, not learned-model advantage."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,re
import numpy as np
import torch
from src.config import resolve,atomic_json
from src.data.babi_signed import assertion,question
from src.memory.liquid_belief import LiquidBelief,INPUTS
from src.provenance import snapshot_source,utc_now


def main():
    root=resolve('runs/signed-belief-v2');root.mkdir(exist_ok=True)
    if (root/'summary.json').exists():print('Completed signed check preserved');return
    signature,archive=snapshot_source()
    atomic_json(root/'protocol.json',dict(source_signature=signature,source_archive=archive,created_at=utc_now(),
        data=json.loads(resolve('data/babi/signed_provenance.json').read_text()),
        design='All native en1k qa9/qa10 train then test. Ontology from train statements only. Full signed likelihood replacement compared with positive-only updates. No learning, no noise or missingness, no time evolution; NOT evidence for LTC superiority. Training audit found11 qa10 labels answering maybe for eitherXorX. Keep logical set semantics, count ALL official labels in primary accuracy; report degenerate-disjunction subset separately without relabeling.'))
    torch.manual_seed(0);model=LiquidBelief();rows=[];summaries=[]
    with torch.no_grad():
        for task in ('qa9_simple-negation','qa10_indefinite-knowledge'):
            train=resolve(f'data/babi/{task}_train.txt').read_text().splitlines()
            rooms=sorted({v for line in train if '\t' not in line for v in assertion(line.split(' ',1)[1])[2]})
            assert len(rooms)==6
            for split in ('train','test'):
                for strategy in ('positive_only','signed_observation'):
                    states={};clauses={};story=-1;condition=[]
                    for line in resolve(f'data/babi/{task}_{split}.txt').read_text().splitlines():
                        n,text=line.split(' ',1);n=int(n)
                        if n==1:states={};clauses={};story+=1
                        if '\t' in text:
                            q,answer,support=text.split('\t');owner,place=question(q)
                            p=states.get(owner,model.initial(1))[1][0];value=float(p[rooms.index(place)])
                            predicted='yes' if value>1-1e-5 else 'no' if value<1e-5 else 'maybe'
                            kind,places=clauses.get(owner,('',()))
                            degenerate=kind=='either' and len(set(places))==1 and place in places
                            condition.append(dict(task=task,split=split,strategy=strategy,story=story,line=n,question=q,degenerate_disjunction=degenerate,
                                answer=answer,prediction=predicted,probabilities=dict(zip(rooms,p.tolist()))))
                        else:
                            owner,kind,places=assertion(text)
                            clauses[owner]=(kind,places)
                            if strategy=='positive_only' and kind!='positive':continue
                            h,p,c=states.get(owner,model.initial(1))
                            event=torch.zeros(1,INPUTS)
                            if kind=='negative':event[0,:6]=1
                            for place in places:event[0,rooms.index(place)]=0 if kind=='negative' else 1
                            event[0,6+('positive','negative','either').index(kind)]=1;event[0,9:11]=1
                            # A newly asserted current location set replaces the
                            # old one. 'Not X' does NOT mean retain every earlier
                            # exclusion forever across unobserved movements.
                            states[owner]=model.observe(h,p,event,torch.tensor([True]))
                    if strategy=='signed_observation' and split=='train' and task.startswith('qa9_'):assert all(r['prediction']==r['answer'] for r in condition)
                    ordinary=[r for r in condition if not r['degenerate_disjunction']]
                    rows.extend(condition);summaries.append(dict(task=task,split=split,strategy=strategy,questions=len(condition),accuracy=sum(r['prediction']==r['answer'] for r in condition)/len(condition),
                        degenerate_disjunctions=sum(r['degenerate_disjunction'] for r in condition),ordinary_questions=len(ordinary),ordinary_accuracy=sum(r['prediction']==r['answer'] for r in ordinary)/len(ordinary)))
    (root/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(root/'summary.json',dict(conditions=summaries,api_calls=0,note='Evidence operator correctness only, no trained neural advantage.'))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

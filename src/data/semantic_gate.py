"""Public pair labels and strictly source-grounded update challenge.

MemOps supplies privileged target binding and trigger selection for evaluation;
models see text pairs only. Absence of an operation is NEVER a duplicate label.
"""
import hashlib,json,re,difflib
from collections import Counter
from pathlib import Path
import numpy as np


def stable_key(text):return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):return ' '.join(str(text).lower().split())


def sentence_groups(rows):
    """Keep all pairs connected by a shared normalized sentence together."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for row in rows:
        a,b=normalize(row['text1']),normalize(row['text2']);ra,rb=find(a),find(b)
        if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
    return [stable_key(find(normalize(r['text1']))) for r in rows]


def paws_rows(root):
    import pyarrow.parquet as pq
    output=[]
    for split,name,count in [('train','train',6000),('validation','validation',None),('test','test',None)]:
        rows=pq.read_table(Path(root)/f'{name}.parquet').to_pylist()
        rows.sort(key=lambda r:stable_key(f'{split}:{r["id"]}'))
        if count:rows=rows[:count]
        groups=sentence_groups([{'text1':r['sentence1'],'text2':r['sentence2']} for r in rows])
        for r,group in zip(rows,groups):
            part=('selection' if int(group[:8],16)%2==0 else 'calibration') if split=='validation' else split
            output.append(dict(id=f'paws:{split}:{r["id"]}',split=part,dataset='paws',
                               group=group,text1=r['sentence1'],text2=r['sentence2'],label=int(r['label'])))
    return output


def update_rows(root):
    output=[];audit=[]
    for path in sorted(Path(root).glob('*.json')):
        obj=json.loads(path.read_text(encoding='utf-8'));turns={}
        for segment in obj['conversations']:
            for i,turn in enumerate(segment['dialogue'],1):turns[(int(segment['segment_index']),i)]=turn
        current={}
        operations=sorted(obj.get('operations',[]),key=lambda op:(int(op['trigger_span']['segment_index']),int(op['trigger_span']['turn_index']),str(op.get('operation_id',''))))
        for op in operations:
            trigger=op['trigger_span'];pos=(int(trigger['segment_index']),int(trigger['turn_index']))
            record=dict(file=path.name,operation_id=op.get('operation_id'),type=op.get('type'),validity=op.get('validity'))
            target=op.get('target',{}).get('target_id');quote=trigger.get('quote','');turn=turns.get(pos,{})
            if not target or not quote or turn.get('role')!='user' or quote not in turn.get('content',''):
                record['status']='unverified_trigger';audit.append(record);continue
            valid=op.get('validity')=='confirmed';kind=op.get('type','').lower()
            if kind=='update' and valid:
                previous=current.get(target)
                if previous is None:record['status']='missing_confirmed_predecessor'
                elif normalize(previous['value'])!=normalize(op.get('old_value')):record['status']='old_value_chain_mismatch'
                elif normalize(op.get('old_value'))==normalize(op.get('new_value')):record['status']='no_changed_value'
                else:
                    output.append(dict(id=f'memops:{path.stem}:{op.get("operation_id")}',split='challenge',dataset='memops',
                        group=path.stem,text1=previous['quote'],text2=quote,label=0,
                        provenance=dict(file=path.name,target=target,old_value=op.get('old_value'),new_value=op.get('new_value'),
                                        old_position=previous['position'],new_position=pos,old_quote=previous['quote'],new_quote=quote),
                        label_basis='Upstream confirmed update, changed value, matched earlier confirmed value, both user quotes verified verbatim; not human-adjudicated.'))
                    record['status']='included_confirmed_revision'
            else:record['status']='not_confirmed_changed_update'
            audit.append(record)
            if valid and kind in ('remember','update'):
                current[target]=dict(value=op.get('new_value'),quote=quote,position=pos)
            elif valid and kind=='forget':current.pop(target,None)
    return output,audit


def lexical_features(a,b):
    x=re.findall(r"\w+",a.lower());y=re.findall(r"\w+",b.lower());sx,sy=set(x),set(y)
    bx,by=set(zip(x,x[1:])),set(zip(y,y[1:]))
    nums=lambda s:set(re.findall(r'\d+(?:[.,]\d+)*',s))
    neg={'not','no','never','without','neither','nor'}
    return np.asarray([float(normalize(a)==normalize(b)),len(sx&sy)/max(len(sx|sy),1),
        len(bx&by)/max(len(bx|by),1),min(len(x),len(y))/max(len(x),len(y),1),
        float(nums(a)!=nums(b)),float(bool(sx&neg)!=bool(sy&neg)),
        difflib.SequenceMatcher(None,x,y,autojunk=False).ratio()],dtype=np.float32)


def overlap_audit(rows):
    pools={s:[r for r in rows if r['split']==s] for s in ('train','selection','calibration','test','challenge')}
    seen_pairs={};seen_sentences={};result={}
    for split,group in pools.items():
        pair_overlap=0;sentence_overlap=0
        for row in group:
            pair=tuple(sorted((normalize(row['text1']),normalize(row['text2']))))
            pair_overlap+=pair in seen_pairs
            sentence_overlap+=any(normalize(row[k]) in seen_sentences for k in ('text1','text2'))
        result[split]=dict(rows=len(group),pairs_seen_in_previous_splits=pair_overlap,rows_with_sentence_seen_before=sentence_overlap)
        for row in group:
            seen_pairs[tuple(sorted((normalize(row['text1']),normalize(row['text2']))))]=split
            for k in ('text1','text2'):seen_sentences[normalize(row[k])]=split
    return result

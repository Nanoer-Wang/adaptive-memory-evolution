"""Materialize explicit assistant review decisions; NOT a reusable automatic judge.

The decisions below were made after reading the actual saved predictions and
references. Every output carries its answer hash. Raw provider grades stay intact.
"""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import utc_now
OUT=ROOT/'runs/task-utility-v1'
ALL={'linear','liquid','gru','risk_liquid','full_reference','linear_addback','linear_retained_control','liquid_addback','liquid_retained_control'}
DECISIONS={
    '6aeb4375':({'linear_addback','full_reference'},'Latest cumulative count is4. Liquid addback incorrectly sums older3 and newer4 to7; other contexts abstain.'),
    '184da446':(ALL,'All frozen answers identify latest page220.'),
    '2311e44b':(ALL-{'linear_addback'},'440 minus250 gives190. Linear addback refuses despite stating both required facts; no truncation.'),
    '6c49646a':({'linear','liquid','gru','risk_liquid'},'Three trips1800 plus distinct Yellowstone1200 gives3000. Supplements either abstain or truncate without delivering total; full reference abstains.'),
    '51b23612':(set(),'All responses abstain; none gives Nu,pogodi. Gold name was removed by common compression before memory writes.'),
    'c8f1aeed':(set(),'All responses abstain; none gives Pennsylvania. Original judge incorrectly accepts linear retained-control refusal.'),
    '54026fce':({'linear_addback','liquid_addback','full_reference'},'Only these responses offer remote-work social connection suggestions tied to earlier virtual coffee discussions. Pure refusals do not satisfy the answerable preference request.'),
    '0a34ad58':(ALL-{'risk_liquid'},'Responses personalize Tokyo guidance using the already obtained Suica card; most also mention TripIt. Sparse liquid abstains. Suica-only answers satisfy a permissive personalization reading but are rubric-sensitive; no real-world travel-fact audit.'),
    '8e9d538c':(ALL-{'risk_liquid'},'All other frozen answers give17 skeins; sparse liquid abstains.'),
    '51a45a95':({'gru','linear_addback','liquid_addback'},'These answers identify Target. Remaining answers refuse, including unlimited-store reference despite mentioning Target in discussion.'),
    'gpt4_8279ba03':(set(),'No frozen response identifies the smoker; all abstain.'),
    'eac54add':(set(),'No response commits to the first-client contract answer. Sparse liquid mentions it then rejects it using erroneous time arithmetic (March1 to March28 is27days, not5weeks); others abstain or truncate.'),
}

def main():
    records=[]
    for p in sorted((OUT/'qa').glob('*.json')):
        r=json.loads(p.read_text(encoding='utf-8'));q=r['question_id']
        if q not in DECISIONS or 'judge_response' not in r:continue
        positives,reason=DECISIONS[q]
        records.append(dict(question_id=q,method=r['method'],answer_sha256=hashlib.sha256(r['hypothesis'].encode()).hexdigest(),
            original_judge_correct=r['custom_judge_correct'],review_correct=float(r['method'] in positives),reason=reason,
            rubric_sensitive=q=='0a34ad58' and r['method'] in {'linear_addback','liquid_addback','full_reference'}))
    atomic_json(OUT/'reference_audit.json',dict(reviewer='Current assistant; explicit reference-based review, not independent human or blinded adjudication',
        timestamp=utc_now(),scope='Each saved generation in reviewed questions; no answer regeneration or raw grade edits. Final task answer judged, not all explanatory facts.',
        preference_rule='Personalized actionable advice using an existing user resource counts; mention of every illustrative reference example not mandatory. Mark Suica-only cases as rubric-sensitive.',records=records))
    print('Reviewed assignments',len(records),'disagreements',sum(r['original_judge_correct']!=r['review_correct'] for r in records))
    extra=[]
    if (OUT/'length_sensitivity/complete.json').exists():
        positives={'6c49646a':ALL-{'linear','liquid','gru'},'eac54add':{'risk_liquid'}}
        for p in sorted((OUT/'length_sensitivity').glob('*.json')):
            if p.name in {'selection.json','complete.json','summary.json'}:continue
            r=json.loads(p.read_text(encoding='utf-8'))
            extra.append(dict(question_id=r['question_id'],method=r['method'],answer_sha256=hashlib.sha256(r['hypothesis'].encode()).hexdigest(),original_judge_correct=r['custom_judge_correct'],
                review_correct=float(r['method'] in positives[r['question_id']]),
                reason='Road-trip answers must commit to3000, not ultimately refuse; business milestone must identify first-client contract, not refuse. All18 expanded predictions inspected.'))
        atomic_json(OUT/'length_reference_audit.json',dict(reviewer='Current assistant, explicit reference review, not blinded/human',records=extra))

if __name__=='__main__':main()

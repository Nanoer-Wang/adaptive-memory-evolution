"""Interventions must not silently replace the original memory context."""
import json
from pathlib import Path
from scripts.task_utility import supplement_indices
from scripts.report_task_utility import pair

def test_addback_is_evicted_control_is_retained_and_length_matched():
    texts=['a'*10,'b'*80,'c'*40,'d'*20,'e'*60,'f'*30]
    added,control=supplement_indices([0,1,2,3],[0],[4,5,1,2,3,0],texts)
    assert added==[4,5]
    assert set(control)=={1,3}  # 100 chars closest to 90 among retained pairs.
    assert not(set(added)&{0,1,2,3})
    assert not(set(control)&{0})

def test_no_retained_control_means_no_unmatched_intervention():
    assert supplement_indices([0],[0],[1,0],['a','b'])==([],[])

def test_pair_uses_questions_not_separate_method_samples():
    rows=[dict(question_id=q,method=m,custom_judge_correct=y) for q,m,y in [('q1','a',1),('q1','b',0),('q2','a',0),('q2','b',1),('q3','a',None),('q3','b',1)]]
    p=pair(rows,'a','b')
    assert p['n']==2 and p['delta_accuracy']==0 and p['wins']==p['losses']==1
    assert p['exact_discordant_p']==1

def test_frozen_task_artifacts_preserve_context_and_budget():
    path=Path(__file__).resolve().parents[1]/'runs/task-utility-v1/qa_tasks.json'
    if not path.exists():
        import pytest
        pytest.skip('Optional experiment artifacts absent')
    tasks=json.loads(path.read_text(encoding='utf-8'));by={(t['question_id'],t['method']):t for t in tasks}
    assert len(tasks)==108
    for t in tasks:
        if t['budget'] is not None:assert t['payload_bytes']<=t['budget']
        base=by[t['question_id'],t['base_method']]
        assert t['context'].startswith(base['context'])
        assert len(t['selected_ids'])==len(set(t['selected_ids']))
        if t['intervention']=='addback':assert not(set(t['added_ids'])&set(t['kept_ids']))
        if t['intervention']=='retained_control':
            assert set(t['added_ids'])<=set(t['kept_ids'])
            assert not(set(t['added_ids'])&set(base['selected_ids']))
            assert len(t['added_ids'])==len(by[t['question_id'],t['base_method']+'_addback']['added_ids'])

import json
import numpy as np
import torch
from src.models.semantic_gate import SemanticGate,calibrate
from src.data.semantic_gate import update_rows,lexical_features,sentence_groups


def test_calibration_counts_dangerous_skips_not_total_errors():
    labels=np.r_[np.zeros(1000),np.ones(1000)]
    scores=np.r_[np.full(1000,.2),np.full(1000,.9)]
    result=calibrate(scores,labels)
    assert result['feasible'] and result['errors']==0 and result['skip_count']==1000
    # Indistinguishable scores cannot safely skip either class.
    result=calibrate(np.full(2000,.9),labels)
    assert result['skip_count']==0


def test_models_are_pair_local_and_have_finite_gradients():
    pair=torch.nn.functional.normalize(torch.randn(3,2,384),dim=-1)
    lex=torch.zeros(3,7)
    for kind in ('mlp','gru','input_flow','ltc'):
        model=SemanticGate(kind)
        logits=model(pair,lex)
        assert logits.shape==(3,)
        assert torch.allclose(model(pair[:1],lex[:1]),logits[:1],atol=1e-6)
        logits.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_source_grounding_and_unconfirmed_update_not_applied(tmp_path):
    def op(i,kind,valid,old,new,quote):
        return dict(operation_id=str(i),type=kind,validity=valid,target={'target_id':'city'},old_value=old,new_value=new,
                    trigger_span={'segment_index':1,'turn_index':i,'quote':quote})
    obj=dict(conversations=[dict(segment_index=1,dialogue=[
        {'role':'user','content':'I live in Paris.'},{'role':'user','content':'I might move to Rome.'},
        {'role':'user','content':'I moved to Lyon.'},{'role':'user','content':'No such quote.'}])],operations=[
        op(1,'remember','confirmed',None,'Paris','I live in Paris.'),
        op(2,'update','tentative','Paris','Rome','I might move to Rome.'),
        op(3,'update','confirmed','Paris','Lyon','I moved to Lyon.'),
        op(4,'update','confirmed','Lyon','London','Hallucinated source')])
    (tmp_path/'sample.json').write_text(json.dumps(obj))
    rows,audit=update_rows(tmp_path)
    assert len(rows)==1 and rows[0]['text1']=='I live in Paris.' and rows[0]['label']==0
    assert any(r['status']=='unverified_trigger' for r in audit)


def test_lexical_guard_detects_changed_number_and_negation():
    f=lexical_features('Budget is 100, available.','Budget is not 200, available.')
    assert f[4]==1 and f[5]==1


def test_transitive_shared_sentences_stay_in_one_partition():
    rows=[{'text1':'a','text2':'b'},{'text1':'c','text2':'d'},{'text1':'b','text2':'c'},{'text1':'x','text2':'y'}]
    groups=sentence_groups(rows)
    assert groups[0]==groups[1]==groups[2] and groups[3]!=groups[0]

import numpy as np
import torch
from src.models.evidence_history import protected_decision,HistoryGate

def test_history_cannot_override_contradiction_or_uncertainty():
    assert not protected_decision([.99,.005,.005],[0,.99,.01],1,1,.5)
    assert not protected_decision([.01,.5,.49],[.01,.99,0],1,1,.5)
    assert protected_decision([.01,.98,.01],[.01,.98,.01],.9,.9,.5)

def test_exact_repeat_and_invalid_evidence():
    assert protected_decision([1,0,0],[1,0,0],0,0,1,seen=True)
    assert not protected_decision([np.nan,0,0],[0,1,0],1,1,0,seen=True)

def test_causal_prefix_and_static_reset():
    torch.manual_seed(1);x=torch.rand(2,5,8)
    for kind in ['static','gru','liquid']:
        m=HistoryGate(kind)
        assert torch.allclose(m(x)[:,:3],m(x[:,:3]),atol=1e-6)
        if kind=='static':assert torch.allclose(m(x),m(x,reset=True))

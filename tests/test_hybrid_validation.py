import numpy as np
import torch
from scripts.hybrid_validation import hybrid
from scripts.content_fallback import fallback_scores
from src.memory.content_store import ContentWriter


def test_frozen_hybrid_reproduces_previous_policy():
    torch.manual_seed(9);m=ContentWriter().eval();v=torch.randn(11,384);dt=torch.rand(11);w=np.random.default_rng(9).normal(size=385).astype(np.float32)
    with torch.no_grad():
        a,ac=hybrid(m,v,dt,w,'hybrid');b,bc=fallback_scores(m,v,dt,w,True)
        assert ac==bc and np.array_equal(a,b)
        altered=v.clone();altered[7:]+=10
        c,_=hybrid(m,altered,dt,w,'hybrid');assert np.array_equal(a[:7],c[:7])


def test_ablations_leave_unscored_current_content_unchanged():
    torch.manual_seed(9);m=ContentWriter().eval();v=torch.randn(11,384);dt=torch.rand(11);w=np.random.default_rng(9).normal(size=385).astype(np.float32)
    with torch.no_grad():
        ref,calls=hybrid(m,v,dt,w,'hybrid')
        for mode in ['reset','zero_time']:
            result,n=hybrid(m,v,dt,w,mode)
            assert n==calls and np.array_equal(result[1::2],ref[1::2])
        _,shift_calls=hybrid(m,v,dt,w,'shift');assert calls==6 and shift_calls==5

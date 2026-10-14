import numpy as np
import torch
import pytest
from src.memory.bounded_codec import BoundedCodec, pack_state, unpack_state


@pytest.mark.parametrize('kind', ['input_flow', 'gru', 'ltc'])
def test_state_is_sufficient_and_query_does_not_write(kind):
    torch.manual_seed(1)
    model = BoundedCodec(kind, 8)
    x = torch.zeros(1, 3, 10); x[0,:,0] = 1; x[0,:,4] = 1
    valid = torch.ones(1,3,dtype=torch.bool)
    q = torch.eye(4)[0:1]
    h = torch.zeros(1,8)
    for e in x.unbind(1): h = model.write(h,e)
    payload = pack_state(h)
    assert len(payload) == 32
    restored = unpack_state(payload,8)
    assert torch.equal(h,restored)
    assert torch.allclose(model(x,valid,q),model.read(restored,q))
    before = pack_state(restored)
    model.read(restored,torch.eye(4)[1:2])
    assert pack_state(restored) == before


def test_wrong_capacity_is_rejected():
    with pytest.raises(ValueError): unpack_state(bytes(31),8)

import numpy as np
import torch
import pytest
from src.memory.addressed_codec import AddressedCodec, GuardedWriter


@pytest.mark.parametrize('kind', ['gru', 'ltc', 'input_flow'])
def test_unrelated_entity_is_bitwise_preserved(kind):
    model = AddressedCodec(kind, 8)
    state = torch.randn(1,4,8)
    before = model.read(state, torch.tensor([2])).detach().clone()
    new = model.write(state, torch.tensor([1]), torch.eye(6)[0:1])
    assert torch.equal(state[:,[0,2,3]], new[:,[0,2,3]])
    assert torch.equal(before, model.read(new, torch.tensor([2])).detach())
    assert torch.equal(new,model.unpack(model.pack(new)))


def test_guard_skips_only_same_entity_same_value_and_restores_all_state():
    model = AddressedCodec('gru', 2)
    writer = GuardedWriter(model,True)
    owner = torch.tensor([0]); value = torch.eye(6)[1:2]
    assert writer.write(0,1,owner,value)
    payload = writer.pack()
    assert len(payload) == writer.payload_bytes == 36
    assert not writer.write(0,1,owner,value)
    assert writer.pack() == payload
    assert writer.write(1,1,torch.tensor([1]),value)
    assert writer.write(0,2,owner,torch.eye(6)[2:3])
    restored = GuardedWriter(model,True); restored.restore(payload)
    assert restored.pack() == payload
    assert not restored.write(0,1,owner,value)
    assert restored.writes == 0


def test_guard_does_not_receive_answers_and_query_cannot_write():
    model=AddressedCodec('ltc',2); writer=GuardedWriter(model,True)
    writer.write(0,1,torch.tensor([0]),torch.eye(6)[1:2])
    before=writer.pack()
    _=model.read(writer.state,torch.tensor([0]))
    assert writer.pack()==before

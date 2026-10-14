import copy
import numpy as np
import torch
from scripts.liquid_attribution import score
from src.memory.content_store import ContentWriter


def test_time_ablation_is_same_weight_fixed_time():
    torch.manual_seed(34)
    model = ContentWriter()
    fixed = copy.deepcopy(model)
    fixed.cell.fixed_time = True
    vectors, dt = torch.randn(8, 384), torch.arange(8).float()
    assert torch.allclose(score(model, vectors, dt, 'zero_time'), fixed(vectors, dt))
    assert not torch.allclose(score(model, vectors, dt, 'normal'), fixed(vectors, dt))


def test_reset_removes_prefix_dependency_without_removing_current_time():
    torch.manual_seed(34)
    model = ContentWriter()
    vectors, dt = torch.randn(8, 384), torch.arange(8).float()
    modified = vectors.clone()
    modified[:4] += 10
    assert torch.allclose(score(model, vectors, dt, 'reset')[4:], score(model, modified, dt, 'reset')[4:])
    assert not torch.allclose(score(model, vectors, dt, 'normal')[4:], score(model, modified, dt, 'normal')[4:])
    assert not torch.allclose(score(model, vectors, dt, 'reset'), score(model, vectors, torch.zeros_like(dt), 'reset'))


def test_normal_writer_has_no_future_access():
    torch.manual_seed(34)
    model = ContentWriter()
    vectors, dt = torch.randn(8, 384), torch.arange(8).float()
    modified = vectors.clone()
    modified[4:] += 10
    assert torch.equal(score(model, vectors, dt, 'normal')[:4], score(model, modified, dt, 'normal')[:4])

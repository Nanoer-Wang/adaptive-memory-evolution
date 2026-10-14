import numpy as np  # Initialize MKL before torch in this Windows environment.
import torch
from src.models.temporal_forecast import TemporalForecast

def test_masked_values_cannot_affect_prediction():
    torch.manual_seed(2);x=torch.randn(3,6,11);v=torch.ones(3,6,dtype=torch.bool);v[:,2]=False;dt=torch.ones(3,6);gap=torch.ones(3)*6
    altered=x.clone();altered[:,2]=999
    for kind in ['gru','liquid','input_flow','last_mlp']:
        m=TemporalForecast(kind);assert torch.allclose(m(x,v,dt,gap),m(altered,v,dt,gap))

def test_no_history_gru_matches_reset_when_single_observation():
    x=torch.randn(2,1,11);v=torch.ones(2,1,dtype=torch.bool);dt=torch.ones(2,1);gap=torch.ones(2)*6
    for kind in ['gru','liquid']:
        m=TemporalForecast(kind);assert torch.allclose(m(x,v,dt,gap),m(x,v,dt,gap,reset=True))

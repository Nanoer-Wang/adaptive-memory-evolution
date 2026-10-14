import numpy as np
import pytest
from src.memory.retained_observation import retained_observation


class Encoder:
    def encode(self, texts):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))


def test_fixed_view_is_nested_and_tracks_omitted_number():
    full='[session=a date=2023/01/01]\nuser: I own 4 bikes.\n[session=b date=2023/02/01]\nuser: I now own 7 bikes.'
    kept='[session=a date=2023/01/01]\nuser: I own 4 bikes.'
    obs=retained_observation('How many bikes?',full,Encoder(),kept)
    assert obs['context']==kept
    assert obs['selected_unit_indices']==[0]
    assert obs['x'][1][4]==0
    assert obs['x'][1][8]==1
    assert len(obs['x'][0])==18


def test_no_invented_or_reassigned_evidence():
    full='[session=a date=2023/01/01]\nuser: I own 4 bikes.'
    for kept in [full.replace('4','5'),full.replace('user:','assistant:')]:
        with pytest.raises(AssertionError):
            retained_observation('bikes',full,Encoder(),kept)

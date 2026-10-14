import numpy as np
import pytest
from src.data.longmemeval import date_days, public_history, encode, load_rows, prepare, prepared
from src.training.train import positive_indices


def test_minute_precision_preserves_one_minute_and_midnight(tiny_config):
    a='2023/01/01 (Sun) 23:59'
    b='2023/01/02 (Mon) 00:00'
    assert (date_days(b,'minute')-date_days(a,'minute'))*1440 == pytest.approx(1,abs=1e-6)
    assert date_days('2023-01-01T12:34:56','minute') > date_days('2023-01-01T12:34:00','minute')
    row=load_rows(tiny_config['data']['file'])[0]
    row['haystack_dates']=['2023/01/01 (Sun) 10:01','2023/01/01 (Sun) 10:00']
    tiny_config['model']['timestamp_precision']='minute'
    h=public_history(row,'minute')
    assert h.session_ids==tuple(reversed(row['haystack_session_ids']))
    assert positive_indices(row,tiny_config)==[1]
    delta=encode(h,tiny_config['model'])[2]
    assert delta[1]==pytest.approx(1/1440/30,rel=1e-6)
    assert np.isfinite(delta).all() and delta[0]==0
    assert date_days(row['haystack_dates'][0])==date_days(row['haystack_dates'][1])


def test_timestamp_configuration_is_versioned_and_split_mismatch_rejected(tiny_config):
    tiny_config['model']['timestamp_precision']='minute'
    meta=prepare(tiny_config)
    assert meta['signature']['features_version']==2
    prepared(tiny_config)
    tiny_config['data']['seed']+=1
    with pytest.raises(ValueError,match='split configuration'):
        prepared(tiny_config)


def test_invalid_clock_rejected_and_date_only_accepted():
    with pytest.raises(ValueError):
        date_days('2023/01/01 (Sun) 25:01','minute')
    assert date_days('2023/01/01','minute')==date_days('2023/01/01')

import numpy as np
from scripts.addressed_transfer_study import parse
import pytest


def test_possession_is_explicit_noop_not_silently_unparsed(tmp_path):
    path=tmp_path/'tiny.txt'
    path.write_text('1 Mary moved to the kitchen.\n2 Mary got the milk there.\n3 Is Mary in the kitchen?\tyes\t1\n')
    rows=parse(path,['Mary'],['kitchen'],True)
    assert [e['type'] for e in rows[0]]==['event','nonlocation','query']
    assert rows[0][-1]['target']==0
    path.write_text('1 Mary teleported beyond the universe.\n')
    with pytest.raises(ValueError):parse(path,['Mary'],['kitchen'],True)

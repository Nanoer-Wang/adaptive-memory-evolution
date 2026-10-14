from src.data.babi_events import statement, read_key, stories
from src.memory.version_ledger import VersionLedger


def test_babi_causal_updates_and_equal_frozen_priority():
    first=statement('Mary moved to the kitchen.',0,1,'train')
    second=statement('Mary went back to the garden.',0,3,'train')
    ledger=VersionLedger(policy='frozen_rule');ledger.write(first)
    initial=ledger.score(ledger.items[first.identity]).item()
    assert ledger.current(read_key('Where is Mary? ')).value=='kitchen'
    ledger.write(second)
    assert ledger.current(first.key).value=='garden'
    assert ledger.score(ledger.items[first.identity]).item()==initial


def test_babi_questions_separate_from_writer(tmp_path):
    path=tmp_path/'sample.txt'
    path.write_text('1 Mary moved to the kitchen.\n2 Where is Mary?\tkitchen\t1\n1 John journeyed to the hallway.\n2 Where is John?\thallway\t1\n')
    data=stories(path,'train')
    assert len(data)==2
    assert data[0][0].quote=='Mary moved to the kitchen.'
    assert data[0][1]['support']==[1]
    assert data[0][0].identity!=data[1][0].identity

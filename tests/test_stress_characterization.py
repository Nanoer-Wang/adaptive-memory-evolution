from src.memory.stress_characterization import stream,execute
from src.memory.b1_policy import frozen_head
from src.config import ROOT
from scripts.ame_validation_batches import read

def test_stress_identical_prefixes_and_constant_shift_frequency():
    stories=read(ROOT/'runs/liquid-consolidation-v2/stories.json');vectors=read(ROOT/'runs/liquid-consolidation-v2/vectors.json')
    long=stream(stories,vectors,94001,8,5000)
    for n in (500,1000):assert stream(stories,vectors,94001,16,n)['events']==long['events'][:n]
    times=[w['input']['time'] for w in long['events'] if w['input']['kind']=='context' and w['evaluation']['phase']>=0]
    assert all(b-a==97 for a,b in zip(times,times[1:]))
    for phase in range(1,50):
        revised={w['input']['key'] for w in long['events'] if w['evaluation']['phase']==phase and w['input']['kind']=='observe' and w['evaluation']['revision']}
        assert len(revised)==4
    head=frozen_head(ROOT/'runs/evolving-memory-v1/checkpoints/learned-11.pt')
    result,check=execute(stream(stories,vectors,94001,8,500),'full_1',11,head)
    assert check['passed'] and len(result['episodes'])>3
    assert result['metrics']['final_partial_episode_excluded']
    assert all(e['query_opportunities']==16 for e in result['episodes'] if e['complete_episode'])

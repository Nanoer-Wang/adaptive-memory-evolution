import numpy as np
from src.memory.semantic_reading import semantic_views,parse_units

class ToyEncoder:
    def encode(self,texts):
        out=[]
        for text in texts:
            relevant=any(w in text.lower() for w in ['photography','camera','score'])
            out.append([1.,0.] if relevant else [0.,1.])
        return np.array(out,dtype=np.float32)

def test_semantic_reading_keeps_topic_without_exact_query_word():
    context='[session=a date=2023/01/01]\nuser: Coffee accessories are useful.\n[session=b date=2023/01/02]\nuser: My camera is a Sony.'
    views=semantic_views('photography',context,ToyEncoder(),(70,200))
    assert 'Sony' in views[70]['context']
    assert set(views[70]['selected_unit_indices'])<=set(views[200]['selected_unit_indices'])
    assert all(len(v['context'])<=budget for budget,v in views.items())
    original=set(parse_units(context))
    assert all(set(parse_units(v['context']))<=original for v in views.values())

def test_date_in_header_is_not_retained_numeric_fact():
    context='[session=a date=2023/01/01]\nuser: My score is 10.\n[session=b date=2023/01/02]\nuser: My score is 2023.'
    view=semantic_views('score',context,ToyEncoder(),(60,))[60]
    assert 'score is 2023' not in view['context']
    assert view['x'][1][8]==1.0

def test_omission_features_detect_loss_despite_embedding_equivalence():
    context='[session=a date=2023/01/01]\nuser: My score is 10.\n[session=b date=2023/01/02]\nuser: My score is 999.'
    view=semantic_views('score',context,ToyEncoder(),(60,))[60]
    assert view['x'][1][5]==1.0  # Deliberately identical semantic vectors.
    assert view['x'][1][8]==1.0  # A changed numeric fact still cannot be assumed covered.

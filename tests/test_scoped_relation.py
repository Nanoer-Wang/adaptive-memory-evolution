import numpy as np
from src.models.scoped_relation import scope_features

def test_modifier_binding_is_not_a_bag_of_words():
    a=[{('dog','amod','black'),('cat','amod','white')}]+[set() for _ in range(6)]
    b=[{('dog','amod','white'),('cat','amod','black')}]+[set() for _ in range(6)]
    assert scope_features(a,b)[0]==0
    assert scope_features(a,a)[0]==1

def test_directional_additions_remain_directional():
    a=[set() for _ in range(7)];b=[{('dog','amod','black')}]+[set() for _ in range(6)]
    assert scope_features(a,b)[1:3].tolist()==[0,1]
    assert scope_features(b,a)[1:3].tolist()==[1,0]
    assert np.isfinite(scope_features(a,a)).all()

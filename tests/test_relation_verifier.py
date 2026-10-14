import numpy as np
from src.models.relation_verifier import changed_literals,relation_scores,choose_candidate,certify


def test_mutual_entailment_and_veto():
    a=np.array([[.01,.98,.01],[.01,.98,.01],[.01,.98,.01]])
    b=np.array([[.01,.98,.01],[.1,.2,.7],[.01,.98,.01]])
    scores=relation_scores(a,b,[('He ran.','He was running.'),('Animal.','Dog.'),('Budget 100','Budget 200')])
    assert scores['nli_bidirectional'][1]==.2
    assert scores['nli_bidirectional_guarded'][2]==0
    assert changed_literals("I don't run.",'I run.')


def test_certificate_not_obtained_from_empty_or_tiny_acceptance():
    assert not certify([0]*1000,[1]*1000,.9)['approved']
    assert not certify([1]*50,[1]*50,.9)['approved']
    assert certify([1]*1000,[1]*1000,.9)['approved']
    assert not certify([1]*1000,[0]*20+[1]*980,.9)['approved']


def test_selective_error_target_is_conditional_on_skips():
    # A low false-positive rate on all negatives is not enough if the proposed
    # skips have many errors among them.
    y=np.array([0]*10000+[1]*100);s=np.array([.99]*10+[.1]*9990+[.99]*100)
    assert not choose_candidate(s,y)['eligible']

from src.models.role_relation import relation_label,source_groups,role_features
import numpy as np

def test_directional_addition_not_neutral_or_contradiction():
    assert relation_label('A_neutral_B','B_entails_A')==1
    assert relation_label('A_entails_B','B_neutral_A')==2
    assert relation_label('A_neutral_B','B_neutral_A')==4
    assert relation_label('A_contradicts_B','B_neutral_A')==-1

def test_original_caption_connects_different_surface_sentences():
    rows=[dict(sentence_A='cat runs',sentence_B='cat moves',sentence_A_original='cat outside',sentence_B_original='cat outside'),
          dict(sentence_A='a cat stops',sentence_B='cat rests',sentence_A_original='cat outside',sentence_B_original='other caption')]
    assert len(set(source_groups(rows)))==1

def test_argument_swap_changes_role_features():
    a=[dict(predicate='chase',agent=['dog'],patient=['cat'],negated=False)]
    b=[dict(predicate='chase',agent=['cat'],patient=['dog'],negated=False)]
    assert not np.array_equal(role_features(a,a),role_features(a,b))
    assert role_features(a,b)[15]==1

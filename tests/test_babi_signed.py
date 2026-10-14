from src.data.babi_signed import assertion,question


def test_signed_parser_preserves_negation_and_alternatives():
    assert assertion('Mary is no longer in the bedroom.')==('Mary','negative',('bedroom',))
    assert assertion('Fred is either in the school or the park.')==('Fred','either',('school','park'))
    assert assertion('John went back to the office.')==('John','positive',('office',))
    assert question('Is Fred in the school? ')==('Fred','school')

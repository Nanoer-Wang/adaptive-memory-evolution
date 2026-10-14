import json
from src.data.grounded_claims import messages,validate,extract


def proposal(quote,mode='asserted',**kw):
    return dict(turn=0,quote=quote,subject='user',relation='says',value='a value',mode=mode,**kw)


def check(turns,claims):
    return validate(json.dumps(dict(claims=claims)),turns,'source',1)


def test_claim_prompt_ignores_annotations_and_question():
    a=[dict(role='user',content='My coat is blue.',has_answer=True,answer='secret')]
    assert messages(a)==messages([dict(role='user',content='My coat is blue.')])


def test_grounding_requires_exact_unique_quotes_and_preserves_unverified_status():
    turns=[dict(role='user',content='My coat is blue.')]
    good,bad=check(turns,[proposal('My coat is blue.'),proposal('red coat')])
    assert len(good)==len(bad)==1
    c=good[0];assert turns[c.turn]['content'][c.start:c.end]==c.quote
    assert c.verification=='unverified_semantic_proposal'
    assert not check([dict(role='user',content='blue blue')],[proposal('blue')])[0]


def test_assistant_does_not_establish_user_fact():
    good,bad=check([dict(role='assistant',content='You could move to Paris.')],[proposal('You could move to Paris.')])
    assert not good and bad


def test_hedges_and_consideration_are_not_silently_strengthened():
    turns=[dict(role='user',content='I think it was called Art of the Ancients.')]
    good,bad=check(turns,[proposal('it was called Art of the Ancients')])
    assert not good and 'uncertainty' in bad[0]['reason']
    assert check(turns,[proposal('it was called Art of the Ancients','tentative')])[0]
    turns=[dict(role='user',content='I was thinking of visiting the museum.')]
    assert not check(turns,[proposal('visiting the museum','planned')])[0]
    # A hedge in a previous sentence must not contaminate a new assertion.
    turns=[dict(role='user',content='I think that is old. My coat is blue.')]
    assert check(turns,[proposal('My coat is blue.')])[0]


def test_format_repair_is_bounded_and_keeps_original_failure():
    class Fake:
        def __init__(self):self.calls=[]
        def complete(self,messages,**kw):
            self.calls.append((messages,kw))
            return dict(content='bad json' if len(self.calls)==1 else '{"claims": []}',finish_reason='stop')
    client=Fake();result=extract(client,[dict(role='user',content='Hello')],'s',1)
    assert result['error'] is None and len(result['responses'])==2
    assert len(client.calls)==2 and result['responses'][0]['content']=='bad json'


def test_truncated_claim_response_is_not_admitted():
    class Fake:
        def complete(self,*a,**kw):return dict(content='{"claims": []}',finish_reason='length')
    assert extract(Fake(),[dict(role='user',content='Hello')],'s',1)['error']=='truncated_response'

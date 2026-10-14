from src.memory.call_budget import CallBudget
import pytest


@pytest.mark.parametrize('rate', [0, .1, .25, .5, 1])
def test_every_prefix_stays_within_budget(rate):
    from fractions import Fraction
    from random import Random
    random = Random(23)
    limiter = CallBudget(rate)
    for i in range(1, 1001):
        requested = random.random() < .8
        granted = limiter.step(requested)
        assert not granted or requested
        assert limiter.used <= int(Fraction(str(rate)) * i)


def test_saved_allowance_and_no_borrowing():
    limiter = CallBudget(.25)
    assert [limiter.step(True) for _ in range(4)] == [False, False, False, True]
    for _ in range(8):
        assert not limiter.step(False)
    assert limiter.step(True)
    assert limiter.step(True)
    assert not limiter.step(True)

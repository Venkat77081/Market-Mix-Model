"""Every file in tests/data/phase_a_tests matches expected_phase_a.json."""
import pytest

from expected_phase_a import EXP, check


@pytest.mark.parametrize("name", sorted(EXP))
def test_matches_answer_key(name):
    _, r, checks = check(name)
    failed = [(desc, actual) for desc, ok, actual in checks if not ok]
    assert not failed, f"{name}: {failed}"

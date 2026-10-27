"""Every file in tests/data/phase_b_tests matches expected_phase_b.json and its truth file."""
import pytest

from expected_phase_b import EXP, check


@pytest.mark.parametrize("name", EXP)
def test_matches_answer_key(name):
    _, checks = check(name)
    failed = [(desc, actual) for desc, ok, actual in checks if not ok]
    assert not failed, f"{name}: {failed}"

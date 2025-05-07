from __future__ import annotations

import pytest

from analora.utils.policy import check_exist_policy, check_missing_policy

########################################
#     Tests for check_exist_policy     #
########################################


@pytest.mark.parametrize("policy", ["ignore", "raise", "warn"])
def test_check_exist_policy_valid(policy: str) -> None:
    check_exist_policy(policy)


def test_check_exist_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'exist_policy': incorrect"):
        check_exist_policy("incorrect")


##########################################
#     Tests for check_missing_policy     #
##########################################


@pytest.mark.parametrize("policy", ["ignore", "raise", "warn"])
def test_check_missing_policy_valid(policy: str) -> None:
    check_missing_policy(policy)


def test_check_missing_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'missing_policy': incorrect"):
        check_missing_policy("incorrect")

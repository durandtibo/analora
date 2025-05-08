from __future__ import annotations

import warnings

import pytest

from analora.utils.mapping import (
    check_missing_key,
    check_missing_keys,
    find_missing_keys,
)


@pytest.fixture
def data() -> dict:
    return {
        "key1": [1, 2, 3, 4, 5],
        "key2": ["1", "2", "3", "4", "5"],
        "key3": ["a ", " b", "  c  ", "d", "e"],
    }


#######################################
#     Tests for check_missing_key     #
#######################################


@pytest.mark.parametrize("missing_policy", ["ignore", "raise", "warn"])
def test_check_missing_key_data(data: dict, missing_policy: str) -> None:
    check_missing_key(data, key="key1", missing_policy=missing_policy)


@pytest.mark.parametrize("missing_policy", ["ignore", "raise", "warn"])
def test_check_missing_key_keys(missing_policy: str) -> None:
    check_missing_key(["key1", "key2", "key3", "key4"], key="key1", missing_policy=missing_policy)


def test_check_missing_key_ignore(data: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_missing_key(data, key="key", missing_policy="ignore")


def test_check_missing_key_raise(data: dict) -> None:
    with pytest.raises(KeyError, match="key 'key' is missing in the data"):
        check_missing_key(data, key="key", missing_policy="raise")


def test_check_missing_key_warn(data: dict) -> None:
    with pytest.warns(RuntimeWarning, match="key 'key' is missing in the data and will be ignored"):
        check_missing_key(data, key="key", missing_policy="warn")


########################################
#     Tests for check_missing_keys     #
########################################


@pytest.mark.parametrize("missing_policy", ["ignore", "raise", "warn"])
def test_check_missing_keys(data: dict, missing_policy: str) -> None:
    check_missing_keys(data, keys=["key1", "key3"], missing_policy=missing_policy)


def test_check_missing_keys_ignore(data: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_missing_keys(data, keys=["key1", "key5"], missing_policy="ignore")


def test_check_missing_keys_raise_1(data: dict) -> None:
    with pytest.raises(KeyError, match="1 key is missing in the data:"):
        check_missing_keys(data, keys=["key1", "key5"])


def test_check_missing_keys_raise_2(data: dict) -> None:
    with pytest.raises(KeyError, match="2 keys are missing in the data:"):
        check_missing_keys(data, keys=["key1", "key5", "key6"])


def test_check_missing_keys_warn_1(data: dict) -> None:
    with pytest.warns(RuntimeWarning, match="1 key is missing in the data and will be ignored:"):
        check_missing_keys(data, keys=["key1", "key5"], missing_policy="warn")


def test_check_missing_keys_warn_2(data: dict) -> None:
    with pytest.warns(RuntimeWarning, match="2 keys are missing in the data and will be ignored:"):
        check_missing_keys(data, keys=["key1", "key5", "key6"], missing_policy="warn")


#######################################
#     Tests for find_missing_keys     #
#######################################


def test_find_missing_keys_data(data: dict) -> None:
    assert find_missing_keys(data, keys=["key1", "key2", "key3", "key4"]) == ("key4",)


def test_find_missing_keys_1() -> None:
    assert find_missing_keys(["key1", "key2", "key3"], keys=["key1"]) == ()


def test_find_missing_keys_2() -> None:
    assert find_missing_keys(["key1", "key2", "key3"], keys=["key1", "key2"]) == ()


def test_find_missing_keys_3() -> None:
    assert find_missing_keys(["key1", "key2", "key3"], keys=["key1", "key2", "key3"]) == ()


def test_find_missing_keys_4() -> None:
    assert find_missing_keys(["key1", "key2", "key3"], keys=["key1", "key2", "key3", "key4"]) == (
        "key4",
    )


def test_find_missing_keys_empty() -> None:
    assert find_missing_keys([], []) == ()

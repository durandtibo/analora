from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from iden.io import save_pickle

from analora.ingestor import PickleIngestor

if TYPE_CHECKING:
    from pathlib import Path

####################################
#     Tests for PickleIngestor     #
####################################


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    path = tmp_path.joinpath("data.pickle")
    save_pickle({"key1": "abc", "key2": 42}, path)
    return path


def test_pickle_ingestor_repr(data_path: Path) -> None:
    assert repr(PickleIngestor(data_path)).startswith("PickleIngestor(")


def test_pickle_ingestor_str(data_path: Path) -> None:
    assert str(PickleIngestor(data_path)).startswith("PickleIngestor(")


def test_pickle_ingestor_equal_true(data_path: Path) -> None:
    assert PickleIngestor(data_path).equal(PickleIngestor(data_path))


def test_pickle_ingestor_equal_false_different_path(data_path: Path, tmp_path: Path) -> None:
    assert not PickleIngestor(data_path).equal(PickleIngestor(tmp_path))


def test_pickle_ingestor_equal_false_different_type(data_path: Path) -> None:
    assert not PickleIngestor(data_path).equal(42)


def test_pickle_ingestor_ingest(data_path: Path) -> None:
    out = PickleIngestor(data_path).ingest()
    assert objects_are_equal(out, {"key1": "abc", "key2": 42})

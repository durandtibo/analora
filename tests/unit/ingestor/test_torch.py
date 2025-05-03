from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from coola.testing import torch_available
from iden.io import save_torch

from analora.ingestor import TorchIngestor

if TYPE_CHECKING:
    from pathlib import Path

###################################
#     Tests for TorchIngestor     #
###################################


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    path = tmp_path.joinpath("data.pt")
    save_torch({"key1": "abc", "key2": 42}, path)
    return path


@torch_available
def test_torch_ingestor_repr(data_path: Path) -> None:
    assert repr(TorchIngestor(data_path)).startswith("TorchIngestor(")


@torch_available
def test_torch_ingestor_str(data_path: Path) -> None:
    assert str(TorchIngestor(data_path)).startswith("TorchIngestor(")


@torch_available
def test_torch_ingestor_equal_true(data_path: Path) -> None:
    assert TorchIngestor(data_path).equal(TorchIngestor(data_path))


@torch_available
def test_torch_ingestor_equal_false_different_path(data_path: Path, tmp_path: Path) -> None:
    assert not TorchIngestor(data_path).equal(TorchIngestor(tmp_path))


@torch_available
def test_torch_ingestor_equal_false_different_type(data_path: Path) -> None:
    assert not TorchIngestor(data_path).equal(42)


@torch_available
def test_torch_ingestor_ingest(data_path: Path) -> None:
    out = TorchIngestor(data_path).ingest()
    assert objects_are_equal(out, {"key1": "abc", "key2": 42})

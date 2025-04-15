from __future__ import annotations

from coola import objects_are_equal

from analora.ingestor import Ingestor

##############################
#     Tests for Ingestor     #
##############################


def test_ingestor_repr() -> None:
    assert repr(Ingestor(data=[1, 2, 3, 4])) == "Ingestor()"


def test_ingestor_str() -> None:
    assert str(Ingestor(data=[1, 2, 3, 4])) == "Ingestor()"


def test_ingestor_equal_true() -> None:
    assert Ingestor([1, 2, 3, 4]).equal(Ingestor([1, 2, 3, 4]))


def test_ingestor_equal_false_different_data() -> None:
    assert not Ingestor([1, 2, 3, 4]).equal(Ingestor(["a", "b", "c"]))


def test_ingestor_equal_false_different_type() -> None:
    assert not Ingestor([1, 2, 3, 4]).equal(42)


def test_ingestor_ingest() -> None:
    data = [1, 2, 3, 4]
    out = Ingestor(data=data).ingest()
    assert data is out
    assert objects_are_equal(out, data)

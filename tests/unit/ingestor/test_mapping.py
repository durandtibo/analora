from __future__ import annotations

from coola import objects_are_equal

from analora.ingestor import Ingestor, MappingIngestor

#####################################
#     Tests for MappingIngestor     #
####################################3


def test_mapping_ingestor_repr() -> None:
    assert repr(
        MappingIngestor({"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")})
    ).startswith("MappingIngestor(")


def test_mapping_ingestor_str() -> None:
    assert str(
        MappingIngestor({"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")})
    ).startswith("MappingIngestor(")


def test_mapping_ingestor_equal_true() -> None:
    assert MappingIngestor(
        {"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")}
    ).equal(
        MappingIngestor({"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")})
    )


def test_mapping_ingestor_equal_false_different_data() -> None:
    assert not MappingIngestor(
        {"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")}
    ).equal(MappingIngestor({"key1": Ingestor(data=[1, 2, 3, 4, 5])}))


def test_mapping_ingestor_equal_false_different_type() -> None:
    assert not MappingIngestor(
        {"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")}
    ).equal(42)


def test_mapping_ingestor_ingest_1() -> None:
    out = MappingIngestor({"key1": Ingestor(data=[1, 2, 3, 4, 5])}).ingest()
    assert objects_are_equal(out, {"key1": [1, 2, 3, 4, 5]})


def test_mapping_ingestor_ingest_2() -> None:
    out = MappingIngestor(
        {"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")}
    ).ingest()
    assert objects_are_equal(out, {"key1": [1, 2, 3, 4, 5], "key2": "meow"})


def test_mapping_ingestor_ingest_3() -> None:
    out = MappingIngestor(
        {
            "key1": Ingestor(data=[1, 2, 3, 4, 5]),
            "key2": Ingestor(data="meow"),
            "key3": Ingestor(42),
        }
    ).ingest()
    assert objects_are_equal(out, {"key1": [1, 2, 3, 4, 5], "key2": "meow", "key3": 42})

from __future__ import annotations

from typing import Any

from analora.content import BaseSectionContentGenerator, ContentGenerator

#################################################
#     Tests for BaseSectionContentGenerator     #
#################################################


class MySectionContentGenerator(BaseSectionContentGenerator):
    def __init__(self, value: str) -> None:
        self._value = value

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        if not isinstance(other, self.__class__):
            return False
        return self._value == other._value

    def generate_content(self) -> str:
        return self._value


def test_base_section_content_generator_compute() -> None:
    assert MySectionContentGenerator("meow").compute().equal(ContentGenerator("meow"))


def test_base_section_content_generator_equal_true() -> None:
    assert MySectionContentGenerator("meow").equal(MySectionContentGenerator("meow"))


def test_base_section_content_generator_equal_false_different_value() -> None:
    assert not MySectionContentGenerator("meow").equal(MySectionContentGenerator("miaou"))


def test_base_section_content_generator_equal_false_different_type() -> None:
    assert not MySectionContentGenerator("meow").equal(42)


def test_base_section_content_generator_generate_content() -> None:
    assert MySectionContentGenerator("meow").generate_content() == "meow"


def test_base_section_content_generator_generate_body() -> None:
    assert MySectionContentGenerator("meow").generate_body() == (
        '<h1 id="">  </h1>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_base_section_content_generator_generate_body_args() -> None:
    assert MySectionContentGenerator("meow").generate_body(number="1.", tags=["meow"], depth=1) == (
        '<h2 id="meow">1. meow </h2>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_base_section_content_generator_generate_body_depth_1() -> None:
    assert MySectionContentGenerator("meow").generate_body(depth=1) == (
        '<h2 id="">  </h2>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_base_section_content_generator_generate_body_depth_2() -> None:
    assert MySectionContentGenerator("meow").generate_body(depth=2) == (
        '<h3 id="">  </h3>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_base_section_content_generator_generate_body_empty() -> None:
    assert MySectionContentGenerator("").generate_body() == (
        '<h1 id="">  </h1>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "\n"
        '<p style="margin-top: 1rem;">'
    )


def test_base_section_content_generator_generate_toc() -> None:
    assert MySectionContentGenerator("meow").generate_toc() == '<li><a href="#"> </a></li>'


def test_base_section_content_generator_generate_toc_args() -> None:
    assert (
        MySectionContentGenerator("meow").generate_toc(
            number="1.", tags=["meow"], depth=1, max_depth=6
        )
        == '<li><a href="#meow">1. meow</a></li>'
    )


def test_base_section_content_generator_generate_toc_too_deep() -> None:
    assert MySectionContentGenerator("meow").generate_toc(number="1.", tags=["meow"], depth=1) == ""

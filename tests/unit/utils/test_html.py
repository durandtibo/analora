from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from analora.utils.html import (
    MISSING_FIGURE_MESSAGE,
    figure2html,
    render_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)

#############################
#     Tests for tags2id     #
#############################


def test_tags2id_empty() -> None:
    assert tags2id([]) == ""


def test_tags2id_1_tag() -> None:
    assert tags2id(["meow"]) == "meow"


def test_tags2id_2_tags() -> None:
    assert tags2id(["super", "meow"]) == "super-meow"


################################
#     Tests for tags2title     #
################################


def test_tags2title_empty() -> None:
    assert tags2title([]) == ""


def test_tags2title_1_tag() -> None:
    assert tags2title(["meow"]) == "meow"


def test_tags2title_2_tags() -> None:
    assert tags2title(["super", "meow"]) == "meow | super"


#################################
#     Tests for valid_h_tag     #
#################################


def test_valid_h_tag_0() -> None:
    assert valid_h_tag(0) == 1


def test_valid_h_tag_1() -> None:
    assert valid_h_tag(1) == 1


def test_valid_h_tag_6() -> None:
    assert valid_h_tag(6) == 6


def test_valid_h_tag_7() -> None:
    assert valid_h_tag(7) == 6


################################
#     Tests for render_toc     #
################################


def test_render_toc_no_tags_and_number() -> None:
    assert render_toc() == '<li><a href="#"> </a></li>'


def test_render_toc_no_tags() -> None:
    assert render_toc(number="1.2.") == '<li><a href="#">1.2. </a></li>'


def test_render_toc_tags() -> None:
    assert (
        render_toc(number="1.2.", tags=("super", "meow"))
        == '<li><a href="#super-meow">1.2. meow</a></li>'
    )


def test_render_toc_tags_without_number() -> None:
    assert render_toc(tags=("super", "meow")) == '<li><a href="#super-meow"> meow</a></li>'


def test_render_toc_max_depth() -> None:
    assert render_toc(depth=2, max_depth=2) == ""


#################################
#     Tests for figure2html     #
#################################


def test_figure2html() -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig), str)


@pytest.mark.parametrize("close_fig", [True, False])
def test_figure2html_close_fig(close_fig: bool) -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig, close_fig=close_fig), str)


@pytest.mark.parametrize("reactive", [True, False])
def test_figure2html_reactive(reactive: bool) -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig, reactive=reactive), str)


def test_figure2html_none() -> None:
    assert figure2html(None) == MISSING_FIGURE_MESSAGE

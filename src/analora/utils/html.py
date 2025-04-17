r"""Contain utility functions to generate sections."""

from __future__ import annotations

__all__ = [
    "GO_TO_TOP",
    "MISSING_FIGURE_MESSAGE",
    "figure2html",
    "render_toc",
    "tags2id",
    "tags2title",
    "valid_h_tag",
]

import base64
import io
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence

GO_TO_TOP = '<a href="#">Go to top</a>'
MISSING_FIGURE_MESSAGE = (
    "<span>&#9888;</span> No figure is generated because of missing or incorrect data"
)


def tags2id(tags: Sequence[str]) -> str:
    r"""Convert a sequence of tags to a string that can be used as ID in
    a HTML file.

    Args:
        tags: The sequence of tags.

    Returns:
        The generated ID from the tags.

    Example usage:

    ```pycon

    >>> from analora.utils.html import tags2id
    >>> out = tags2id(["super", "meow"])
    >>> out
    super-meow

    ```
    """
    return "-".join(tags).replace(" ", "-").lower()


def tags2title(tags: Sequence[str]) -> str:
    r"""Convert a sequence of tags to a string that can be used as title.

    Args:
        tags: The sequence of tags.

    Returns:
        The generated title from the tags.

    Example usage:

    ```pycon

    >>> from analora.utils.html import tags2title
    >>> out = tags2title(["super", "meow"])
    >>> out
    meow | super

    ```
    """
    return " | ".join(tags[::-1])


def valid_h_tag(index: int) -> int:
    r"""Return a valid number of a h HTML tag.

    Args:
        index: The original value.

    Returns:
        A valid value.

    Example usage:

    ```pycon

    >>> from analora.utils.html import valid_h_tag
    >>> out = valid_h_tag(4)
    >>> out
    4

    ```
    """
    return max(1, min(6, index))


def render_toc(
    number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
) -> str:
    r"""Return the HTML table of content (TOC) associated to the section.

    Args:
        number: The section number associated to the section.
        tags: The tags associated to the section.
        depth: The depth in the report.
        max_depth: The maximum depth to generate in the TOC.

    Returns:
        The HTML table of content associated to the section.

    Example usage:

    ```pycon

    >>> from analora.utils.html import render_toc
    >>> out = render_toc(number="2.0", tags=["super", "meow"])
    >>> out
    <li><a href="#super-meow">2.0 meow</a></li>

    ```
    """
    if depth >= max_depth:
        return ""
    tag = tags[-1] if tags else ""
    return f'<li><a href="#{tags2id(tags)}">{number} {tag}</a></li>'


def figure2html(fig: plt.Figure | None, reactive: bool = True, close_fig: bool = False) -> str:
    r"""Convert a matplotlib figure to a string that can be used in a
    HTML file.

    Args:
        fig: The figure to convert.
        reactive: If ``True``, the generated is configured to be
            reactive to the screen size.
        close_fig: If ``True``, the figure is closed after it is
            converted to HTML format.

    Returns:
        The converted figure to a string.

    Example usage:

    ```pycon

    >>> from matplotlib import pyplot as plt
    >>> from analora.utils.html import figure2html
    >>> fig, ax = plt.subplots()
    >>> string = figure2html(fig)

    ```
    """
    if fig is None:
        return MISSING_FIGURE_MESSAGE
    fig.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    data = base64.b64encode(img.getvalue()).decode("utf-8")
    if close_fig:
        plt.close(fig)
    style = 'style="width:100%; height:auto;" ' if reactive else ""
    return f'<img {style}src="data:image/png;charset=utf-8;base64, {data}">'

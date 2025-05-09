r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "check_hya",
    "check_hydra",
    "check_markdown",
    "check_objectory",
    "check_omegaconf",
    "check_polars",
    "check_scipy",
    "check_tqdm",
    "colorlog_available",
    "hya_available",
    "hydra_available",
    "is_colorlog_available",
    "is_hya_available",
    "is_hydra_available",
    "is_markdown_available",
    "is_objectory_available",
    "is_omegaconf_available",
    "is_polars_available",
    "is_scipy_available",
    "is_tqdm_available",
    "markdown_available",
    "objectory_available",
    "omegaconf_available",
    "polars_available",
    "scipy_available",
    "tqdm_available",
]

from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available, package_available

if TYPE_CHECKING:
    from collections.abc import Callable


####################
#     colorlog     #
####################


def is_colorlog_available() -> bool:
    r"""Indicate if the ``colorlog`` package is installed or not.

    Returns:
        ``True`` if ``colorlog`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_colorlog_available
    >>> is_colorlog_available()

    ```
    """
    return package_available("colorlog")


def check_colorlog() -> None:
    r"""Check if the ``colorlog`` package is installed.

    Raises:
        RuntimeError: if the ``colorlog`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_colorlog
    >>> check_colorlog()

    ```
    """
    if not is_colorlog_available():
        msg = (
            "'colorlog' package is required but not installed. "
            "You can install 'colorlog' package with the command:\n\n"
            "pip install colorlog\n"
        )
        raise RuntimeError(msg)


def colorlog_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``colorlog``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``colorlog`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import colorlog_available
    >>> @colorlog_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_colorlog_available)


###############
#     hya     #
###############


def is_hya_available() -> bool:
    r"""Indicate if the ``hya`` package is installed or not.

    Returns:
        ``True`` if ``hya`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_hya_available
    >>> is_hya_available()

    ```
    """
    return package_available("hya")


def check_hya() -> None:
    r"""Check if the ``hya`` package is installed.

    Raises:
        RuntimeError: if the ``hya`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_hya
    >>> check_hya()

    ```
    """
    if not is_hya_available():
        msg = (
            "'hya' package is required but not installed. "
            "You can install 'hya' package with the command:\n\n"
            "pip install hya\n"
        )
        raise RuntimeError(msg)


def hya_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``hya``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``hya`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import hya_available
    >>> @hya_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_hya_available)


#################
#     hydra     #
#################


def is_hydra_available() -> bool:
    r"""Indicate if the ``hydra`` package is installed or not.

    Returns:
        ``True`` if ``hydra`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_hydra_available
    >>> is_hydra_available()

    ```
    """
    return package_available("hydra")


def check_hydra() -> None:
    r"""Check if the ``hydra`` package is installed.

    Raises:
        RuntimeError: if the ``hydra`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_hydra
    >>> check_hydra()

    ```
    """
    if not is_hydra_available():
        msg = (
            "'hydra' package is required but not installed. "
            "You can install 'hydra' package with the command:\n\n"
            "pip install hydra-core\n"
        )
        raise RuntimeError(msg)


def hydra_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``hydra``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``hydra`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import hydra_available
    >>> @hydra_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_hydra_available)


####################
#     markdown     #
####################


def is_markdown_available() -> bool:
    r"""Indicate if the ``markdown`` package is installed or not.

    Returns:
        ``True`` if ``markdown`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_markdown_available
    >>> is_markdown_available()

    ```
    """
    return package_available("markdown")


def check_markdown() -> None:
    r"""Check if the ``markdown`` package is installed.

    Raises:
        RuntimeError: if the ``markdown`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_markdown
    >>> check_markdown()

    ```
    """
    if not is_markdown_available():
        msg = (
            "'markdown' package is required but not installed. "
            "You can install 'markdown' package with the command:\n\n"
            "pip install markdown\n"
        )
        raise RuntimeError(msg)


def markdown_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``markdown``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``markdown`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import markdown_available
    >>> @markdown_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_markdown_available)


#####################
#     objectory     #
#####################


def is_objectory_available() -> bool:
    r"""Indicate if the ``objectory`` package is installed or not.

    Returns:
        ``True`` if ``objectory`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_objectory_available
    >>> is_objectory_available()

    ```
    """
    return package_available("objectory")


def check_objectory() -> None:
    r"""Check if the ``objectory`` package is installed.

    Raises:
        RuntimeError: if the ``objectory`` package is not installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_objectory
    >>> check_objectory()

    ```
    """
    if not is_objectory_available():
        msg = (
            "'objectory' package is required but not installed. "
            "You can install 'objectory' package with the command:\n\n"
            "pip install objectory\n"
        )
        raise RuntimeError(msg)


def objectory_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``objectory``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``objectory`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import objectory_available
    >>> @objectory_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_objectory_available)


#####################
#     omegaconf     #
#####################


def is_omegaconf_available() -> bool:
    r"""Indicate if the ``omegaconf`` package is installed or not.

    Returns:
        ``True`` if ``omegaconf`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_omegaconf_available
    >>> is_omegaconf_available()

    ```
    """
    return package_available("omegaconf")


def check_omegaconf() -> None:
    r"""Check if the ``omegaconf`` package is installed.

    Raises:
        RuntimeError: if the ``omegaconf`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_omegaconf
    >>> check_omegaconf()

    ```
    """
    if not is_omegaconf_available():
        msg = (
            "'omegaconf' package is required but not installed. "
            "You can install 'omegaconf' package with the command:\n\n"
            "pip install omegaconf\n"
        )
        raise RuntimeError(msg)


def omegaconf_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``omegaconf``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``omegaconf`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import omegaconf_available
    >>> @omegaconf_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_omegaconf_available)


##################
#     polars     #
##################


def is_polars_available() -> bool:
    r"""Indicate if the ``polars`` package is installed or not.

    Returns:
        ``True`` if ``polars`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_polars_available
    >>> is_polars_available()

    ```
    """
    return package_available("polars")


def check_polars() -> None:
    r"""Check if the ``polars`` package is installed.

    Raises:
        RuntimeError: if the ``polars`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_polars
    >>> check_polars()

    ```
    """
    if not is_polars_available():
        msg = (
            "'polars' package is required but not installed. "
            "You can install 'polars' package with the command:\n\n"
            "pip install polars\n"
        )
        raise RuntimeError(msg)


def polars_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``polars``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``polars`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import polars_available
    >>> @polars_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_polars_available)


#################
#     scipy     #
#################


def is_scipy_available() -> bool:
    r"""Indicate if the ``scipy`` package is installed or not.

    Returns:
        ``True`` if ``scipy`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_scipy_available
    >>> is_scipy_available()

    ```
    """
    return package_available("scipy")


def check_scipy() -> None:
    r"""Check if the ``scipy`` package is installed.

    Raises:
        RuntimeError: if the ``scipy`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_scipy
    >>> check_scipy()

    ```
    """
    if not is_scipy_available():
        msg = (
            "'scipy' package is required but not installed. "
            "You can install 'scipy' package with the command:\n\n"
            "pip install scipy\n"
        )
        raise RuntimeError(msg)


def scipy_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``scipy``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``scipy`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import scipy_available
    >>> @scipy_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_scipy_available)


###################
#     sklearn     #
###################


def is_sklearn_available() -> bool:
    r"""Indicate if the ``sklearn`` package is installed or not.

    Returns:
        ``True`` if ``sklearn`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_sklearn_available
    >>> is_sklearn_available()

    ```
    """
    return package_available("sklearn")


def check_sklearn() -> None:
    r"""Check if the ``sklearn`` package is installed.

    Raises:
        RuntimeError: if the ``sklearn`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_sklearn
    >>> check_sklearn()

    ```
    """
    if not is_sklearn_available():
        msg = (
            "'sklearn' package is required but not installed. "
            "You can install 'sklearn' package with the command:\n\n"
            "pip install scikit-learn\n"
        )
        raise RuntimeError(msg)


def sklearn_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``sklearn``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``sklearn`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import sklearn_available
    >>> @sklearn_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_sklearn_available)


################
#     tqdm     #
################


def is_tqdm_available() -> bool:
    r"""Indicate if the ``tqdm`` package is installed or not.

    Returns:
        ``True`` if ``tqdm`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import is_tqdm_available
    >>> is_tqdm_available()

    ```
    """
    return package_available("tqdm")


def check_tqdm() -> None:
    r"""Check if the ``tqdm`` package is installed.

    Raises:
        RuntimeError: if the ``tqdm`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import check_tqdm
    >>> check_tqdm()

    ```
    """
    if not is_tqdm_available():
        msg = (
            "'tqdm' package is required but not installed. "
            "You can install 'tqdm' package with the command:\n\n"
            "pip install tqdm\n"
        )
        raise RuntimeError(msg)


def tqdm_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``tqdm``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``tqdm`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from analora.utils.imports import tqdm_available
    >>> @tqdm_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_tqdm_available)

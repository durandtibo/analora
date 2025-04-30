from __future__ import annotations

from unittest.mock import patch

import pytest

from analora.utils.imports import (
    check_colorlog,
    check_hya,
    check_hydra,
    check_markdown,
    check_objectory,
    check_omegaconf,
    check_polars,
    check_scipy,
    check_sklearn,
    check_tqdm,
    colorlog_available,
    hya_available,
    hydra_available,
    is_colorlog_available,
    is_hya_available,
    is_hydra_available,
    is_markdown_available,
    is_objectory_available,
    is_omegaconf_available,
    is_polars_available,
    is_scipy_available,
    is_sklearn_available,
    is_tqdm_available,
    markdown_available,
    objectory_available,
    omegaconf_available,
    polars_available,
    scipy_available,
    sklearn_available,
    tqdm_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


####################
#     colorlog     #
####################


def test_check_colorlog_with_package() -> None:
    with patch("analora.utils.imports.is_colorlog_available", lambda: True):
        check_colorlog()


def test_check_colorlog_without_package() -> None:
    with (
        patch("analora.utils.imports.is_colorlog_available", lambda: False),
        pytest.raises(RuntimeError, match="'colorlog' package is required but not installed."),
    ):
        check_colorlog()


def test_is_colorlog_available() -> None:
    assert isinstance(is_colorlog_available(), bool)


def test_colorlog_available_with_package() -> None:
    with patch("analora.utils.imports.is_colorlog_available", lambda: True):
        fn = colorlog_available(my_function)
        assert fn(2) == 44


def test_colorlog_available_without_package() -> None:
    with patch("analora.utils.imports.is_colorlog_available", lambda: False):
        fn = colorlog_available(my_function)
        assert fn(2) is None


def test_colorlog_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_colorlog_available", lambda: True):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_colorlog_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_colorlog_available", lambda: False):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


###############
#     hya     #
###############


def test_check_hya_with_package() -> None:
    with patch("analora.utils.imports.is_hya_available", lambda: True):
        check_hya()


def test_check_hya_without_package() -> None:
    with (
        patch("analora.utils.imports.is_hya_available", lambda: False),
        pytest.raises(RuntimeError, match="'hya' package is required but not installed."),
    ):
        check_hya()


def test_is_hya_available() -> None:
    assert isinstance(is_hya_available(), bool)


def test_hya_available_with_package() -> None:
    with patch("analora.utils.imports.is_hya_available", lambda: True):
        fn = hya_available(my_function)
        assert fn(2) == 44


def test_hya_available_without_package() -> None:
    with patch("analora.utils.imports.is_hya_available", lambda: False):
        fn = hya_available(my_function)
        assert fn(2) is None


def test_hya_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_hya_available", lambda: True):

        @hya_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_hya_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_hya_available", lambda: False):

        @hya_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#################
#     hydra     #
#################


def test_check_hydra_with_package() -> None:
    with patch("analora.utils.imports.is_hydra_available", lambda: True):
        check_hydra()


def test_check_hydra_without_package() -> None:
    with (
        patch("analora.utils.imports.is_hydra_available", lambda: False),
        pytest.raises(RuntimeError, match="'hydra' package is required but not installed."),
    ):
        check_hydra()


def test_is_hydra_available() -> None:
    assert isinstance(is_hydra_available(), bool)


def test_hydra_available_with_package() -> None:
    with patch("analora.utils.imports.is_hydra_available", lambda: True):
        fn = hydra_available(my_function)
        assert fn(2) == 44


def test_hydra_available_without_package() -> None:
    with patch("analora.utils.imports.is_hydra_available", lambda: False):
        fn = hydra_available(my_function)
        assert fn(2) is None


def test_hydra_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_hydra_available", lambda: True):

        @hydra_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_hydra_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_hydra_available", lambda: False):

        @hydra_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


####################
#     markdown     #
####################


def test_check_markdown_with_package() -> None:
    with patch("analora.utils.imports.is_markdown_available", lambda: True):
        check_markdown()


def test_check_markdown_without_package() -> None:
    with (
        patch("analora.utils.imports.is_markdown_available", lambda: False),
        pytest.raises(RuntimeError, match="'markdown' package is required but not installed."),
    ):
        check_markdown()


def test_is_markdown_available() -> None:
    assert isinstance(is_markdown_available(), bool)


def test_markdown_available_with_package() -> None:
    with patch("analora.utils.imports.is_markdown_available", lambda: True):
        fn = markdown_available(my_function)
        assert fn(2) == 44


def test_markdown_available_without_package() -> None:
    with patch("analora.utils.imports.is_markdown_available", lambda: False):
        fn = markdown_available(my_function)
        assert fn(2) is None


def test_markdown_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_markdown_available", lambda: True):

        @markdown_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_markdown_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_markdown_available", lambda: False):

        @markdown_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#####################
#     objectory     #
#####################


def test_check_objectory_with_package() -> None:
    with patch("analora.utils.imports.is_objectory_available", lambda: True):
        check_objectory()


def test_check_objectory_without_package() -> None:
    with (
        patch("analora.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        check_objectory()


def test_is_objectory_available() -> None:
    assert isinstance(is_objectory_available(), bool)


def test_objectory_available_with_package() -> None:
    with patch("analora.utils.imports.is_objectory_available", lambda: True):
        fn = objectory_available(my_function)
        assert fn(2) == 44


def test_objectory_available_without_package() -> None:
    with patch("analora.utils.imports.is_objectory_available", lambda: False):
        fn = objectory_available(my_function)
        assert fn(2) is None


def test_objectory_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_objectory_available", lambda: True):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_objectory_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_objectory_available", lambda: False):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#####################
#     omegaconf     #
#####################


def test_check_omegaconf_with_package() -> None:
    with patch("analora.utils.imports.is_omegaconf_available", lambda: True):
        check_omegaconf()


def test_check_omegaconf_without_package() -> None:
    with (
        patch("analora.utils.imports.is_omegaconf_available", lambda: False),
        pytest.raises(RuntimeError, match="'omegaconf' package is required but not installed."),
    ):
        check_omegaconf()


def test_is_omegaconf_available() -> None:
    assert isinstance(is_omegaconf_available(), bool)


def test_omegaconf_available_with_package() -> None:
    with patch("analora.utils.imports.is_omegaconf_available", lambda: True):
        fn = omegaconf_available(my_function)
        assert fn(2) == 44


def test_omegaconf_available_without_package() -> None:
    with patch("analora.utils.imports.is_omegaconf_available", lambda: False):
        fn = omegaconf_available(my_function)
        assert fn(2) is None


def test_omegaconf_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_omegaconf_available", lambda: True):

        @omegaconf_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_omegaconf_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_omegaconf_available", lambda: False):

        @omegaconf_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     polars     #
##################


def test_check_polars_with_package() -> None:
    with patch("analora.utils.imports.is_polars_available", lambda: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with (
        patch("analora.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match="'polars' package is required but not installed."),
    ):
        check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)


def test_polars_available_with_package() -> None:
    with patch("analora.utils.imports.is_polars_available", lambda: True):
        fn = polars_available(my_function)
        assert fn(2) == 44


def test_polars_available_without_package() -> None:
    with patch("analora.utils.imports.is_polars_available", lambda: False):
        fn = polars_available(my_function)
        assert fn(2) is None


def test_polars_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_polars_available", lambda: True):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_polars_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_polars_available", lambda: False):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#################
#     scipy     #
#################


def test_check_scipy_with_package() -> None:
    with patch("analora.utils.imports.is_scipy_available", lambda: True):
        check_scipy()


def test_check_scipy_without_package() -> None:
    with (
        patch("analora.utils.imports.is_scipy_available", lambda: False),
        pytest.raises(RuntimeError, match="'scipy' package is required but not installed."),
    ):
        check_scipy()


def test_is_scipy_available() -> None:
    assert isinstance(is_scipy_available(), bool)


def test_scipy_available_with_package() -> None:
    with patch("analora.utils.imports.is_scipy_available", lambda: True):
        fn = scipy_available(my_function)
        assert fn(2) == 44


def test_scipy_available_without_package() -> None:
    with patch("analora.utils.imports.is_scipy_available", lambda: False):
        fn = scipy_available(my_function)
        assert fn(2) is None


def test_scipy_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_scipy_available", lambda: True):

        @scipy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_scipy_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_scipy_available", lambda: False):

        @scipy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


###################
#     sklearn     #
###################


def test_check_sklearn_with_package() -> None:
    with patch("analora.utils.imports.is_sklearn_available", lambda: True):
        check_sklearn()


def test_check_sklearn_without_package() -> None:
    with (
        patch("analora.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        check_sklearn()


def test_is_sklearn_available() -> None:
    assert isinstance(is_sklearn_available(), bool)


def test_sklearn_available_with_package() -> None:
    with patch("analora.utils.imports.is_sklearn_available", lambda: True):
        fn = sklearn_available(my_function)
        assert fn(2) == 44


def test_sklearn_available_without_package() -> None:
    with patch("analora.utils.imports.is_sklearn_available", lambda: False):
        fn = sklearn_available(my_function)
        assert fn(2) is None


def test_sklearn_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_sklearn_available", lambda: True):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_sklearn_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_sklearn_available", lambda: False):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


################
#     tqdm     #
################


def test_check_tqdm_with_package() -> None:
    with patch("analora.utils.imports.is_tqdm_available", lambda: True):
        check_tqdm()


def test_check_tqdm_without_package() -> None:
    with (
        patch("analora.utils.imports.is_tqdm_available", lambda: False),
        pytest.raises(RuntimeError, match="'tqdm' package is required but not installed."),
    ):
        check_tqdm()


def test_is_tqdm_available() -> None:
    assert isinstance(is_tqdm_available(), bool)


def test_tqdm_available_with_package() -> None:
    with patch("analora.utils.imports.is_tqdm_available", lambda: True):
        fn = tqdm_available(my_function)
        assert fn(2) == 44


def test_tqdm_available_without_package() -> None:
    with patch("analora.utils.imports.is_tqdm_available", lambda: False):
        fn = tqdm_available(my_function)
        assert fn(2) is None


def test_tqdm_available_decorator_with_package() -> None:
    with patch("analora.utils.imports.is_tqdm_available", lambda: True):

        @tqdm_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_tqdm_available_decorator_without_package() -> None:
    with patch("analora.utils.imports.is_tqdm_available", lambda: False):

        @tqdm_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None

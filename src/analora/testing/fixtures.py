r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = [
    "colorlog_available",
    "hya_available",
    "hydra_available",
    "markdown_available",
    "objectory_available",
    "omegaconf_available",
    "polars_available",
    "scipy_available",
    "sklearn_available",
    "tqdm_available",
]

import pytest

from analora.utils.imports import (
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
)

colorlog_available = pytest.mark.skipif(not is_colorlog_available(), reason="requires colorlog")
hya_available = pytest.mark.skipif(not is_hya_available(), reason="requires hya")
hydra_available = pytest.mark.skipif(not is_hydra_available(), reason="requires hydra")
markdown_available = pytest.mark.skipif(not is_markdown_available(), reason="requires markdown")
objectory_available = pytest.mark.skipif(not is_objectory_available(), reason="Require objectory")
omegaconf_available = pytest.mark.skipif(not is_omegaconf_available(), reason="requires omegaconf")
polars_available = pytest.mark.skipif(not is_polars_available(), reason="requires polars")
scipy_available = pytest.mark.skipif(not is_scipy_available(), reason="requires scipy")
sklearn_available = pytest.mark.skipif(not is_sklearn_available(), reason="requires sklearn")
tqdm_available = pytest.mark.skipif(not is_tqdm_available(), reason="requires tqdm")

r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = [
    "colorlog_available",
    "hya_available",
    "hydra_available",
    "markdown_available",
    "objectory_available",
    "omegaconf_available",
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
    is_tqdm_available,
)

colorlog_available = pytest.mark.skipif(not is_colorlog_available(), reason="requires colorlog")
hya_available = pytest.mark.skipif(not is_hya_available(), reason="requires hya")
hydra_available = pytest.mark.skipif(not is_hydra_available(), reason="requires hydra")
markdown_available = pytest.mark.skipif(not is_markdown_available(), reason="requires markdown")
omegaconf_available = pytest.mark.skipif(not is_omegaconf_available(), reason="requires omegaconf")
tqdm_available = pytest.mark.skipif(not is_tqdm_available(), reason="requires tqdm")
objectory_available = pytest.mark.skipif(not is_objectory_available(), reason="Require objectory")

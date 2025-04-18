r"""Contain a function to instantiate an object from its
configuration."""

from __future__ import annotations

__all__ = ["setup_object"]

import logging
from typing import TypeVar
from unittest.mock import Mock

from analora.utils.imports import check_objectory, is_objectory_available

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()

logger = logging.getLogger(__name__)

T = TypeVar("T")


def setup_object(obj_or_config: T | dict) -> T:
    r"""Set up an object from its configuration.

    Args:
        obj_or_config: The object or its configuration.

    Returns:
        The instantiated object.

    Example usage:

    ```pycon

    >>> from analora.utils.factory import setup_object
    >>> obj = setup_object({"_target_": "collections.deque", "iterable": [1, 2, 1, 3]})
    >>> obj
    deque([1, 2, 1, 3])
    >>> setup_object(obj)  # Do nothing because the object is already instantiated
    deque([1, 2, 1, 3])

    ```
    """
    if isinstance(obj_or_config, dict):
        check_objectory()
        logger.info("Initializing an object from its configuration... ")
        return objectory.factory(**obj_or_config)
    return obj_or_config

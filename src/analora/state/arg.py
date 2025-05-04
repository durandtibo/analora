r"""Contain an abstract state to more easily manage arbitrary keyword
arguments."""

from __future__ import annotations

__all__ = ["BaseArgState"]

import copy
import sys
from abc import abstractmethod
from typing import Any

import numpy as np
from coola import objects_are_equal
from coola.utils import str_indent, str_mapping
from coola.utils.format import repr_mapping_line

from analora.state.base import BaseState

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import (
        Self,  # use backport because it was added in python 3.11
    )


class BaseArgState(BaseState):
    r"""Define a base class to manage arguments."""

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                key: val.shape if isinstance(val, np.ndarray) else val
                for key, val in self.get_args().items()
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    key: val.shape if isinstance(val, np.ndarray) else val
                    for key, val in self.get_args().items()
                }
            )
        )
        return f"{self.__class__.__qualname__}({args})"

    def clone(self, deep: bool = True) -> Self:
        args = self.get_args()
        if deep:
            args = copy.deepcopy(args)
        return self.__class__(**args)

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    @abstractmethod
    def get_args(self) -> dict:
        r"""Get a dictionary with all the arguments of the state.

        Returns:
            The dictionary with all the arguments.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from analora.state import AccuracyState
        >>> state = AccuracyState(
        ...     y_true=np.array([1, 0, 0, 1, 1]),
        ...     y_pred=np.array([1, 0, 0, 1, 1]),
        ...     y_true_name="target",
        ...     y_pred_name="pred",
        ... )
        >>> args = state.get_args()
        >>> args
        {'y_true': array([1, 0, 0, 1, 1]),
         'y_pred': array([1, 0, 0, 1, 1]),
         'y_true_name': 'target',
         'y_pred_name': 'pred',
         'nan_policy': 'propagate'}

        ```
        """

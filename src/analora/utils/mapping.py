r"""Contain mapping utility functions."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

from analora.utils.policy import check_missing_policy


def check_missing_keys(
    mapping_or_keys: Mapping | Sequence, keys: Sequence, missing_policy: str = "raise"
) -> None:
    r"""Check if some keys are missing.

    Args:
        mapping_or_keys: The mapping or its keys.
        keys: The keys to check.
        missing_policy: The policy on how to handle missing keys.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one key is missing.
            If ``'warn'``, a warning is raised if at least one key
            is missing and the missing keys are ignored.
            If ``'ignore'``, the missing keys are ignored and
            no warning message appears.

    Raises:
        KeyError: if at least one key is missing and
            ``missing_policy='raise'``.

    Example usage:

    ```pycon

    >>> from analora.utils.mapping import check_missing_keys
    >>> data = {
    ...     "key1": [1, 2, 3, 4, 5],
    ...     "key2": ["1", "2", "3", "4", "5"],
    ...     "key3": ["a ", " b", "  c  ", "d", "e"],
    ...     "key4": ["a ", " b", "  c  ", "d", "e"],
    ... }
    >>> check_missing_keys(data, keys=["key1", "key5"], missing_policy="warn")

    ```
    """
    check_missing_policy(missing_policy)
    missing_keys = find_missing_keys(mapping_or_keys=mapping_or_keys, keys=keys)
    if not missing_keys:
        return
    m = "key is" if len(missing_keys) == 1 else "keys are"
    if missing_policy == "raise":
        msg = f"{len(missing_keys):,} {m} missing in the data: {missing_keys}"
        raise KeyError(msg)
    if missing_policy == "warn":
        msg = f"{len(missing_keys):,} {m} missing in the data and will be ignored: {missing_keys}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def find_missing_keys(mapping_or_keys: Mapping | Sequence, keys: Sequence[str]) -> tuple[str, ...]:
    r"""Find the keys that are in the given keys but not in the mapping.

    Args:
        mapping_or_keys: The mapping or its keys.
        keys: The keys to check.

    Returns:
        The list of missing keys i.e. the keys that are in
            ``keys`` but not in ``mapping_or_keys``.

    Example usage:

    ```pycon

    >>> from analora.utils.mapping import find_missing_keys
    >>> data = {
    ...     "key1": [1, 2, 3, 4, 5],
    ...     "key2": ["1", "2", "3", "4", "5"],
    ...     "key3": ["a ", " b", "  c  ", "d", "e"],
    ... }
    >>> keys = find_missing_keys(data, keys=["key1", "key2", "key3", "key4"])
    >>> keys
    ('key4',)

    ```
    """
    data_keys = set(
        mapping_or_keys.keys() if isinstance(mapping_or_keys, Mapping) else mapping_or_keys
    )
    keys = set(keys)
    return tuple(sorted(keys.difference(data_keys).intersection(keys)))

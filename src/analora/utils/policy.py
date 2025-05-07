r"""Contain utility functions to manage policies."""

from __future__ import annotations

__all__ = ["check_exist_policy", "check_missing_policy"]


def check_exist_policy(exist_policy: str) -> None:
    r"""Check the policy on how to handle existing columns.

    Args:
        exist_policy: The policy on how to handle existing columns.

    Raises:
        ValueError: if ``exist_policy`` is not ``'ignore'``,
            ``'warn'``, or ``'raise'``.

    Example usage:

    ```pycon

    >>> from analora.utils.policy import check_exist_policy
    >>> check_exist_policy("ignore")

    ```
    """
    if exist_policy not in {"ignore", "warn", "raise"}:
        msg = (
            f"Incorrect 'exist_policy': {exist_policy}. The valid values are: "
            f"'ignore', 'raise', 'warn'"
        )
        raise ValueError(msg)


def check_missing_policy(missing_policy: str) -> None:
    r"""Check the policy on how to handle missing columns.

    Args:
        missing_policy: The policy on how to handle missing columns.

    Raises:
        ValueError: if ``missing_policy`` is not ``'ignore'``,
            ``'warn'``, or ``'raise'``.

    Example usage:

    ```pycon

    >>> from analora.utils.policy import check_missing_policy
    >>> check_missing_policy("ignore")

    ```
    """
    if missing_policy not in {"ignore", "warn", "raise"}:
        msg = (
            f"Incorrect 'missing_policy': {missing_policy}. The valid values are: "
            f"'ignore', 'raise', 'warn'"
        )
        raise ValueError(msg)

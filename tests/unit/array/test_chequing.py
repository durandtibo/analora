from __future__ import annotations

import numpy as np
import pytest

from analora.array import check_square_matrix

#########################################
#     Tests for check_square_matrix     #
#########################################


def test_check_square_matrix_correct() -> None:
    check_square_matrix("my_var", np.ones((3, 3)))


def test_check_square_matrix_1d() -> None:
    with pytest.raises(ValueError, match="Incorrect 'my_var'"):
        check_square_matrix("my_var", np.ones((3,)))


def test_check_square_matrix_3d() -> None:
    with pytest.raises(ValueError, match="Incorrect 'my_var'"):
        check_square_matrix("my_var", np.ones((3, 3, 3)))


def test_check_square_matrix_not_square() -> None:
    with pytest.raises(ValueError, match="Incorrect 'my_var'"):
        check_square_matrix("my_var", np.ones((3, 4)))

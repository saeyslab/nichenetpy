"""
Here we test the evaluation metrics.
"""

import numpy as np
import pytest
from common import equals

from nichenetpy.metrics import calculate_aupr


def test_calculate_aupr_0():
    assert (
        calculate_aupr(
            (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
            (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66, 0.24, 0.42, 0.95, 0.47, 0.39, 0.07),
        )
        == 0.8851473922902493
    )


def test_calculate_aupr_1():
    assert equals(
        calculate_aupr(
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66, 0.24, 0.42, 0.95, 0.47, 0.39, 0.07),
        ),
        1.0,
    )


def test_calculate_aupr_2():
    assert np.isnan(
        calculate_aupr(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66, 0.24, 0.42, 0.95, 0.47, 0.39, 0.07),
        )
    )


def test_calculate_aupr_invalid_0():
    with pytest.raises(ValueError):
        calculate_aupr(
            (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
            (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66),
        )


def test_calculate_aupr_invalid_1():
    with pytest.raises(ValueError):
        calculate_aupr((1.0, 0.0, 1.0, 0.0, 0.0), (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66))

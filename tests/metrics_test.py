# type: ignore

"""
Here we test the evaluation metrics.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from common import equals

from nichenetpy.metrics import calculate_aupr, group_metrics


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


def _group_metrics(mat, groups, **kwargs):
    obs = pd.DataFrame({"g": groups}, index=[f"c{i}" for i in range(len(groups))])
    ann = AnnData(X=mat.copy(), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(mat.shape[1])]))
    ann.layers["data"] = mat.copy()
    group_metrics(ann, "g", **kwargs)
    return ann.uns["group_metrics"]


@pytest.mark.parametrize("kwargs", [{}, {"group_oi": "A"}, {"wilcoxon_limma": True}, {"group_ref": "B"}])
def test_group_metrics_cells_without_group(kwargs):
    # a cell without a group is left out: the same result as without that cell (it raised a TypeError), with a warning
    # unless group_ref is given (then it isn't used anyway)
    mat = np.log1p(np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0], [4.0, 0.0], [5.0, 2.0], [0.0, 1.0], [3.0, 3.0]]))
    groups = ["A", "A", "B", "B", None, "B", "C"]
    expected = _group_metrics(np.delete(mat, 4, axis=0), ["A", "A", "B", "B", "B", "C"], **kwargs)
    if "group_ref" in kwargs:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*cells without a value")
            result = _group_metrics(mat, groups, **kwargs)
    else:
        with pytest.warns(UserWarning, match="1 cells without a value in the column 'g' are left out"):
            result = _group_metrics(mat, groups, **kwargs)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

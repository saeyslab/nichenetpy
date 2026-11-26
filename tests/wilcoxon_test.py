# type: ignore

"""
Here we test some functions from nichenetpy.wilcoxon (the comparison with Seurat is in group_metrics_seurat_test.py,
the limma version in wilcoxon_limma_test.py).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from nichenetpy.wilcoxon import wilcoxon_rank_sum_test


def _ann(mat, groups):
    obs = pd.DataFrame({"g": groups}, index=[f"c{i}" for i in range(len(groups))])
    ann = AnnData(X=mat.copy(), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(mat.shape[1])]))
    ann.layers["data"] = mat.copy()
    return ann


@pytest.mark.parametrize("missing_value", [None, np.nan])
@pytest.mark.parametrize("zero_row", [False, True])
def test_wilcoxon_rank_sum_test_cells_without_group(missing_value, zero_row):
    # a cell without a group is left out (with a warning): the same p-values as without that cell; it gave a KeyError
    # (non-zero values) or wrong p-values (only zeros)
    mat = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0], [4.0, 0.0], [5.0, 2.0], [0.0, 1.0]])
    if zero_row:
        mat[4] = 0
    groups = ["A", "A", "B", "B", missing_value, "B"]
    expected = wilcoxon_rank_sum_test(_ann(np.delete(mat, 4, axis=0), ["A", "A", "B", "B", "B"]), "g")
    for column in (groups, pd.Categorical(groups)):
        with pytest.warns(UserWarning, match="1 cells without a value in the column 'g' are left out"):
            result = wilcoxon_rank_sum_test(_ann(mat, column), "g")
        assert result == expected

# type: ignore

"""
Here we test some functions from nichenetpy.utils
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from nichenetpy.utils import (
    _ligand_sort_key,
    combine_by_key,
    combine_dicts,
    df_grouped_apply,
    ligand_activities_df,
    remove_zero_rows_cols,
    subset_matrix,
)


def matrix_equal(m1, m2):
    if type(m1) is np.ndarray:
        if m1.shape != m2.shape:
            return False
        if m1.shape == 1:
            for e1, e2 in zip(m1[0], m2[0], strict=True):
                if e1 != e2:
                    return False
            return True
        else:
            for e1, e2 in zip(m1, m2, strict=True):
                if not matrix_equal(e1, e2):
                    return False
            return True
    else:
        return m1 == m2


def subset_matrix_template(input, exp):
    res = subset_matrix(*input)
    assert matrix_equal(res, exp), f"expected {exp}, got {res}"


def test_subset_matrix_index_0():
    nrows = 6
    ncols = 5
    subset_matrix_template(
        (np.array(range(nrows * ncols)).reshape(nrows, ncols), [0, 2, 5], [1, 2]),
        np.array([[1, 2], [11, 12], [26, 27]]),
    )


def test_subset_matrix_bool_0():
    nrows = 5
    ncols = 4
    subset_matrix_template(
        (
            np.array(range(nrows * ncols)).reshape(nrows, ncols),
            [False, True, False, True, True],
            [True, False, False, True],
        ),
        np.array([[4, 7], [12, 15], [16, 19]]),
    )


def test_subset_matrix_rows_0():
    nrows = 6
    ncols = 5
    subset_matrix_template(
        (np.array(range(nrows * ncols)).reshape(nrows, ncols), [0, 2, 5], None),
        np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [25, 26, 27, 28, 29]]),
    )


def test_subset_matrix_cols_0():
    nrows = 6
    ncols = 5
    subset_matrix_template(
        (np.array(range(nrows * ncols)).reshape(nrows, ncols), None, [1, 2]),
        np.array([[1, 2], [6, 7], [11, 12], [16, 17], [21, 22], [26, 27]]),
    )


def subset_matrix_sparse_template(input, exp):
    res = np.array(subset_matrix(*input).todense())
    assert matrix_equal(res, exp), f"expected {exp}, got {res}"


def test_subset_matrix_csc_index_0():
    nrows = 6
    ncols = 5
    subset_matrix_sparse_template(
        (csc_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)), [0, 2, 5], [1, 2]),
        np.array([[1, 2], [11, 12], [26, 27]]),
    )


def test_subset_matrix_csc_bool_0():
    nrows = 5
    ncols = 4
    subset_matrix_sparse_template(
        (
            csc_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)),
            [False, True, False, True, True],
            [True, False, False, True],
        ),
        np.array([[4, 7], [12, 15], [16, 19]]),
    )


def test_subset_matrix_csc_rows_0():
    nrows = 6
    ncols = 5
    subset_matrix_sparse_template(
        (csc_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)), [0, 2, 5], None),
        np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [25, 26, 27, 28, 29]]),
    )


def test_subset_matrix_csc_cols_0():
    nrows = 6
    ncols = 5
    subset_matrix_sparse_template(
        (csc_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)), None, [1, 2]),
        np.array([[1, 2], [6, 7], [11, 12], [16, 17], [21, 22], [26, 27]]),
    )


def test_subset_matrix_csr_index_0():
    nrows = 6
    ncols = 5
    subset_matrix_sparse_template(
        (csr_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)), [0, 2, 5], [1, 2]),
        np.array([[1, 2], [11, 12], [26, 27]]),
    )


def test_subset_matrix_csr_bool_0():
    nrows = 5
    ncols = 4
    subset_matrix_sparse_template(
        (
            csr_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)),
            [False, True, False, True, True],
            [True, False, False, True],
        ),
        np.array([[4, 7], [12, 15], [16, 19]]),
    )


def test_subset_matrix_csr_rows_0():
    nrows = 6
    ncols = 5
    subset_matrix_sparse_template(
        (csr_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)), [0, 2, 5], None),
        np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [25, 26, 27, 28, 29]]),
    )


def test_subset_matrix_csr_cols_0():
    nrows = 6
    ncols = 5
    subset_matrix_sparse_template(
        (csr_matrix(np.array(range(nrows * ncols)).reshape(nrows, ncols)), None, [1, 2]),
        np.array([[1, 2], [6, 7], [11, 12], [16, 17], [21, 22], [26, 27]]),
    )


def test_subset_data_format_ndarray_0():
    res = subset_matrix(
        np.array(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]],
            order="C",
        ),
        rows=(1, 3, 4),
        cols=(1, 3),
    )
    assert res.flags.c_contiguous


def test_subset_data_format_ndarray_1():
    res = subset_matrix(
        np.array(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]],
            order="F",
        ),
        rows=(1, 3, 4),
        cols=(1, 3),
    )
    assert res.flags.c_contiguous


def test_subset_data_format_ndarray_2():
    res = subset_matrix(
        np.array(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]],
            order="C",
        ),
        rows=(1, 3, 4),
    )
    assert res.flags.c_contiguous


def test_subset_data_format_ndarray_3():
    res = subset_matrix(
        np.array(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]],
            order="F",
        ),
        rows=(1, 3, 4),
    )
    assert res.flags.c_contiguous


def test_subset_data_format_ndarray_4():
    res = subset_matrix(
        np.array(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]],
            order="C",
        ),
        cols=(1, 3),
    )
    assert res.flags.f_contiguous


def test_subset_data_format_ndarray_5():
    res = subset_matrix(
        np.array(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]],
            order="F",
        ),
        cols=(1, 3),
    )
    assert res.flags.f_contiguous


def test_subset_data_format_csr_0():
    res = subset_matrix(
        csr_matrix(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]]
        ),
        rows=(1, 3, 4),
        cols=(1, 3),
    )
    assert type(res) is csr_matrix


def test_subset_data_format_csr_1():
    res = subset_matrix(
        csr_matrix(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]]
        ),
        rows=(1, 3, 4),
    )
    assert type(res) is csr_matrix


def test_subset_data_format_csr_2():
    res = subset_matrix(
        csr_matrix(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]]
        ),
        cols=(1, 3),
    )
    assert type(res) is csr_matrix


def test_subset_data_format_csc_0():
    res = subset_matrix(
        csc_matrix(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]]
        ),
        rows=(1, 3, 4),
        cols=(1, 3),
    )
    assert type(res) is csc_matrix


def test_subset_data_format_csc_1():
    res = subset_matrix(
        csc_matrix(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]]
        ),
        rows=(1, 3, 4),
    )
    assert type(res) is csc_matrix


def test_subset_data_format_csc_2():
    res = subset_matrix(
        csc_matrix(
            [[1, 2, 27, 0], [6, 7, 4, 42], [11, 12, 25, 32], [16, 17, 45, 36], [21, 22, 50, 31], [26, 27, 31, 29]]
        ),
        cols=(1, 3),
    )
    assert type(res) is csc_matrix


def remove_zero_rows_cols_template(input, exp):
    res = remove_zero_rows_cols(input)
    assert matrix_equal(res, exp), f"expected {exp}, got {res}"


def test_remove_zero_rows_cols_0():
    remove_zero_rows_cols_template(
        np.array(
            [
                [0, 4, 1, 0, 6, 0, 0],
                [2, 0, 0, 0, 0, 0, 0],
                [4, 1, 0, 0, 4, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 7, 0, 0, 6, 0, 2],
            ]
        ),
        np.array([[0, 4, 1, 6, 0], [2, 0, 0, 0, 0], [4, 1, 0, 4, 2], [0, 7, 0, 6, 2]]),
    )


def combine_by_key_template(input, exp):
    res = combine_by_key(*input)
    assert len(res) == len(exp), f"expected to have size {len(exp)}, got {len(res)}"
    for key in exp.keys():
        assert res[key] == exp[key], f"expected {exp[key]} for key={key}, got {res[key]}"


def test_combine_by_key_0():
    combine_by_key_template(
        [
            (["a", "b", "c", "d"], [4, 1, 8, 5]),
            (["b", "c", "d", "a"], [7, 3, 2, 1]),
            (["b", "a", "c", "d"], [9, 6, 5, 3]),
        ],
        {"a": [4, 1, 6], "b": [1, 7, 9], "c": [8, 3, 5], "d": [5, 2, 3]},
    )


def combine_dicts_template(input, exp):
    res = combine_dicts(*input)
    assert len(res) == len(exp), f"expected to have size {len(exp)}, got {len(res)}"
    for key in exp.keys():
        assert res[key] == exp[key], f"expected {exp[key]} for key={key}, got {res[key]}"


def test_combine_dicts_0():
    combine_dicts_template(
        ({"a": 0, "b": 1, "c": 2, "d": 3}, {"a": 4, "b": 5, "c": 6, "d": 7}),
        {"a": (0, 4), "b": (1, 5), "c": (2, 6), "d": (3, 7)},
    )


def test_combine_dicts_1():
    combine_dicts_template(
        ({"a": 0, "b": 1, "c": 2, "d": 3}, {"a": 4, "b": 5, "c": 6, "e": 7}),
        {
            "a": (0, 4),
            "b": (1, 5),
            "c": (2, 6),
        },
    )


@pytest.mark.parametrize("group", [["a", "b", np.nan, "a"], [1.0, 2.0, np.nan, 1.0]])
def test_df_grouped_apply_missing_group(group):
    df = pd.DataFrame({"group": group, "value": [1.0, 2.0, 3.0, 4.0]})
    result = df_grouped_apply(df, "group", lambda g: g["value"] - g["value"].mean(), "centered")
    # every row is kept, the missing value is a group of its own (last)
    assert result.index.tolist() == [0, 3, 1, 2]
    assert result["centered"].tolist() == [-1.5, 1.5, 0.0, 0.0]


def test_ligand_activities_df_metric_order():
    # every value in its own column, whatever the order of a ligand's metrics
    df = ligand_activities_df({"A": {"x": 1, "y": 2}, "B": {"y": 3, "x": 4}})
    assert df.columns.tolist() == ["x", "y"]
    assert df.loc["A"].tolist() == [1, 2]
    assert df.loc["B"].tolist() == [4, 3]
    assert df.index.name == "ligand"


def test_ligand_activities_df_inputs():
    activities = {"A": {"x": 1, "y": 2}, "B": {"x": 3, "y": 4}}
    expected = ligand_activities_df(activities)
    # (ligand, metrics) pairs, also as a generator, and a dict subclass give the same data frame
    pd.testing.assert_frame_equal(ligand_activities_df(list(activities.items())), expected)
    pd.testing.assert_frame_equal(ligand_activities_df(pair for pair in activities.items()), expected)
    pd.testing.assert_frame_equal(ligand_activities_df(OrderedDict(activities)), expected)
    # without ligands an empty data frame
    for empty in ({}, []):
        df = ligand_activities_df(empty)
        assert len(df) == 0
        assert df.index.name == "ligand"


def test_ligand_activities_df_different_metrics():
    with pytest.raises(ValueError, match="same metrics"):
        ligand_activities_df({"A": {"x": 1, "y": 2}, "B": {"x": 3}})
    with pytest.raises(ValueError, match="same metrics"):
        ligand_activities_df({"A": {"x": 1}, "B": {"x": 3, "z": 4}})


def test_combine_dicts_dict_subclass():
    dict1, dict2 = {"a": 1, "b": 2}, {"a": 3, "b": 4}
    assert combine_dicts(OrderedDict(dict1), OrderedDict(dict2)) == combine_dicts(dict1, dict2)


def test_ligand_sort_key():
    # ligands are sorted by their members: a combination right after its first member, int ids numerically
    assert sorted({"B", ("A", "B"), "A", ("A", "C")}, key=_ligand_sort_key) == ["A", ("A", "B"), ("A", "C"), "B"]
    assert sorted({10, 2, (2, 10), 1}, key=_ligand_sort_key) == [1, 2, (2, 10), 10]

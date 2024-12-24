from nichenetpy.utils import subset_matrix, remove_zero_rows_cols, combine_by_key

import numpy as np

def matrix_equal(m1, m2):
    if type(m1) is np.ndarray:
        for e1, e2 in zip(m1, m2):
            if matrix_equal(e1, e2):
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
        (
            np.array(range(nrows*ncols)).reshape(nrows, ncols),
            [0, 2, 5],
            [1, 2]
        ),
        np.array([
            [1, 2],
            [11, 12],
            [26, 27]
        ])
    )

def test_subset_matrix_bool_0():
    nrows = 5
    ncols = 4
    subset_matrix_template(
        (
            np.array(range(nrows*ncols)).reshape(nrows, ncols),
            [False, True, False, True, True],
            [True, False, False, True]
        ),
        np.array([
            [4, 7],
            [12, 15],
            [16, 19]
        ])
    )

def remove_zero_rows_cols_template(input, exp):
    res = remove_zero_rows_cols(input)
    assert matrix_equal(res, exp), f"expected {exp}, got {res}"

def test_remove_zero_rows_cols_0():
    remove_zero_rows_cols_template(
        np.array([
            [0, 4, 1, 0, 6, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [4, 1, 0, 0, 4, 0, 2],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 6, 0, 2]
        ]),
        np.array([
            [0, 4, 1, 6, 0],
            [2, 0, 0, 0, 0],
            [4, 1, 0, 4, 2],
            [0, 7, 0, 6, 2]
        ])
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
            (["b", "a", "c", "d"], [9, 6, 5, 3])
        ],
        {
            "a": [4, 1, 6],
            "b": [1, 7, 9],
            "c": [8, 3, 5],
            "d": [5, 2, 3]
        }
    )
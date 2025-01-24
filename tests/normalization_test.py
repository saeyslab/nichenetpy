from nichenetpy.normalization import scale_quantile, scale_quantile_adapted

import numpy as np


err_bound = 0.001

def equals_iter(xs, ys):
    for x, y in zip(xs, ys):
        if abs(x - y) > err_bound:
            return False
    return True

def equals_ndarray(xs, ys):
    return equals_iter(xs.reshape(-1), ys.reshape(-1))

def template_scale_quantile(data, cutoff, exp, equals=equals_iter):
    res = scale_quantile(data, cutoff)
    print(res)
    assert equals(res, exp), f"the following output is incorrect: {res}"

def test_scale_quantile_list_0():
    template_scale_quantile(
        [0.1, 0.25, 0.3 ,0.5 ,0.75, 0.9, 1.2],
        0.2,
        [0, 0, 0.066, 0.393, 0.803, 1, 1]
    )

def test_scale_quantile_list_1():
    template_scale_quantile(
        [
            [0.1, 0.25, 0.3 ,0.5 ,0.75, 0.9, 1.2],
            [0.2, 0.28, 0.3 , 0.52, 0.7, 0.81, 1.1]
        ],
        0.2,
        np.array([
            [0, 0, 0.066, 0.393, 0.803, 1, 1],
            [0, 0, 0.032, 0.468, 0.825, 1, 1]
        ]),
        equals=equals_ndarray
    )

def test_scale_quantile_ndarray_0():
    template_scale_quantile(
        np.array([0.1, 0.25, 0.3 ,0.5 ,0.75, 0.9, 1.2]),
        0.2,
        np.array([0, 0, 0.066, 0.393, 0.803, 1, 1])
    )

def test_scale_quantile_ndarray_1():
    template_scale_quantile(
        np.array([
            [0.1, 0.25, 0.3 ,0.5 ,0.75, 0.9, 1.2],
            [0.2, 0.28, 0.3 , 0.52, 0.7, 0.81, 1.1]
        ]),
        0.2,
        np.array([
            [0, 0, 0.066, 0.393, 0.803, 1, 1],
            [0, 0, 0.032, 0.468, 0.825, 1, 1]
        ]),
        equals=equals_ndarray
    )

def template_scale_quantile_adapted(data, cutoff, exp, equals=equals_iter):
    res = scale_quantile_adapted(data, cutoff)
    print(res)
    assert equals(res, exp), f"the following output is incorrect: {res}"

def test_scale_quantile_adapted_std_0():
    template_scale_quantile_adapted(
        [1, 5, 9, 16, 20, 32, 45, 50, 54, 71],
        0,
        [0.001, 0.058, 0.115, 0.215, 0.272, 0.444, 0.630, 0.701, 0.758, 1.001]
    )
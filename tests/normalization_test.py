from nichenetpy.normalization import scale_quantile

import os
import numpy as np


err_bound = 0.001

def equals(xs, ys):
    for x, y in zip(xs, ys):
        if abs(x - y) > err_bound:
            return False
    return True

def template_scale_quantile(data, cutoff, exp):
    res = scale_quantile(data, cutoff)
    assert equals(res, exp), f"the following output is incorrect: {res}"

def test_scale_quantile_list_0():
    template_scale_quantile(
        [0.1, 0.25, 0.3 ,0.5 ,0.75, 0.9, 1.2],
        0.2,
        [0, 0, 0.066, 0.393, 0.803, 1, 1]
    )

def test_scale_quantile_ndarray_0():
    template_scale_quantile(
        np.array([0.1, 0.25, 0.3 ,0.5 ,0.75, 0.9, 1.2]),
        0.2,
        np.array([0, 0, 0.066, 0.393, 0.803, 1, 1])
    )
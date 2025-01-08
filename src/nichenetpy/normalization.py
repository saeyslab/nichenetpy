from scipy.sparse import lil_matrix

import numpy as np


def relative_counts(data):
    mp = [10000/float(e) for e in data.sum(axis=1)]
    mp_mat = lil_matrix((len(mp), len(mp)))
    mp_mat.setdiag(mp)
    return mp_mat * data

def log_normalize(data):
    return np.log1p(relative_counts(data))
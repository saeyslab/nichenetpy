from scipy.sparse import lil_matrix, csc_matrix, csr_matrix

import numpy as np


def relative_counts(
    data:np.ndarray|csc_matrix|csr_matrix,
    scale_factor:float=10000
) -> np.ndarray|csc_matrix|csr_matrix:
    '''
    Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale_factor.

    Parameters
    ----------
    data : np.ndarray or np.csc_matrix or np.csr.matrix
        the data to normalize
    scale_factor : float
        the multiplier
    
    Returns
    -------
    ndarray or csc_matrix or csr_matrix
        the normalized data
    '''
    mp = [scale_factor/float(e) for e in data.sum(axis=1)]
    mp_mat = lil_matrix((len(mp), len(mp)))
    mp_mat.setdiag(mp)
    return mp_mat * data

def log_normalize(
    data:np.ndarray|csc_matrix|csr_matrix,
    scale_factor:float=10000
) -> np.ndarray|csc_matrix|csr_matrix:
    '''
    Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale.factor.
    This is then natural-log transformed using log1p

    Parameters
    ----------
    data : np.ndarray or np.csc_matrix or np.csr.matrix
        the data to normalize
    scale_factor : float
        the multiplier
    
    Returns
    -------
    ndarray or csc_matrix or csr_matrix
        the normalized data
    '''
    return np.log1p(relative_counts(data, scale_factor))

def _scale_quantile_seq(data:list|tuple|np.ndarray, cutoff:float=0.05) -> np.ndarray:
    qs = (np.quantile(data, cutoff), np.quantile(data, 1 - cutoff))
    sub = qs[0]
    div = 1 if qs[0] == qs[1] else qs[1] - qs[0]
    if type(data) is list or type(data) is tuple:
        output = [(x - sub) / div for x in data]
    else:
        output = (data - sub) / div
    return np.clip(output, 0, 1)

def scale_quantile(data:list|tuple|np.ndarray, cutoff:float=0.05) -> np.ndarray:
    if type(data) is list or type(data) is tuple:
        if len(data) > 0:
            if type(data[0]) is list or type(data[0]) is tuple:
                return np.concatenate([_scale_quantile_seq(row, cutoff=cutoff).reshape(1, -1) for row in data], axis=0)
            elif type(data[0]) is float or type(data[0]) is int:
                return _scale_quantile_seq(data, cutoff=cutoff)
            else:
                raise ValueError(f"data should be a 1-dimensional or 2-dimensional array of numbers")
    elif type(data) is np.ndarray:
        if len(data.shape) == 2:
            return np.concatenate([_scale_quantile_seq(data[i, :], cutoff=cutoff).reshape(1, -1) for i in range(data.shape[0])], axis=0)
        elif len(data.shape) == 1:
            return _scale_quantile_seq(data, cutoff=cutoff)
        else:
            raise ValueError(f"data should be 1-dimensional or 2-dimensional, shape was {data.shape}")
    else:
        raise TypeError(f"data should be of type list, tuple or numpy.ndarray, was {type(data)}")

def scale_quantile_adapted(data:np.ndarray, cutoff=0):
    return scale_quantile(data, cutoff=cutoff) + 0.001

def scaling_zscore(data:list):
    if len(data) == 1:
        return [0]
    sd = np.std(data)
    avg = np.mean(data)
    return [(x - avg) / sd for x in data] if sd > 0 else [x - avg for x in data]
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix

from numbers import Number

import numpy as np
import pandas as pd


def relative_counts(
    data:np.ndarray|csc_matrix|csr_matrix,
    scale_factor:float=10000
) -> np.ndarray|csc_matrix|csr_matrix:
    '''
    Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale_factor.

    Parameters
    ----------
    data : numpy.ndarray or scipy.csc_matrix or scipy.csr.matrix
        the data to normalize
    scale_factor : float
        the multiplier
    
    Returns
    -------
    numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
        the normalized data
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(data) is not np.ndarray and type(data) is not csc_matrix and type(data) is not csr_matrix:
        raise TypeError(f"data should have type numpy.ndarray, scipy.csc_matrix or scipy.csr_matrix, was {type(data)}")
    if not isinstance(scale_factor, Number):
        raise TypeError(f"scale_factor should have type float, was {type(scale_factor)}")
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
    data : numpy.ndarray or scipy.csc_matrix or scipy.csr.matrix
        the data to normalize
    scale_factor : float
        the multiplier
    
    Returns
    -------
    numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
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

def scale_quantile(
    data:list|tuple|np.ndarray|pd.Series,
    cutoff:float=0.05,
    by_row:bool=False
) -> np.ndarray:
    '''
    Cut off outer quantiles and rescale to a [0, 1] range. 

    Parameters
    ----------
    data : list or tuple or numpy.ndarray or pandas.Series
        the data to normalize
    cutoff : float
        the quantile cutoff for outliers
    by_row : bool
        indicates whether to apply the computation on each row or on each column
    
    Returns
    -------
    numpy.ndarray
        the normalized data
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if data is invalid
    '''
    if not isinstance(cutoff, Number):
        raise TypeError(f"cutoff should have type float, was {type(cutoff)}")
    if type(by_row) is not bool:
        raise TypeError(f"by_row should have type bool, was {type(by_row)}")
    if type(data) is pd.Series:
        data = data.to_numpy()
    if type(data) is list or type(data) is tuple:
        if len(data) > 0:
            if type(data[0]) is list or type(data[0]) is tuple:
                if by_row:
                    return np.concatenate([_scale_quantile_seq(row, cutoff=cutoff).reshape(1, -1) for row in data], axis=0)
                else:
                    return np.concatenate([_scale_quantile_seq(col, cutoff=cutoff).reshape(-1, 1) for col in zip(*data)], axis=1)
            elif type(data[0]) is float or type(data[0]) is int:
                return _scale_quantile_seq(data, cutoff=cutoff)
            else:
                raise ValueError(f"data should be a 1-dimensional or 2-dimensional array of numbers")
    elif type(data) is np.ndarray:
        if len(data.shape) == 2:
            if by_row:
                return np.concatenate(
                    [_scale_quantile_seq(data[i, :], cutoff=cutoff).reshape(1, -1) for i in range(data.shape[0])],
                    axis=0
                )
            else:
                return np.concatenate(
                    [_scale_quantile_seq(data[:, i], cutoff=cutoff).reshape(-1, 1) for i in range(data.shape[1])],
                    axis=1
                )
        elif len(data.shape) == 1:
            return _scale_quantile_seq(data, cutoff=cutoff)
        else:
            raise ValueError(f"data should be 1-dimensional or 2-dimensional, shape was {data.shape}")
    else:
        raise TypeError(f"data should be of type list, tuple, numpy.ndarray or pandas.Series, was {type(data)}")

def scale_quantile_adapted(
    data:list|tuple|np.ndarray|pd.Series,
    cutoff=0,
    by_row:bool=False
) -> np.ndarray:
    '''
    Normalize values in a vector by quantile scaling. Add a pseudovalue of 0.001 to avoid having a score of 0 for the lowest value.

    Parameters
    ----------
    data : list or tuple or numpy.ndarray or pandas.Series
        the data to normalize
    cutoff : float
        the quantile cutoff for outliers
    by_row : bool
        indicates whether to apply the computation on each row or on each column
    
    Returns
    -------
    numpy.ndarray
        the normalized data
    
    Raises
    ------
    TypeError
        if data has the wrong type
    ValueError
        if data is invalid
    '''
    return scale_quantile(data, cutoff=cutoff, by_row=by_row) + 0.001

def scaling_zscore(data:list[float]) -> list[float]:
    '''
    Normalize values in a vector by the z-score method.

    Parameters
    ----------
    data : list of float
        the data to normalize
    
    Returns
    -------
    list of float
        the normalized data
    '''
    if len(data) == 1:
        return [0]
    sd = np.std(data)
    avg = np.mean(data)
    return [(x - avg) / sd for x in data] if sd > 0 else [x - avg for x in data]
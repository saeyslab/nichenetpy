from nichenetpy.typing import nichenet_matrix

from scipy.sparse import lil_matrix
from scipy.stats import median_abs_deviation
from numbers import Number

import numpy as np
import pandas as pd


def relative_counts(
    data:nichenet_matrix,
    scale_factor:float=10000
) -> nichenet_matrix:
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
    if not isinstance(data, nichenet_matrix):
        raise TypeError(f"data should have type numpy.ndarray, scipy.csc_matrix or scipy.csr_matrix, was {type(data)}")
    if not isinstance(scale_factor, Number):
        raise TypeError(f"scale_factor should have type float, was {type(scale_factor)}")
    mp = scale_factor/np.array(data.sum(axis=1)).reshape((-1,))
    mp_mat = lil_matrix((len(mp), len(mp)))
    mp_mat.setdiag(mp)
    return mp_mat * data

def log_normalize(
    data:nichenet_matrix,
    scale_factor:float=10000
) -> nichenet_matrix:
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
    by_row:bool=False,
    pseudo_count=0.001
) -> np.ndarray:
    '''
    Normalize values in a vector by quantile scaling. Add a pseudovalue to avoid having a score of 0 for the lowest value.

    Parameters
    ----------
    data : list or tuple or numpy.ndarray or pandas.Series
        the data to normalize
    cutoff : float
        the quantile cutoff for outliers
    by_row : bool
        indicates whether to apply the computation on each row or on each column
    pseudo_count : float
        a small value to add in order to prevent 0 values
    
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
    if not isinstance(pseudo_count, Number):
        raise TypeError(f"pseudo_count should have type float, was {type(pseudo_count)}")
    return scale_quantile(data, cutoff=cutoff, by_row=by_row) + pseudo_count

def scaling_zscore(data:list[float]|np.ndarray|pd.Series) -> list[float]|np.ndarray|pd.Series:
    '''
    Normalize values in a vector by the z-score method.

    Parameters
    ----------
    data : list of float or numpy.ndarray or pandas.Series
        the data to normalize
    
    Returns
    -------
    list of float or numpy.ndarray or pandas.Series
        the normalized data
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(data) is list:
        if len(data) == 1:
            return [0]
        sd = np.std(data)
        avg = np.mean(data)
        return [(x - avg) / sd for x in data] if sd > 0 else [x - avg for x in data]
    elif type(data) is np.ndarray or type(data) is pd.Series:
        if len(data) == 1:
            return np.array([0])
        sd = np.std(data)
        avg = np.mean(data)
        return (data - avg) / sd if sd > 0 else data - avg
    else:
        raise TypeError(f"data should have type list[float] or numpy.ndarray or pandas.Series, was {type(data)}")

def scaling_modified_zscore(
    data:list[float]|np.ndarray|pd.Series,
    scale_factor:float=0.6744907594765952
) -> list[float]|np.ndarray|pd.Series:
    '''
    Normalize values by the modified z-score method (uses median and median absolute deviation instead of mean)

    Parameters
    ----------
    data : list of float or numpy.ndarray or pandas.Series
        the data to normalize
    
    scale_factor : float
        the scale factor used in the computation
    
    Returns
    -------
    list of float or numpy.ndarray or pandas.Series
        the normalized data
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(data) is list:
        md = np.median(data)
        if median_abs_deviation(data, nan_policy="omit", scale=scale_factor) == 0:
            return [0.6745 * (x - md) for x in data]
        else:
            mad = median_abs_deviation(data, scale=scale_factor)
            return [0.6745 * (x - md) / mad for x in data]
    elif type(data) is np.ndarray or type(data) is pd.Series:
        md = np.median(data)
        if median_abs_deviation(data, nan_policy="omit", scale=scale_factor) == 0:
            return 0.6745 * (data - md)
        else:
            mad = median_abs_deviation(data, scale=scale_factor) # TODO: check if multiplication can be removed and scale set to 1
            return 0.6745 * (data - md) / mad
    else:
        raise TypeError(f"data should have type list[float] or numpy.ndarray or pandas.Series, was {type(data)}")
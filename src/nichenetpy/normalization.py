from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import diags_array
from scipy.stats import median_abs_deviation

from nichenetpy.typing import nichenet_matrix, validate_matrix_type


def relative_counts(data: nichenet_matrix, scale_factor: float = 10000) -> nichenet_matrix:
    """
    Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale_factor.
    Cells without counts stay 0 (as in Seurat and scanpy).

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix or scipy.sparse.csc_matrix or scipy.sparse.csr_array or scipy.sparse.csc_array
        the data to normalize, cells x features
    scale_factor : float
        the multiplier

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix or scipy.sparse.csc_matrix or scipy.sparse.csr_array or scipy.sparse.csc_array
        the normalized data, of the same type as data

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    validate_matrix_type(data, f"data should have type nichenet_matrix, was {type(data)}")
    if not isinstance(scale_factor, Number):
        raise TypeError(f"scale_factor should have type float, was {type(scale_factor)}")
    totals = np.asarray(data.sum(axis=1), dtype=np.float64).reshape(-1)
    # the multiplier of each cell, 0 for cells without counts
    mp = np.divide(scale_factor, totals, out=np.zeros(len(totals)), where=totals != 0)
    if isinstance(data, np.ndarray):
        return data * mp[:, None]
    return type(data)(diags_array(mp) @ data)  # pyright: ignore[reportCallIssue, reportArgumentType] # same sparse type as data


def log_normalize(data: nichenet_matrix, scale_factor: float = 10000) -> nichenet_matrix:
    """
    Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale.factor.
    This is then natural-log transformed using `numpy.log1p`

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr.matrix
        the data to normalize
    scale_factor : float
        the multiplier

    Returns
    -------
    numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        the normalized data
    """
    data_rc = relative_counts(data, scale_factor)
    if isinstance(data_rc, np.ndarray):
        return np.log1p(data_rc)
    else:
        return data_rc.log1p()  # pyright: ignore[reportReturnType] # log1p exists for all spmatrix


def _scale_quantile_seq(
    data: np.ndarray[Any, np.dtype[np.floating]], cutoff: float = 0.05
) -> np.ndarray[Any, np.dtype[np.floating]]:
    qs: tuple[float, float] = (np.quantile(data, cutoff), np.quantile(data, 1 - cutoff))
    sub = qs[0]
    div = 1 if qs[0] == qs[1] else qs[1] - qs[0]
    return np.clip((data - sub) / div, 0, 1)


def scale_quantile(
    data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray[Any, np.dtype[np.floating]] | pd.Series[float],
    cutoff: float = 0.05,
    by_row: bool = False,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """
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
    """
    if not isinstance(cutoff, Number):
        raise TypeError(f"cutoff should have type float, was {type(cutoff)}")
    if type(by_row) is not bool:
        raise TypeError(f"by_row should have type bool, was {type(by_row)}")
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    if isinstance(data, Sequence):
        if len(data) == 0:
            raise ValueError("data was empty")
        try:
            # also accepts numpy numbers and lists of 1-dimensional arrays (rows)
            data = np.asarray(data)
        except ValueError as error:  # rows of different lengths
            raise ValueError("data should be a 1-dimensional or 2-dimensional array of numbers") from error
    if type(data) is np.ndarray:
        if data.dtype.kind == "O":  # e.g. a pandas Series with a nullable dtype (Float64, Int64)
            try:
                data = data.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("data should be a 1-dimensional or 2-dimensional array of numbers") from error
        if data.dtype.kind not in "iuf":  # bool, complex and str arrays
            raise ValueError("data should be a 1-dimensional or 2-dimensional array of numbers")
        if len(data.shape) == 2:
            if by_row:
                return np.concatenate(
                    [_scale_quantile_seq(data[i, :], cutoff=cutoff).reshape(1, -1) for i in range(data.shape[0])],
                    axis=0,
                )
            else:
                return np.concatenate(
                    [_scale_quantile_seq(data[:, i], cutoff=cutoff).reshape(-1, 1) for i in range(data.shape[1])],
                    axis=1,
                )
        elif len(data.shape) == 1:
            return _scale_quantile_seq(data, cutoff=cutoff)
        else:
            raise ValueError(f"data should be 1-dimensional or 2-dimensional, shape was {data.shape}")
    else:
        raise TypeError(f"data should be of type Sequence, numpy.ndarray or pandas.Series, was {type(data)}")


def scale_quantile_adapted(
    data: Sequence[float] | np.ndarray[Any, np.dtype[np.floating]] | pd.Series[float],
    cutoff: float = 0,
    by_row: bool = False,
    pseudo_count: float = 0.001,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """
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
    """
    if not isinstance(pseudo_count, Number):
        raise TypeError(f"pseudo_count should have type float, was {type(pseudo_count)}")
    return scale_quantile(data, cutoff=cutoff, by_row=by_row) + pseudo_count


def scaling_zscore(
    data: Sequence[float] | np.ndarray[Any, np.dtype[np.floating]] | pd.Series[float],
) -> Sequence[float] | np.ndarray[Any, np.dtype[np.floating]] | pd.Series[float]:
    """
    Normalize values in a vector by the z-score method.

    Parameters
    ----------
    data : Sequence of float or numpy.ndarray or pandas.Series
        the data to normalize

    Returns
    -------
    Sequence of float or numpy.ndarray or pandas.Series
        the normalized data

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(data) is list:
        if len(data) == 1:
            return [0]
        sd = np.std(data)
        avg = np.mean(data)
        return [(x - avg) / sd for x in data] if sd > 0 else [0 for _ in data]
    elif isinstance(data, (np.ndarray, pd.Series)):
        if len(data) == 1:
            return np.array([0])
        sd = np.std(data)
        avg = np.mean(data)
        return (data - avg) / sd if sd > 0 else data - avg
    else:
        raise TypeError(f"data should have type Sequence[float] or numpy.ndarray or pandas.Series, was {type(data)}")


def scaling_modified_zscore(
    data: Sequence[float] | np.ndarray[Any, np.dtype[np.floating]] | pd.Series[float],
    scale_factor: float = 0.6744907594765952,
) -> Sequence[float] | np.ndarray[Any, np.dtype[np.floating]] | pd.Series[float]:
    """
    Normalize values by the modified z-score method (uses median and median absolute deviation instead of mean)

    Parameters
    ----------
    data : Sequence of float or numpy.ndarray or pandas.Series
        the data to normalize

    scale_factor : float
        the scale factor used in the computation

    Returns
    -------
    Sequence of float or numpy.ndarray or pandas.Series
        the normalized data

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if isinstance(data, Sequence):
        md = np.median(data)
        if median_abs_deviation(data, nan_policy="omit", scale=scale_factor) == 0:
            return [scale_factor * (x - md) for x in data]
        else:
            mad = median_abs_deviation(data, scale=scale_factor)
            return [scale_factor * (x - md) / mad for x in data]
    elif isinstance(data, (np.ndarray, pd.Series)):
        md = np.median(data)
        if median_abs_deviation(data, nan_policy="omit", scale=scale_factor) == 0:
            return scale_factor * (data - md)
        else:
            mad = median_abs_deviation(data, scale=scale_factor)
            return scale_factor * (data - md) / mad
    else:
        raise TypeError(f"data should have type Sequence[float] or numpy.ndarray or pandas.Series, was {type(data)}")

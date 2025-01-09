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
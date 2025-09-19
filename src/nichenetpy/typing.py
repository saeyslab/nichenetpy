from typing import TypeAlias
from scipy.sparse import csr_matrix, csc_matrix
from numpy import ndarray


nichenet_matrix : TypeAlias = csr_matrix|csc_matrix|ndarray
'''
Matrix types that are supported in NicheNetPy. 
'''
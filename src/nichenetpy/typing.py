from typing import TypeAlias

from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix

nichenet_matrix: TypeAlias = csr_matrix | csc_matrix | ndarray
"""
Matrix types that are supported in NicheNetPy.
"""

gene_t: TypeAlias = str | int
"""
Gene symbol types that are supported in NicheNetPy.
"""

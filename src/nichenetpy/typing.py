from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix
from typing import TypeVar

type nichenet_matrix = csr_matrix | csc_matrix | ndarray
"""Matrix types that are supported in NicheNetPy."""

type gene_t = str | int
"""Gene symbol types that are supported in NicheNetPy."""

GeneSymbol = TypeVar('GeneSymbol', str, int)
"""Gene symbol types that are supported in NicheNetPy."""

NNMatrix = TypeVar('NNMatrix', str, int)
"""Matrix types that are supported in NicheNetPy."""

from typing import TypeVar

from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix, csr_array, csc_array

type nichenet_matrix = csr_matrix | csc_matrix | csr_array | csc_array | ndarray
"""Matrix types that are supported in NicheNetPy."""

type gene_t = str | int
"""Gene symbol types that are supported in NicheNetPy."""

GeneSymbol = TypeVar("GeneSymbol", str, int)
"""Gene symbol types that are supported in NicheNetPy."""

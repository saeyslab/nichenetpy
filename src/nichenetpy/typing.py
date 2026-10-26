from typing import Any, TypeVar

from numpy import dtype, float64, ndarray
from scipy.sparse import csc_matrix, csr_matrix, csr_array, csc_array

type nichenet_matrix = csr_matrix[float64] | csc_matrix[float64] | csr_array[float64] | csc_array[float64] | ndarray[Any, dtype[float64]]
"""Matrix types that are supported in NicheNetPy."""

type gene_t = str | int
"""Gene symbol types that are supported in NicheNetPy."""

GeneSymbol = TypeVar("GeneSymbol", str, int)
"""Gene symbol types that are supported in NicheNetPy."""

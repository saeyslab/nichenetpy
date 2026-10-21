from __future__ import annotations  # Series[T] typehint can throw an Error at runtime

from collections.abc import Iterable, Sequence
from math import erfc, sqrt
from numbers import Number
from typing import Literal, overload

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import t

from nichenetpy.typing import GeneSymbol, nichenet_matrix
from nichenetpy.utils import subset_matrix


def _rank_cells(mat: nichenet_matrix, cell_groups: Sequence[str] | pd.Series[str], tie_correction: bool = True):
    if type(cell_groups) is pd.Series:
        # indexing series is slow and deprecated (warning is thrown)
        cell_groups = cell_groups.tolist()
    if type(mat) is not csc_matrix:
        try:
            mat = csc_matrix(mat)
        except TypeError as error:
            raise TypeError(f"mat should be a 2-D np.ndarray or a sparse matrix, was {type(mat)}") from error
    if type(mat.shape) is None:
        raise ValueError("mat has shape None")
    output = []
    group_mat = []
    nrows, ncols = mat.shape
    tie_stat = np.zeros(shape=(ncols,))
    # for each gene
    for ci in range(ncols):
        ranks = []
        # the row indices of non-zero elements
        indices_non_zero = mat.indices[mat.indptr[ci] : mat.indptr[ci + 1]]
        values_non_zero = mat.data[mat.indptr[ci] : mat.indptr[ci + 1]]
        # reorder the groups of the cells so it matches the ranking
        if len(values_non_zero) > 0:
            groups_sorted, non_zero = zip(
                *sorted(
                    zip((cell_groups[i] for i in indices_non_zero), values_non_zero, strict=True), key=lambda x: x[1]
                ),
                strict=True,
            )
        else:
            non_zero = []
            groups_sorted = []
        n_zero = nrows - (mat.indptr[ci + 1] - mat.indptr[ci])
        if tie_correction:
            tie_stat[ci] += (float(n_zero) ** 2 - 1) * float(n_zero)
        n_neg = 0
        while n_neg < len(non_zero) and non_zero[n_neg] < 0:
            n_neg += 1
        indices_non_zero = set(indices_non_zero)
        group_mat.append(groups_sorted)
        # original rank for a value of 0 (ranks will be translated to get a 0-rank for 0-values)
        # compute average using gaussian summation (the +1 has been moved into the division)
        zero_rank = 0 if n_zero == 0 else n_neg + (n_zero + 1) / 2
        # compute the rank of each cell
        # negative
        i = 0
        while i < n_neg:
            n_tied = 1
            while i + n_tied < n_neg and non_zero[i] == non_zero[i + n_tied]:
                n_tied += 1
            if tie_correction and n_tied > 1:
                tie_stat[ci] += (n_tied**2 - 1) * n_tied
            # compute average using gaussian summation
            rank = i + 1 + (n_tied - 1) / 2 - zero_rank
            for _ in range(n_tied):
                ranks.append(rank)
            i += n_tied
        # positive
        i = n_neg
        while i < len(non_zero):
            n_tied = 1
            while i + n_tied < len(non_zero) and non_zero[i] == non_zero[i + n_tied]:
                n_tied += 1
            if tie_correction and n_tied > 1:
                tie_stat[ci] += (n_tied**2 - 1) * n_tied
            # compute average using gaussian summation
            rank = n_zero + i + 1 + (n_tied - 1) / 2 - zero_rank
            for _ in range(n_tied):
                ranks.append(rank)
            i += n_tied
        output.append(ranks)
    return (output, group_mat, tie_stat)


@overload
def wilcoxon_rank_sum_test(
    ann: AnnData,
    groupby: str,
    as_dataframe: Literal[True],
    tie_correction: bool = True,
    layer: str | None = "data",
    genes: list[GeneSymbol] | tuple[GeneSymbol] | None = None,
) -> pd.DataFrame: ...


@overload
def wilcoxon_rank_sum_test(
    ann: AnnData,
    groupby: str,
    as_dataframe: Literal[False],
    tie_correction: bool = True,
    layer: str | None = "data",
    genes: list[GeneSymbol] | tuple[GeneSymbol] | None = None,
) -> dict[str, GeneSymbol]: ...


def wilcoxon_rank_sum_test(
    ann: AnnData,
    groupby: str,
    as_dataframe: bool = False,
    tie_correction: bool = True,
    layer: str | None = "data",
    genes: Sequence[GeneSymbol] | None = None,
) -> pd.DataFrame | dict[str, GeneSymbol]:
    """
    Perform the wilcoxon rank sum test and return the p-values

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    groupby : str
        the column to group by
    as_dataframe : bool
        if True, a pandas DataFrame is returned
    tie_correction : bool
        if True, tie correction is performed
    layer : str or None
        the layer of the AnnData object to use
    genes : list of gene_t or tuple of gene_t or None
        if provided, only consider these genes

    Returns
    -------
    pandas.DataFrame or dict
        the p-value for each gene

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid

    Notes
    -----
    implementation based on https://github.com/bnprks/BPCells

    please make sure the optional "genes" argument does not contain duplicates
    """
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type AnnData, was {type(ann)}")
    if type(groupby) is not str:
        raise TypeError(f"groupby should have type str, was {type(groupby)}")
    if type(as_dataframe) is not bool:
        raise TypeError(f"as_dataframe should have type bool, was {type(as_dataframe)}")
    if type(tie_correction) is not bool:
        raise TypeError(f"tie_correction should have type bool, was {type(tie_correction)}")
    if layer is not None:
        if type(layer) is not str:
            raise TypeError(f"layer should have type str, was {type(layer)}")
        if layer not in ann.layers:
            raise ValueError(f"There is no layer '{layer}' in the AnnData object")
    if groupby not in ann.obs.columns:
        raise ValueError(f"There is no column '{groupby}' in the AnnData object")
    group_sizes = dict(ann.obs[groupby].value_counts())
    n_total = len(ann.obs)
    pvals = {}
    if layer is None:
        if (
            not isinstance(ann.X, np.ndarray)
            and not isinstance(ann.X, csr_matrix)
            and not isinstance(ann.X, csc_matrix)
        ):
            raise TypeError(f"ann.X should be of type nichenet_matrix, was {type(ann.X)}")
        mat = ann.X
    else:
        mat = ann.layers[layer]
        if not isinstance(mat, np.ndarray) and not isinstance(mat, csr_matrix) and not isinstance(mat, csc_matrix):
            raise ValueError(f"the selected layer should be of type nichenet_matrix, was {type(ann.X)}")
    if genes is None:
        genes = ann.var_names  # type: ignore # this works fine
    else:
        if type(genes) is not list and type(genes) is not tuple:
            raise TypeError(f"genes should be of type list or tuple, was {type(genes)}")
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names)), strict=True))
        mat = subset_matrix(mat, cols=[gene2index[gene] for gene in genes])
    ranks, sorted_groups, tie_stats = _rank_cells(mat, ann.obs[groupby], tie_correction=tie_correction)
    rank_sums = {}
    # for each gene
    for ranking, groups, tie_stat in zip(ranks, sorted_groups, tie_stats, strict=True):
        for group in group_sizes.keys():
            rank_sums[group] = 0.0
        for rank, group in zip(ranking, groups, strict=True):
            rank_sums[group] += rank
        total_rank = sum(rank_sums.values())
        # for each group
        for group in rank_sums.keys():
            # test statistic
            n_group = group_sizes[group]
            n_other = n_total - n_group
            rank_offset = (n_total + 1) / 2 - (total_rank / n_total)
            u_group = rank_sums[group] + n_group * (rank_offset - (n_group + 1) / 2)
            u_other = total_rank - rank_sums[group] + n_other * (rank_offset - (n_other + 1) / 2)
            u = max(u_group, u_other)
            u_mean = n_group * n_other / 2
            u_std = sqrt(u_mean / 6 * (n_total + 1 - tie_stat / (n_total * (n_total - 1))))
            continuity_correction = 0.5 if u > u_mean else 0
            if u_std == 0:
                pval = 1
            else:
                z_score = (u - continuity_correction - u_mean) / u_std
                pval = erfc(z_score / sqrt(2))
            if group in pvals:
                pvals[group].append(pval)
            else:
                pvals[group] = [pval]
    if as_dataframe:
        pvals = pd.DataFrame(pvals, index=genes)  # type: ignore # this behaves as expected
        pvals.index.name = "gene"  # type: ignore
    return pvals  # type: ignore


def wilcoxon_rank_sum_test_with_correlation(
    index: Iterable[int] | Iterable[bool], statistics: Sequence[float], correlation: float = 0, df: float = np.inf
):
    """
    Rank sum test as for two-sample Wilcoxon-Mann-Whitney test, but allowing for correlation between members of test set.

    Parameters
    ----------
    index : Iterable of int or Iterable of bool
        indices or mask of the test group
    statistics : Iterable[float]
        the values of the test statistic
    correlation : float
        The average correlation between cases in the test group.
        Cases in the second group are assumed independent of each other and other the first group.
    df : float
        the degrees of freedom which the correlation has been estimated

    Returns
    -------
    float
        lower tail p-value
    float
        upper tail p-value

    Raises
    ------
    TypeError
        if the arguments have the wrong type

    Notes
    -----
    implementation based on https://github.com/gangwug/limma/blob/master/R/rankSumTestWithCorrelation.R
    """
    if not isinstance(index, Iterable):
        raise TypeError(f"index should have type Iterable[int] or Iterable[bool], was {type(index)}")
    if not isinstance(statistics, Iterable):
        raise TypeError(f"statistics should have type Iterable[float], was {type(statistics)}")
    if not isinstance(correlation, Number):
        raise TypeError(f"correlation should have type float, was {type(correlation)}")
    if not isinstance(df, Number):
        raise TypeError(f"df should have type float, was {type(df)}")
    n = len(statistics)
    r = pd.DataFrame(statistics, columns=("rank",)).rank(method="average", ascending=True)
    if type(index) is not list and type(index) is not tuple:
        index = list(index)
    r1 = r[index] if type(index[0]) is bool or type(index[0]) is np.bool else r.iloc[index]
    n1 = len(r1)
    n2 = n - n1
    u = n1 * n2 + n1 * (n1 + 1) / 2 - sum(r1["rank"])
    mu = n1 * n2 / 2
    if correlation == 0 or n1 == 1:
        sigma2 = n1 * n2 * (n + 1) / 12
    else:
        sigma2 = (
            1.57079633 * n1 * n2
            + 0.523598776 * n1 * n2 * (n2 - 1)
            + np.arcsin(correlation / 2) * n1 * (n1 - 1) * n2 * (n2 - 1)
            + np.arcsin((correlation + 1) / 2) * n1 * (n1 - 1) * n2
        ) / (2 * np.pi)
    ties = r.groupby("rank").size()
    adjustment = sum(ties * (ties + 1) * (ties - 1)) / (n * (n + 1) * (n - 1))
    if adjustment < 1:
        sigma2 *= 1 - adjustment
    zlowertail = (u + 0.5 - mu) / sqrt(sigma2)
    zuppertail = (u - 0.5 - mu) / sqrt(sigma2)
    return (t.cdf(zlowertail, df), 1 - t.cdf(zuppertail, df))

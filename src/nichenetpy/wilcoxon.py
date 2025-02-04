from nichenetpy.utils import subset_matrix

from scipy.sparse import csc_matrix
from anndata import AnnData
from itertools import chain
from collections.abc import Iterable
from math import sqrt, erfc

import pandas as pd
import numpy as np


def _rank_cells(
    mat:csc_matrix,
    cell_groups:Iterable[str],
    tie_correction:bool=True
):
    if type(cell_groups) is pd.Series:
        # indexing series is slow and deprecated (warning is thrown)
        cell_groups = list(cell_groups)
    if type(mat) is not csc_matrix:
        raise TypeError(f"mat should have type scipy.csc_matrix, was {type(mat)}")
    output = []
    group_mat = []
    nrows, ncols = mat.shape
    tie_stat = np.zeros(shape=(ncols,))
    for ci in range(ncols):
        ranks = []
        indices_non_zero = mat.indices[mat.indptr[ci]:mat.indptr[ci+1]]
        values_non_zero = mat.data[mat.indptr[ci]:mat.indptr[ci+1]]
        if len(values_non_zero) > 0:
            groups_sorted, non_zero = zip(
                *sorted(
                    zip(
                        (cell_groups[i] for i in indices_non_zero),
                        values_non_zero
                    ),
                    key=lambda x : x[1]
                )
            )
        else:
            non_zero = []
            groups_sorted = []
        n_zero = nrows - (mat.indptr[ci+1] - mat.indptr[ci])
        if tie_correction:
            tie_stat[ci] += (float(n_zero)**2 - 1)*float(n_zero)
        n_neg = 0
        while n_neg < len(non_zero) and non_zero[n_neg] < 0:
            n_neg += 1
        indices_non_zero = set(indices_non_zero)
        # reorder the groups of the cells so it matches the ranking
        group_mat.append(
            list(
                chain(
                    groups_sorted[:n_neg],
                    (cell_groups[i] for i in range(nrows) if i not in indices_non_zero),# bottleneck
                    groups_sorted[n_neg:]
                )
            )
        )
        # original rank for a value of 0 (ranks will be translated to get a 0-rank for 0-values)
        # compute average using gaussian summation (the +1 has been moved into the division)
        zero_rank = 0 if n_zero == 0 else n_neg + (n_zero + 1)/2
        # negative 
        i = 0
        while i < n_neg:
            n_tied = 1
            while i + n_tied < n_neg and non_zero[i] == non_zero[i + n_tied]:
                n_tied += 1
            if tie_correction and n_tied > 1:
                tie_stat[ci] += (n_tied**2 - 1)*n_tied
            # compute average using gaussian summation
            rank = i + 1 + (n_tied - 1)/2 - zero_rank
            for _ in range(n_tied):
                ranks.append(rank)
            i += n_tied
        # zero
        for _ in range(n_zero):
            ranks.append(0)
        # positive
        i = n_neg
        while i < len(non_zero):
            n_tied = 1
            while i + n_tied < len(non_zero) and non_zero[i] == non_zero[i + n_tied]:
                n_tied += 1
            if tie_correction and n_tied > 1:
                tie_stat[ci] += (n_tied**2 - 1)*n_tied
            # compute average using gaussian summation
            rank = n_zero + i + 1 + (n_tied - 1)/2 - zero_rank
            for _ in range(n_tied):
                ranks.append(rank)
            i += n_tied
        output.append(ranks)
    return (output, group_mat, tie_stat)

def wilcoxon_rank_sum_test(
    ann:AnnData,
    groupby:str,
    as_dataframe:bool=False,
    tie_correction:bool=True,
    layer:str="data",
    genes:list[str]=None
):
    '''
    perform the wilcoxon rank sum test and return the p-values

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
    layer : str
        the layer of the AnnData object to use
    genes : list of str
        if provided, only consider these genes
    
    Returns
    -------
    pandas.DataFrame or dict
        the p-value for each gene
    
    Notes
    -----
    implementation based on https://github.com/bnprks/BPCells
    '''
    group_sizes = dict(ann.obs[groupby].value_counts())
    n_total = len(ann.obs)
    pvals = dict()
    mat = ann.layers[layer]
    if genes is None:
        genes = ann.var_names
    else:
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names))))
        mat = subset_matrix(mat, cols=[gene2index[gene] for gene in genes])
    ranks, sorted_groups, tie_stats = _rank_cells(
        mat,
        ann.obs[groupby],
        tie_correction=tie_correction
    )
    rank_sums = dict()
    for ranking, groups, tie_stat in zip(ranks, sorted_groups, tie_stats):
        rank_sums.clear()
        for rank, group in zip(ranking, groups):
            if group in rank_sums:
                rank_sums[group] += rank
            else:
                rank_sums[group] = float(rank)
        total_rank = sum(rank_sums.values())
        for group in rank_sums.keys():
            # test statistic
            n_group = group_sizes[group]
            n_other = n_total - n_group
            rank_offset = (n_total + 1) / 2 - (total_rank / n_total)
            u_group = rank_sums[group] + n_group * (rank_offset - (n_group + 1)/2)
            u_other = total_rank - rank_sums[group] + n_other * (rank_offset - (n_other + 1)/2)
            u = max(u_group, u_other)
            u_mean = n_group * n_other / 2
            u_std = sqrt(u_mean / 6 * (n_total + 1 - tie_stat / (n_total*(n_total - 1))))
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
        pvals = pd.DataFrame(pvals, index=genes)
    return pvals
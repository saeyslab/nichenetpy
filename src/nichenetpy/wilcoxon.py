from scipy.sparse import csc_matrix

from itertools import chain


def _rank_cells(
    mat:csc_matrix,
    cells:list[str]
):
    if type(mat) is not csc_matrix:
        raise TypeError(f"mat should have type scipy.csc_matrix, was {type(mat)}")
    output = []
    mat_cells = []
    nrows, ncols = mat.shape
    for ci in range(ncols):
        ranks = []
        indices_non_zero = mat.indices[mat.indptr[ci]:mat.indptr[ci+1]]
        values_non_zero = mat.data[mat.indptr[ci]:mat.indptr[ci+1]]
        cells_sorted, non_zero = zip(
            *sorted(
                zip(
                    (cells[i] for i in indices_non_zero),
                    values_non_zero
                ),
                key=lambda x : x[1]
            )
        )
        n_zero = nrows - (mat.indptr[ci+1] - mat.indptr[ci])
        n_neg = 0
        while n_neg < len(non_zero) and non_zero[n_neg] < 0:
            n_neg += 1
        mat_cells.append(
            list(
                chain(
                    cells_sorted[:n_neg],
                    (cells[i] for i in range(nrows) if i not in indices_non_zero),
                    cells_sorted[n_neg:]
                )
            )
        )
        # compute average using gaussian summation
        zero_rank = n_neg + (n_zero - 1)/2
        i = 0
        while i < n_neg:
            n_tied = 1
            while i + n_tied < n_neg and non_zero[i] == non_zero[i + n_tied]:
                n_tied += 1
            # compute average using gaussian summation
            rank = i + 1 + (n_tied - 1)/2 - zero_rank
            for _ in range(n_tied):
                ranks.append(rank)
            i += n_tied
        for _ in range(n_zero):
            ranks.append(0)
        i = n_neg
        while i < len(non_zero):
            n_tied = 1
            while i + n_tied < len(non_zero) and non_zero[i] == non_zero[i + n_tied]:
                n_tied += 1
            # compute average using gaussian summation
            rank = n_zero + i + 1 + (n_tied - 1)/2 - zero_rank
            for _ in range(n_tied):
                ranks.append(rank)
            i += n_tied
        output.append(ranks)
    return (mat_cells, output)
from scipy.sparse import csc_matrix


def _rank_cells(
    mat:csc_matrix
):
    if type(mat) is not csc_matrix:
        raise TypeError(f"mat should have type scipy.csc_matrix, was {type(mat)}")
    output = []
    nrows, ncols = mat.shape
    for ci in range(ncols):
        ranks = []
        col = mat[:, ci]
        n_neg = 0
        n_zero = nrows
        for e in col.data:
            if e < 0:
                n_neg += 1
            n_zero -= 1
        # compute average using gaussian summation
        zero_rank = n_neg + (n_zero - 1)/2
        non_zero = sorted(col.data)
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
    return output
import heapq

from scipy.sparse import csr_matrix
from collections.abc import Iterable

import numpy as np


def dijkstra_spl(
    graph:csr_matrix,
    src:int
):
    if type(graph) is not csr_matrix:
        raise TypeError(f"graph should have type scipy.csr_matrix, was {type(csr_matrix)}")
    if type(src) is not int:
        raise TypeError(f"src should have type int, was {type(src)}")
    l = graph.shape[0]
    spl = [np.inf for _ in range(l)]
    done = set()
    pq = [(0, src)]
    spl[src] = 0
    while len(pq) > 0:
        cur = heapq.heappop(pq)[1]
        if cur not in done:
            for nb, val in zip(
                graph.indices[graph.indptr[cur]:graph.indptr[cur+1]],
                graph.data[graph.indptr[cur]:graph.indptr[cur+1]]
            ):
                pl = spl[cur] + val
                if pl < spl[nb]:
                    spl[nb] = pl
                    heapq.heappush(pq, (pl, nb))
            done.add(cur)
    return spl
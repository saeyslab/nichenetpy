import heapq

from scipy.sparse import csr_matrix
from collections.abc import Iterator

import numpy as np


def dijkstra_spl(
    graph:csr_matrix,
    src:int
) -> list[float]:
    '''
    Computes the shortest path length from the source vertex to every vertex in the graph using dijkstra's algorithm. 

    Parameters
    ----------
    graph : scipy.csr_matrix
        the graph
    src : int
        the source vertex
    
    Returns
    -------
    list of float
        the shortest path lengths from the source vertex to every vertex in the graph
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
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

def _walk_graph(
    graph:csr_matrix,
    src:int,
) -> Iterator[int]:
    yield src
    for nb, val in zip(
        graph.indices[graph.indptr[src]:graph.indptr[src+1]],
        graph.data[graph.indptr[src]:graph.indptr[src+1]]
    ):
        if val != 0:
            # more efficient than removing from graph.indices and graph.data
            graph[src, nb] = 0
            graph[nb, src] = 0
            yield from _walk_graph(graph, nb)

def walk_graph(
    graph:csr_matrix,
    src:int,
) -> Iterator[int]:
    yield from _walk_graph(graph.copy(), src)
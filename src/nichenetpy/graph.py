import numpy as np
from scipy.sparse import csr_array


def get_reachable_nodes(graph: csr_array[np.float64], src: int) -> set[int]:
    """
    Returns all nodes in the graph which are reachable from the specified source node.

    Parameters
    ----------
    graph : scipy.sparse.csr_array
        the graph (as adjacency matrix)
    src : int
        the source node

    Returns
    -------
    set of int
        the set of reachable nodes

    Raises
    ------
    TypeError
        if the arguments are not of the correct type
    """
    if type(graph) is not csr_array:
        raise TypeError(f"graph should have type scipy.sparse.csr_array, was {type(graph)}")
    if type(src) is not int:
        raise TypeError(f"src should have type int, was {type(src)}")
    output = {src}
    q = [src]
    while len(q) > 0:
        cur = q.pop()
        nxts = graph.indices[graph.indptr[cur] : graph.indptr[cur + 1]]
        for nxt in nxts:
            if (
                nxt not in output
            ):  # if nxt is in output it's neighbours have already been visited or nxt is already queued
                q.append(nxt)
                output.add(nxt)
    return output

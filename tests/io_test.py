# type: ignore

"""
Here we test the custom io functions: a ligand-target matrix and (weighted) networks are written to a file and read
back. The network files are 2-5 times smaller than pickle files (see the Notes of `io.write_network` and
`io.write_weighted_network`).
"""

import random
from os import remove
from pickle import dumps

import numpy as np
from common import equals_iter, random_string

from nichenetpy.io import (
    read_ligand_target_matrix,
    write_ligand_target_matrix,
    write_network,
    write_weighted_network,
)
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.prediction import LigandActivityPredictor

pickle_protocol = 4


def test_io(n=10):
    filename = "temp.bin"
    predictor_init = LigandActivityPredictor(
        np.array(np.random.rand(n, n), order="F"),
        [random_string() for _ in range(n)],
        [random_string() for _ in range(n)],
    )
    lr_network_init = LigandReceptorNetwork(
        mapping=sorted([(random_string(), random_string()) for _ in range(n)], key=lambda x: x[0])
    )
    lr_sig_init = WeightedNetwork(
        mapping=sorted([(random_string(), random_string(), random.random()) for _ in range(n)], key=lambda x: x[0])
    )
    write_ligand_target_matrix(filename, predictor_init, column_major=True)
    predictor_wrt = LigandActivityPredictor(*read_ligand_target_matrix(filename, column_major=True))
    assert dumps(predictor_init, protocol=pickle_protocol) == dumps(predictor_wrt, protocol=pickle_protocol)
    write_network(filename, lr_network_init._mapping)
    lr_network_wrt = LigandReceptorNetwork(filename=filename)
    assert equals_iter(lr_network_init._mapping, lr_network_wrt._mapping, err_bound=0, zero_bound=0)
    write_weighted_network(filename, lr_sig_init._mapping)
    lr_sig_wrt = WeightedNetwork(filename=filename)
    assert equals_iter(lr_sig_init._mapping, lr_sig_wrt._mapping, err_bound=0, zero_bound=0)
    remove(filename)

'''
Here we test the custom oi functions. These functions are not very useful though as they
turned out to have the same memory consumption as pickle files. 
'''

from nichenetpy.io import (
    write_ligand_target_matrix,
    read_ligand_target_matrix,
    write_network,
    write_weighted_network,
)
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork

from pickle import dumps
from os import remove

from common import (
    get_model_pickle,
    equals_iter
)


def test_io():
    filename = "temp.bin"
    model = get_model_pickle()
    predictor_init : LigandActivityPredictor = model["predictor"]
    lr_network_init : LigandReceptorNetwork = model["lr_network"]
    lr_sig_init : WeightedNetwork = model["lr_sig"]
    write_ligand_target_matrix(
        filename,
        predictor_init,
        column_major=True
    )
    predictor_wrt = LigandActivityPredictor(*read_ligand_target_matrix(filename, column_major=True))
    assert dumps(predictor_init) == dumps(predictor_wrt)
    write_network(filename, lr_network_init._mapping)
    lr_network_wrt = LigandReceptorNetwork(filename=filename)
    assert equals_iter(lr_network_init._mapping, lr_network_wrt._mapping, err_bound=0, zero_bound=0)
    write_weighted_network(filename, lr_sig_init._mapping)
    lr_sig_wrt = WeightedNetwork(filename=filename)
    assert equals_iter(lr_sig_init._mapping, lr_sig_wrt._mapping, err_bound=0, zero_bound=0)
    remove(filename)
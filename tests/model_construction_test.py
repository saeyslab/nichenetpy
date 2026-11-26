# type: ignore

"""
Here we test some functions from nichenetpy.model_construction (the model construction on the real networks is tested in
full_nichenet_test.py, personalized PageRank in ppr_nichenetr_test.py).
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from nichenetpy.model_construction import construct_ligand_tf_matrix, construct_weighted_networks


def test_construct_weighted_networks_empty_source_weights():
    # no source weights give empty networks, as weights of 0 do
    lr = pd.DataFrame({"from": ["L1"], "to": ["R1"], "source": ["lr"]})
    sig = pd.DataFrame({"from": ["R1"], "to": ["G1"], "source": ["sig"]})
    gr = pd.DataFrame({"from": ["G1"], "to": ["G2"], "source": ["gr"]})
    empty = construct_weighted_networks(lr, sig, gr, {})
    zero = construct_weighted_networks(lr, sig, gr, {"lr": 0, "sig": 0, "gr": 0})
    for key in ("lr_sig", "gr"):
        assert len(empty[key]) == len(zero[key]) == 0
        assert empty[key].columns.tolist() == zero[key].columns.tolist()


def test_dict_subclasses():
    # a dict subclass (e.g. an OrderedDict) is accepted as source weights and as weighted networks
    lr = pd.DataFrame({"from": ["L1"], "to": ["R1"], "source": ["lr"]})
    sig = pd.DataFrame({"from": ["R1"], "to": ["G1"], "source": ["sig"]})
    gr = pd.DataFrame({"from": ["G1"], "to": ["G2"], "source": ["gr"]})
    weights = {"lr": 1, "sig": 0.5, "gr": 1}
    networks = construct_weighted_networks(lr, sig, gr, weights)
    networks_od = construct_weighted_networks(lr, sig, gr, OrderedDict(weights))
    for key in ("lr_sig", "gr"):
        pd.testing.assert_frame_equal(networks_od[key], networks[key])
    expected = construct_ligand_tf_matrix(networks, [["L1"]], ltf_cutoff=0)
    result = construct_ligand_tf_matrix(OrderedDict(networks), [["L1"]], ltf_cutoff=0)
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]

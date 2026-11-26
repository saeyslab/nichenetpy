# type: ignore

"""
Compare the personalized PageRank ligand-tf matrix of construct_ligand_tf_matrix with nichenetr (igraph PRPACK).
The dummy network has genes without outgoing links (also a ligand), self-loops and ligand combinations.
The reference matrices are generated with data/ppr_nichenetr/generate_reference.R
With more ligands than fit in one batch, the scores are compared with the exact solution of the linear system.
"""

import os

import numpy as np
import pandas as pd
import pytest

from nichenetpy import model_construction, parameter_optimization
from nichenetpy.model_construction import (
    construct_ligand_target_matrix,
    construct_ligand_tf_matrix,
    construct_model_from_source_weights,
)
from nichenetpy.parameter_optimization import construct_and_evaluate

root = os.path.join(os.path.dirname(__file__), "data", "ppr_nichenetr")
weighted_networks = {
    "lr_sig": pd.read_csv(os.path.join(root, "lr_sig.csv")),
    "gr": pd.read_csv(os.path.join(root, "gr.csv")),
}
ligands = [["L1"], ["L2"], ["L3"], ["L4"], ["L1", "L2"], ["L1", "L4"]]


@pytest.mark.parametrize("damping_factor,ltf_cutoff", [(0, 0), (0.2, 0), (0.5, 0), (0.85, 0), (0.99, 0), (0.5, 0.5)])
def test_ppr_same_as_nichenetr(damping_factor, ltf_cutoff):
    reference = pd.read_csv(os.path.join(root, f"ltf_damping{damping_factor}_cutoff{ltf_cutoff}.csv"), index_col=0)
    mat, row_names, col_names = construct_ligand_tf_matrix(
        weighted_networks, ligands, ltf_cutoff=ltf_cutoff, algorithm="PPR", damping_factor=damping_factor
    )
    result = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert set(result.index) == set(reference.index)
    assert set(result.columns) == set(reference.columns)
    # both stop iterating when the scores change by less than 1e-10, so they agree up to about 1e-9
    np.testing.assert_allclose(
        result.loc[reference.index, reference.columns].to_numpy(), reference.to_numpy(), rtol=0, atol=1e-8
    )


def test_ppr_split_same_as_nichenetr():
    reference = pd.read_csv(os.path.join(root, "ltf_damping0.5_cutoff0.5.csv"), index_col=0)
    (reflexive, non_reflexive), row_names, col_names = construct_ligand_tf_matrix(
        weighted_networks, ligands, ltf_cutoff=0.5, algorithm="PPR", damping_factor=0.5, split_ppr=True
    )
    result = pd.DataFrame(reflexive + non_reflexive, index=row_names, columns=col_names)
    np.testing.assert_allclose(
        result.loc[reference.index, reference.columns].to_numpy(), reference.to_numpy(), rtol=0, atol=1e-8
    )
    # the reflexive scores are only at the ligand itself (the members of a combination)
    for row, _ligands in zip(reflexive, ligands, strict=True):
        assert {col_names[i] for i in np.flatnonzero(row)} == set(_ligands)


def test_damping_factor_one_refused():
    with pytest.raises(ValueError):
        construct_ligand_tf_matrix(weighted_networks, ligands, damping_factor=1)


def test_ppr_implementation_refused():
    with pytest.raises(ValueError):
        construct_ligand_tf_matrix(weighted_networks, ligands, ppr_implementation="igraph")


def test_ppr_scikit_network():
    # the implementation of nichenetpy 1.x, which differs from nichenetr for genes without outgoing links
    pytest.importorskip("sknetwork")
    sknetwork, _, _ = construct_ligand_tf_matrix(
        weighted_networks, ligands, ltf_cutoff=0, damping_factor=0.85, ppr_implementation="scikit-network"
    )
    nichenetr, _, _ = construct_ligand_tf_matrix(weighted_networks, ligands, ltf_cutoff=0, damping_factor=0.85)
    assert sknetwork.shape == nichenetr.shape
    np.testing.assert_allclose(sknetwork.sum(axis=1), 1)
    assert np.abs(sknetwork - nichenetr).max() > 1e-3


# the scikit-network implementation (nichenetpy 1.x) differs from nichenetr, more so for a higher damping_factor;
# the tolerances are just above the differences on this network (0, 4.2e-4, 0.013 and 0.089)
@pytest.mark.parametrize(
    "damping_factor,ltf_cutoff,tolerance",
    [(0, 0, 1e-12), (0.2, 0, 1e-3), (0.5, 0, 0.02), (0.85, 0, 0.1), (0.5, 0.5, 0.02)],
)
def test_ppr_scikit_network_close_to_nichenetr(damping_factor, ltf_cutoff, tolerance):
    pytest.importorskip("sknetwork")
    reference = pd.read_csv(os.path.join(root, f"ltf_damping{damping_factor}_cutoff{ltf_cutoff}.csv"), index_col=0)
    mat, row_names, col_names = construct_ligand_tf_matrix(
        weighted_networks,
        ligands,
        ltf_cutoff=ltf_cutoff,
        algorithm="PPR",
        damping_factor=damping_factor,
        ppr_implementation="scikit-network",
    )
    result = pd.DataFrame(mat, index=row_names, columns=col_names)
    np.testing.assert_allclose(
        result.loc[reference.index, reference.columns].to_numpy(), reference.to_numpy(), rtol=0, atol=tolerance
    )


def _random_network(n_genes, n_links, seed):
    # a random weighted network with self-loops and genes without outgoing links (every sixth gene), one link per pair
    rng = np.random.default_rng(seed)
    genes = [f"G{i:03d}" for i in range(n_genes)]
    pairs = np.unique(rng.integers(0, n_genes, size=(n_links, 2)), axis=0)
    pairs = pairs[pairs[:, 0] % 6 != 0]
    weights = rng.random(len(pairs)) + 0.1
    lr_sig = pd.DataFrame(
        {"from": [genes[i] for i in pairs[:, 0]], "to": [genes[i] for i in pairs[:, 1]], "weight": weights}
    )
    gr = pd.DataFrame({"from": [genes[1]], "to": [genes[2]], "weight": [1.0]})
    return genes, pairs, weights, {"lr_sig": lr_sig, "gr": gr}


def _exact_ppr(n_genes, pairs, weights, damping_factor):
    # the scores of seed s solve (M - d e_s u^T) x = (1 - d) e_s, with M = I - d P^T, e_s the seed and u the genes
    # without outgoing links; with y = M^-1 e_s (Sherman-Morrison): x = (1 - d) y / (1 - d u^T y), so a single
    # inverse gives the scores of every seed (one row per seed)
    adjacency = np.zeros((n_genes, n_genes))
    adjacency[pairs[:, 0], pairs[:, 1]] = weights
    out_weight = adjacency.sum(axis=1)
    dangling = out_weight == 0
    transition = np.divide(adjacency, out_weight[:, None], out=np.zeros_like(adjacency), where=~dangling[:, None])
    inverse = np.linalg.inv(np.eye(n_genes) - damping_factor * transition.T)
    return ((1 - damping_factor) * inverse / (1 - damping_factor * inverse[dangling].sum(axis=0))).T


@pytest.mark.parametrize("damping_factor", [0.5, 0.85, 0.99])
def test_ppr_many_ligands_exact(damping_factor):
    # 300 ligands, more than one batch of 256
    genes, pairs, weights, networks = _random_network(400, 4000, seed=0)
    ligands = [[gene] for gene in genes[:300]]
    mat, row_names, col_names = construct_ligand_tf_matrix(
        networks, ligands, ltf_cutoff=0, algorithm="PPR", damping_factor=damping_factor
    )
    assert row_names == genes[:300]
    assert col_names == genes
    exact = _exact_ppr(len(genes), pairs, weights, damping_factor)[:300]
    np.testing.assert_allclose(mat, exact, rtol=0, atol=1e-8)


def test_ppr_independent_of_other_ligands():
    # a ligand's scores are exactly the same whichever other ligands are requested (also across batches)
    genes, _, _, networks = _random_network(400, 4000, seed=1)
    ligands = [[gene] for gene in genes[:300]]
    full, row_names, _ = construct_ligand_tf_matrix(networks, ligands, ltf_cutoff=0, damping_factor=0.85)
    for subset in (ligands[::7], ligands[260:], [ligands[5]], [ligands[290], ligands[3]]):
        part, part_rows, _ = construct_ligand_tf_matrix(networks, subset, ltf_cutoff=0, damping_factor=0.85)
        assert np.array_equal(part, full[[row_names.index(row) for row in part_rows]])


@pytest.fixture
def sknetwork_calls(monkeypatch):
    # counts the calls of the scikit-network implementation, which is replaced by the default one (so scikit-network
    # isn't needed)
    calls = []

    def spy(adjacency, seeds, damping):
        calls.append(len(seeds))
        return model_construction._personalized_pagerank(adjacency, seeds, damping)

    monkeypatch.setattr(model_construction, "_personalized_pagerank_sknetwork", spy)
    return calls


@pytest.mark.parametrize("split_direct", ["no", "ltf", "tft", "ltf-tft"])
def test_ppr_implementation_passed_on_target_matrix(sknetwork_calls, split_direct):
    lr_network = pd.DataFrame({"from": ["L1", "L2", "L3", "L4"], "to": ["R1", "R2", "R3", "R1"]})
    construct_ligand_target_matrix(
        weighted_networks, lr_network, ligands, split_direct=split_direct, ppr_implementation="scikit-network"
    )
    assert len(sknetwork_calls) > 0


def test_ppr_implementation_passed_on_source_weights(sknetwork_calls):
    lr_network = pd.DataFrame({"from": ["L1", "L2", "L3", "L4"], "to": ["R1", "R2", "R3", "R1"], "source": "lr"})
    sig_network = weighted_networks["lr_sig"][["from", "to"]].assign(source="sig")
    gr_network = weighted_networks["gr"][["from", "to"]].assign(source="gr")
    construct_model_from_source_weights(
        {"lr": 1, "sig": 1, "gr": 1},
        0,
        0,
        0.5,
        0.5,
        lr_network,
        gr_network,
        sig_network,
        ppr_implementation="scikit-network",
    )
    assert len(sknetwork_calls) > 0


def test_ppr_implementation_passed_on_construct_and_evaluate(monkeypatch):
    # construct_and_evaluate passes it to construct_model_from_source_weights (the evaluation itself isn't needed)
    class Stop(Exception):
        pass

    def spy(*args, **kwargs):
        raise Stop(kwargs["ppr_implementation"])

    class Evaluation:
        def get_ligands(self):
            return ["L1"]

    monkeypatch.setattr(parameter_optimization, "construct_model_from_source_weights", spy)
    lr_network = pd.DataFrame({"from": ["L1"], "to": ["R1"], "source": "lr"})
    sig_network = pd.DataFrame({"from": ["R1"], "to": ["G1"], "source": "sig"})
    gr_network = pd.DataFrame({"from": ["G1"], "to": ["G2"], "source": "gr"})
    with pytest.raises(Stop, match="scikit-network"):
        construct_and_evaluate(
            {"lr": 1, "sig": 1, "gr": 1},
            0,
            0,
            0.5,
            0.5,
            lr_network,
            gr_network,
            sig_network,
            Evaluation(),
            ppr_implementation="scikit-network",
        )

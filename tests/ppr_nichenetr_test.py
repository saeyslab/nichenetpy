# type: ignore

"""
Compare the personalized PageRank ligand-tf matrix of construct_ligand_tf_matrix with nichenetr (igraph PRPACK).
The dummy network has genes without outgoing links (also a ligand), self-loops and ligand combinations.
The reference matrices are generated with data/ppr_nichenetr/generate_reference.R
"""

import os

import numpy as np
import pandas as pd
import pytest

from nichenetpy.model_construction import construct_ligand_tf_matrix

root = os.path.join(os.path.dirname(__file__), "data", "ppr_nichenetr")
weighted_networks = {
    "lr_sig": pd.read_csv(os.path.join(root, "lr_sig.csv")),
    "gr": pd.read_csv(os.path.join(root, "gr.csv")),
}
ligands = [["L1"], ["L2"], ["L3"], ["L4"], ["L1", "L2"], ["L1", "L4"]]


@pytest.mark.parametrize(
    "damping_factor,ltf_cutoff", [(0, 0), (0.2, 0), (0.5, 0), (0.85, 0), (0.99, 0), (0.5, 0.5)]
)
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

# type: ignore

"""
Check that wrappers.get_geneset_oi compares the condition of interest with the reference condition (as NicheNetR's
FindMarkers(ident.1 = condition_oi, ident.2 = condition_reference)) when condition_ref is given, and with all other
conditions otherwise. It uses the data of group_metrics_seurat_test.py (groups A, B and C as conditions); the expected
genes are the genes of Seurat's FindMarkers / FindAllMarkers with an adjusted p-value of at most 0.05.
The gene set of interest of wrappers.run_nichenet is compared with the one of nichenetr's nichenet_seuratobj_aggregate,
generated with data/geneset_oi_nichenetr/generate_reference.R.
"""

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from nichenetpy.network import LigandReceptorNetwork
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.wrappers import get_geneset_oi, run_nichenet

root = os.path.join(os.path.dirname(__file__), "data", "group_metrics_seurat")
data = pd.read_csv(os.path.join(root, "data.csv"), index_col=0)  # genes x cells
meta = pd.read_csv(os.path.join(root, "meta.csv"), index_col=0)
reference = pd.read_csv(os.path.join(root, "seurat_markers.csv"))
settings = [(0.1, 0), (0.1, 0.5), (0.3, 0)]
# nichenetr's gene sets of interest: (condition_oi, condition_reference, expression_pct, lfc_cutoff) -> genes, the
# output of data/geneset_oi_nichenetr/generate_reference.R
NICHENETR_GENESET_OI = {
    ("A", "B", 0.1, 0.25): {"gene0", "gene1"},
    ("A", "B", 0.3, 0.25): {"gene0", "gene1"},
    ("A", "B", 0.1, 0.5): {"gene0", "gene1"},
    ("A", "C", 0.1, 0.25): {"gene0", "gene1", "gene2"},
    ("A", "C", 0.3, 0.25): {"gene0", "gene1", "gene2"},
    ("A", "C", 0.1, 0.5): {"gene0", "gene1", "gene2"},
    ("B", "C", 0.1, 0.25): {"gene2"},
    ("B", "C", 0.3, 0.25): {"gene2"},
    ("B", "C", 0.1, 0.5): {"gene2"},
    ("C", "A", 0.1, 0.25): {"gene0", "gene1", "gene2"},
    ("C", "A", 0.3, 0.25): {"gene0", "gene1", "gene2"},
    ("C", "A", 0.1, 0.5): {"gene0", "gene1", "gene2"},
}


def _ann(conditions=None):
    # all cells are of the receiver cell type "R", the conditions are the groups of the Seurat data
    obs = meta.assign(celltype="R")
    mat = data.to_numpy().T
    if conditions is not None:
        mask = obs["group"].isin(conditions).to_numpy()
        obs, mat = obs[mask], mat[mask]
    ann = AnnData(X=csr_matrix(mat), obs=obs, var=pd.DataFrame(index=data.index))
    ann.layers["data"] = csr_matrix(mat)
    return ann


def _seurat_genes(mode, min_pct, min_abs_lfc):
    # Seurat's adjusted p-value is the Bonferroni correction over all genes
    ref = reference[
        (reference["mode"] == mode)
        & (reference["min_pct"] == min_pct)
        & (reference["lfc_thr"] == min_abs_lfc)
        & (reference["group"] == "A")
    ]
    return set(ref[np.minimum(ref["pval"] * data.shape[0], 1) <= 0.05]["gene"])


@pytest.mark.parametrize(("min_pct", "min_abs_lfc"), settings)
def test_geneset_oi_with_reference_same_as_seurat(min_pct, min_abs_lfc):
    geneset = get_geneset_oi(_ann(), "R", "group", "A", "B", min_pct=min_pct, min_abs_lfc=min_abs_lfc)
    assert geneset == _seurat_genes("A_vs_B", min_pct, min_abs_lfc)


@pytest.mark.parametrize(("min_pct", "min_abs_lfc"), settings)
def test_geneset_oi_without_reference_same_as_seurat(min_pct, min_abs_lfc):
    geneset = get_geneset_oi(_ann(), "R", "group", "A", None, min_pct=min_pct, min_abs_lfc=min_abs_lfc)
    assert geneset == _seurat_genes("one_vs_rest", min_pct, min_abs_lfc)


@pytest.mark.parametrize("use_scanpy", [False, True])
def test_geneset_oi_reference_ignores_other_conditions(use_scanpy):
    # the cells of condition C don't change the comparison of A with B
    with_c = get_geneset_oi(_ann(), "R", "group", "A", "B", use_scanpy=use_scanpy)
    without_c = get_geneset_oi(_ann(["A", "B"]), "R", "group", "A", None, use_scanpy=use_scanpy)
    assert with_c == without_c


@pytest.mark.parametrize(("condition_oi", "condition_ref"), [("A", "D"), ("D", "B"), ("A", "data"), ("A", "A")])
def test_geneset_oi_invalid_conditions(condition_oi, condition_ref):
    # conditions that the receiver cells don't have (e.g. the layer passed where condition_ref is expected), and
    # the condition of interest as its own reference
    with pytest.raises(ValueError):
        get_geneset_oi(_ann(), "R", "group", condition_oi, condition_ref)


def _model():
    # the dummy model of data/geneset_oi_nichenetr/generate_reference.R
    genes = data.index.tolist()
    ligands = [f"gene{i}" for i in range(30, 35)]
    receptors = ["gene8", "gene28", "gene38", "gene5", "gene20"]
    i, j = np.meshgrid(np.arange(len(genes)), np.arange(len(ligands)), indexing="ij")
    ligand_target_matrix = np.asfortranarray(((i * 7 + j * 3) % 11 + 1) / 12)
    predictor = LigandActivityPredictor(ligand_target_matrix, genes, ligands)
    return predictor, LigandReceptorNetwork(list(zip(ligands, receptors, strict=True)))


@pytest.mark.parametrize(
    ("condition_oi", "condition_ref", "expression_pct", "min_abs_lfc"), sorted(NICHENETR_GENESET_OI)
)
def test_run_nichenet_geneset_oi_same_as_nichenetr(condition_oi, condition_ref, expression_pct, min_abs_lfc):
    predictor, lr_network = _model()
    res = run_nichenet(
        _ann(),
        predictor,
        lr_network,
        "R",
        "group",
        condition_oi,
        condition_ref,
        expression_pct=expression_pct,
        min_abs_lfc=min_abs_lfc,
    )
    assert res["geneset_oi"] == NICHENETR_GENESET_OI[(condition_oi, condition_ref, expression_pct, min_abs_lfc)]

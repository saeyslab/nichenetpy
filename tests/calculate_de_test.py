# type: ignore

"""
Check the scanpy path of prioritization.calculate_de (use_scanpy=True): it applies features, min_pct, min_abs_lfc and
pval_thresh as the default path (metrics.group_metrics) does, and the pct_ref column of
utils.rank_genes_groups_to_dataframe that the min_pct filter uses. It uses the data of group_metrics_seurat_test.py, with
the groups A, B and C as cell types of one condition.
"""

import os

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from nichenetpy.metrics import group_metrics
from nichenetpy.prioritization import calculate_de
from nichenetpy.utils import rank_genes_groups_to_dataframe

root = os.path.join(os.path.dirname(__file__), "data", "group_metrics_seurat")
data = pd.read_csv(os.path.join(root, "data.csv"), index_col=0)  # genes x cells
meta = pd.read_csv(os.path.join(root, "meta.csv"), index_col=0)


def _ann():
    mat = data.to_numpy().T
    obs = meta.assign(celltype=meta["group"], condition="c")
    ann = AnnData(X=csr_matrix(mat), obs=obs, var=pd.DataFrame(index=data.index))
    ann.layers["data"] = csr_matrix(mat)
    return ann


def _expressed_fraction(cells):
    return pd.Series((data.to_numpy()[:, cells] > 0).mean(axis=1), index=data.index)


@pytest.mark.parametrize("reference", ["rest", "B"])
def test_rank_genes_groups_pct_ref(reference):
    ann = _ann()
    groups = "all" if reference == "rest" else ["A", "C"]
    sc.tl.rank_genes_groups(
        ann, groupby="celltype", layer="data", method="wilcoxon", pts=True, groups=groups, reference=reference
    )
    res = rank_genes_groups_to_dataframe(ann, "celltype")
    celltypes = meta["group"].to_numpy()
    for group in ("A", "C"):
        rows = res[res["celltype"] == group].set_index("gene")
        reference_cells = celltypes != group if reference == "rest" else celltypes == reference
        np.testing.assert_allclose(rows["pct"], _expressed_fraction(celltypes == group)[rows.index])
        np.testing.assert_allclose(rows["pct_ref"], _expressed_fraction(reference_cells)[rows.index])


@pytest.mark.parametrize(
    ("features", "min_pct", "min_abs_lfc", "pval_thresh"),
    [
        (None, 0.3, 0, 2),
        (None, 0, 0.5, 2),
        (None, 0, 0, 0.01),
        (["gene0", "gene5", "gene20"], 0, 0, 2),
        (None, 0.2, 0.25, 0.05),
    ],
)
def test_calculate_de_scanpy_filters(features, min_pct, min_abs_lfc, pval_thresh):
    # all results (pval_thresh 2 keeps every p-value), filtered by hand
    everything = calculate_de(_ann(), "celltype", "c", "condition", use_scanpy=True, pval_thresh=2)
    expected = everything[
        (np.maximum(everything["pct"], everything["pct_ref"]) >= min_pct)
        & (everything["lfc"].abs() >= min_abs_lfc)
        & (everything["pval"] < pval_thresh)
        & (everything["gene"].isin(data.index.tolist() if features is None else features))
    ]
    result = calculate_de(
        _ann(),
        "celltype",
        "c",
        "condition",
        features=features,
        min_pct=min_pct,
        min_abs_lfc=min_abs_lfc,
        pval_thresh=pval_thresh,
        use_scanpy=True,
    )
    pd.testing.assert_frame_equal(result, expected.reset_index(drop=True))


def test_calculate_de_scanpy_min_pct_as_group_metrics():
    # min_pct keeps the same (gene, cell type) pairs as group_metrics: expressed in the cell type or in the other cells
    for min_pct in (0.1, 0.3, 0.6):
        scanpy_pairs = calculate_de(
            _ann(), "celltype", "c", "condition", min_pct=min_pct, pval_thresh=2, use_scanpy=True
        )
        ann = _ann()
        group_metrics(ann, groupby="celltype", layer="data", min_pct=min_pct)
        default_pairs = ann.uns["group_metrics"]
        assert set(zip(scanpy_pairs["gene"], scanpy_pairs["celltype"], strict=True)) == set(
            zip(default_pairs["gene"], default_pairs["celltype"], strict=True)
        )


def test_calculate_de_scanpy_unknown_feature():
    with pytest.raises(KeyError, match="Eugene"):
        calculate_de(_ann(), "celltype", "c", "condition", features=["gene0", "Eugene"], use_scanpy=True)

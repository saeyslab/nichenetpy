# type: ignore

"""
Compare group_metrics with Seurat's FindMarkers (two groups, and each group vs a reference group for group_ref without
group_oi) and FindAllMarkers (one group vs the rest):
the (gene, group) pairs that pass the min_pct / min_abs_lfc prefilter, the p-values, log fold changes and pct values.
The reference results are generated with data/group_metrics_seurat/generate_reference.R
"""

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from nichenetpy.metrics import group_metrics

root = os.path.join(os.path.dirname(__file__), "data", "group_metrics_seurat")
data = pd.read_csv(os.path.join(root, "data.csv"), index_col=0)  # genes x cells
meta = pd.read_csv(os.path.join(root, "meta.csv"), index_col=0)
reference = pd.read_csv(os.path.join(root, "seurat_markers.csv"))
settings = sorted(set(zip(reference["mode"], reference["min_pct"], reference["lfc_thr"], strict=True)))


@pytest.mark.parametrize(("mode", "min_pct", "min_abs_lfc"), settings)
def test_group_metrics_same_as_seurat(mode, min_pct, min_abs_lfc):
    if mode == "one_vs_rest":
        kwargs = {}
    elif mode.startswith("vs_"):
        kwargs = {"group_ref": mode.removeprefix("vs_")}
    else:
        kwargs = dict(zip(("group_oi", "group_ref"), mode.split("_vs_"), strict=True))
    mat = data.to_numpy().T
    ann = AnnData(X=csr_matrix(mat), obs=meta, var=pd.DataFrame(index=data.index))
    ann.layers["data"] = csr_matrix(mat)
    group_metrics(ann, groupby="group", layer="data", min_pct=min_pct, min_abs_lfc=min_abs_lfc, **kwargs)
    res = ann.uns["group_metrics"].set_index(["gene", "group"])
    ref = reference[
        (reference["mode"] == mode) & (reference["min_pct"] == min_pct) & (reference["lfc_thr"] == min_abs_lfc)
    ].set_index(["gene", "group"])
    # the same (gene, group) pairs pass the prefilter
    assert set(res.index) == set(ref.index)
    ref = ref.loc[res.index]
    np.testing.assert_allclose(res["pval"].to_numpy(), ref["pval"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(res["lfc"].to_numpy(), ref["lfc"].to_numpy(), rtol=0, atol=1e-10)
    # Seurat rounds pct.1 and pct.2 to 3 decimals
    np.testing.assert_allclose(res["pct"].to_numpy(), ref["pct"].to_numpy(), rtol=0, atol=1e-3)
    np.testing.assert_allclose(res["pct_ref"].to_numpy(), ref["pct_ref"].to_numpy(), rtol=0, atol=1e-3)

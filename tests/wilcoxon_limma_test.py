# type: ignore

"""
Compare the limma implementation of the Wilcoxon rank sum test in group_metrics (wilcoxon_limma=True) with R:
limma::rankSumTestWithCorrelation, as used by Seurat's FindMarkers(test.use = "wilcox_limma").
The reference p-values are generated with data/wilcoxon_limma/generate_reference.R
"""

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_array, csr_matrix

from nichenetpy.metrics import group_metrics

root = os.path.join(os.path.dirname(__file__), "data", "wilcoxon_limma")
data = pd.read_csv(os.path.join(root, "data.csv"), index_col=0)  # genes x cells
meta = pd.read_csv(os.path.join(root, "meta.csv"), index_col=0)
reference = pd.read_csv(os.path.join(root, "pvals_limma.csv")).set_index(["gene", "group"])["pval"]
features = ["gene0", "gene3", "gene7", "gene11", "gene12", "gene20"]


def _group_metrics(layer_type, **kwargs):
    mat = data.to_numpy().T
    ann = AnnData(X=csr_matrix(mat), obs=meta, var=pd.DataFrame(index=data.index))
    ann.layers["data"] = layer_type(mat)
    group_metrics(ann, groupby="group", layer="data", **kwargs)
    return ann.uns["group_metrics"]


@pytest.mark.parametrize("layer_type", [np.asarray, csr_matrix, csr_array])
@pytest.mark.parametrize("genes", [None, features])
def test_wilcoxon_limma_same_as_r(layer_type, genes):
    pvals = _group_metrics(layer_type, wilcoxon_limma=True, features=genes).set_index(["gene", "group"])["pval"]
    assert set(pvals.index.get_level_values("gene")) == set(data.index if genes is None else genes)
    np.testing.assert_allclose(pvals.to_numpy(), reference.reindex(pvals.index).to_numpy(), rtol=1e-6)


@pytest.mark.parametrize("kwargs", [{"min_pct": 0.5}, {"min_abs_lfc": 0.25}])
def test_wilcoxon_limma_filters_same_genes_as_default(kwargs):
    limma = _group_metrics(np.asarray, wilcoxon_limma=True, **kwargs)
    default = _group_metrics(np.asarray, wilcoxon_limma=False, **kwargs)
    assert set(limma["gene"]) == set(default["gene"])
    pvals = limma.set_index(["gene", "group"])["pval"]
    np.testing.assert_allclose(pvals.to_numpy(), reference.reindex(pvals.index).to_numpy(), rtol=1e-6)

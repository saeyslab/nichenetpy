'''
A small test for get_expressed_genes
'''

from nichenetpy.extraction import get_expressed_genes
from nichenetpy.utils import read_list_from_csv

import anndata
import os


root = os.path.dirname(__file__)
ann_2000_50 =  anndata.io.read_h5ad(os.path.join(root, "data/AnnData/anndata_2000_50.h5"))
ann_2000_50.var_names = ann_2000_50.var["gene"]

def template_expressed_genes(celltype, ann, pct, exp):
    res = set(get_expressed_genes(celltype, ann, pct))
    assert len(res) == len(exp), f"expected {len(exp)} expressed genes, got {len(res)}"
    for gene in exp:
        assert gene in res, f"expected {gene} in expressed genes"

def test_expressed_genes_0():
    template_expressed_genes(
        "CD8 T",
        ann_2000_50,
        0.05,
        read_list_from_csv(os.path.join(root, "data/expressed_genes_receiver/0/exp_genes.csv"))
    )
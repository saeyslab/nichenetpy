from nichenetpy.extraction import subset_ann_celltype

import anndata
import os


root = os.path.dirname(__file__)

def template_subset_ann_celltype(ann, celltype, layers=None, celltype_col="celltype", empty=False):
    res = subset_ann_celltype(ann, celltype, layers, celltype_col)
    if empty:
        assert res == None, "expected None to be returned"
        return
    else:
        assert type(res) is anndata.AnnData, f"wrong return type, expected AnnData, got {type(res)}"
    if type(celltype) is str:
        for ct in res.obs[celltype_col]:
            assert ct == celltype, f"expected celltype to be {celltype}, got {ct}"
    elif type(celltype) is list:
        for ct in res.obs[celltype_col]:
            assert ct in celltype, f"expected celltype to be in {celltype}, got {ct}"

def test_subset_ann_celltype_0():
    template_subset_ann_celltype(
        anndata.io.read_h5ad(os.path.join(root, "data/AnnData/anndata_2000_50.h5")),
        "B",
        layers=["data"]
    )

def test_subset_ann_celltype_1():
    template_subset_ann_celltype(
        anndata.io.read_h5ad(os.path.join(root, "data/AnnData/anndata_2000_50.h5")),
        ["CD4 T", "CD8 T"],
        layers=["data"]
    )

def test_subset_ann_celltype_2():
    template_subset_ann_celltype(
        anndata.io.read_h5ad(os.path.join(root, "data/AnnData/anndata_2000_50.h5")),
        "DC",
        layers=["data"],
        empty=True
    )

def test_subset_ann_celltype_3():
    template_subset_ann_celltype(
        anndata.io.read_h5ad(os.path.join(root, "data/AnnData/anndata_2000_50.h5")),
        "CD4 T"
    )
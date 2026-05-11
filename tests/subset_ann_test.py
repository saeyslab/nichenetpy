'''
Here we test subset_ann. 
'''

from nichenetpy.ann_utils import subset_ann

import anndata
import os


root = os.path.dirname(__file__)
ann_2000_50 =  anndata.io.read_h5ad(os.path.join(root, "data/AnnData/anndata_2000_50.h5"))

def template_subset_ann_celltype(ann, celltype, layers=None, celltype_col="celltype", empty=False):
    res = subset_ann(ann, val=celltype, layers=layers, val_col=celltype_col)
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
    for layer in res.layers.keys():
        res_shape = res.layers[layer].shape
        exp_shape = (len(res.obs), ann.layers[layer].shape[1])
        assert res_shape == exp_shape, f"expected shape of {layer}-layer to be {exp_shape}, got {res_shape}"

def test_subset_ann_celltype_0():
    template_subset_ann_celltype(
        ann_2000_50,
        "B",
        layers=["data"]
    )

def test_subset_ann_celltype_1():
    template_subset_ann_celltype(
        ann_2000_50,
        ["CD4 T", "CD8 T"],
        layers=["data"]
    )

def test_subset_ann_celltype_2():
    template_subset_ann_celltype(
        ann_2000_50,
        "CD4 T"
    )
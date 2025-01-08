from nichenetpy.extraction import subset_ann

from anndata import AnnData

import scanpy as sc


def calculate_de(
    ann:AnnData,
    condition_oi:str,
    condition_col:str,
    #condition_ref:str,
    layer="data"
):
    ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    genes = ann.var["gene"]
    ann.var_names = genes
    sc.pp.log1p(ann, layer=layer)
    sc.tl.rank_genes_groups(
        ann,
        groupby=condition_col,
        method="wilcoxon",
        layer=layer,
        #groups=[condition_oi],
        #reference=condition_ref
    )
    return ann.uns["rank_genes_groups"]

def get_avg_exp(
    ann:AnnData,
    celltype_col:str,
    condition_oi:str=None,
    condition_col:str=None,
    layer:str="data"
):
    ann = subset_ann(ann, condition_oi, layers=["data"], val_col=condition_col)
    celltypes = set(ann.obs["celltype"])
from nichenetpy.extraction import subset_ann, average_expression
from nichenetpy.normalization import relative_counts

from anndata import AnnData

import scanpy as sc
import pandas as pd


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
    layer:str="counts"
):
    if condition_col is not None and condition_oi is not None:
        ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    celltypes = set(ann.obs[celltype_col])
    avg_celltype = average_expression(
        ann,
        celltype_col,
        keys=celltypes,
        layer=layer,
        norm_f=relative_counts
    )
    avg_celltype.reset_index(inplace=True)
    return pd.melt(
        avg_celltype,
        id_vars=["gene"],
        value_vars=celltypes,
        var_name="cluster_id",
        value_name="avg_exp"
    )
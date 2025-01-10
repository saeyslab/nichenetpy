from nichenetpy.extraction import subset_ann, average_expression
from nichenetpy.normalization import relative_counts
from nichenetpy.network import LigandReceptorNetwork

from anndata import AnnData

import scanpy as sc
import pandas as pd


def calculate_de(
    ann:AnnData,
    condition_oi:str,
    condition_col:str,
    #condition_ref:str,
    layer="data"
) -> pd.DataFrame:
    ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    ann.var_names = ann.var["gene"]
    sc.pp.log1p(ann, layer=layer)
    sc.tl.rank_genes_groups(
        ann,
        #groupby=condition_col,
        method="wilcoxon",
        layer=layer,
        #groups=[condition_oi],
        #reference=condition_ref,
        pts=True
    )
    res = ann.uns["rank_genes_groups"]
    return pd.DataFrame({
        "gene": [e[0] for e in res["names"]],
        "score": [e[0] for e in res["scores"]],
        "pval": [e[0] for e in res["pvals"]],
        "pval_adj": [e[0] for e in res["pvals_adj"]],
        "lfc": [e[0] for e in res["logfoldchanges"]]
    }).merge(res["pts"].reset_index(), on="gene", how="inner")

def get_avg_exp(
    ann:AnnData,
    celltype_col:str,
    condition_oi:str=None,
    condition_col:str=None,
    layer:str="counts"
) -> pd.DataFrame:
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

def process_table_to_ic(
    tab:pd.DataFrame,
    table_type:str,
    lr_network:LigandReceptorNetwork,
    senders_oi:list[str]=None,
    receivers_oi:list[str]=None
):
    ligands = lr_network.get_ligands()
    receptors = lr_network.get_receptors()
    if table_type == "expression":
        sender_table = tab.rename({
            "cluster_id": "sender",
            "gene": "ligand",
            "avg_exp": "avg_ligand"
        })
        receiver_table = tab.rename({
            "cluster_id": "receiver",
            "gene": "receptor",
            "avg_exp": "avg_receptor"
        })
    elif (table_type == "celltype_DE"):
        sender_table = tab.rename({
            "gene": "ligand",
            "lfc": "avg_ligand",
            "pval": "pval_ligand",
            "pval_adj": "pval_adj_ligand",
            "score": "score_ligand"
        })
        receiver_table = tab.rename({
            "gene": "receiver",
            "lfc": "avg_receiver",
            "pval": "pval_receiver",
            "pval_adj": "pval_adj_receiver",
            "score": "score_receiver"
        })
    elif table_type == "group_DE":
        sender_table = tab.rename({
            "gene": "ligand",
            "lfc": "avg_ligand",
            "pval": "pval_ligand",
            "pval_adj": "pval_adj_ligand",
            "score": "score_ligand"
        })
        receiver_table = tab.rename({
            "gene": "receiver",
            "lfc": "avg_receiver",
            "pval": "pval_receiver",
            "pval_adj": "pval_adj_receiver",
            "score": "score_receiver"
        })
        # TODO: merge if possible
    else:
        raise ValueError("table_type argument should be 'expression', 'celltype_DE' or 'group_DE'")
from nichenetpy.extraction import subset_ann, average_expression, _subset_layer
from nichenetpy.normalization import relative_counts
from nichenetpy.network import LigandReceptorNetwork

from anndata import AnnData

import scanpy as sc
import pandas as pd


def calculate_de(
    ann:AnnData,
    celltype_col:str,
    condition_oi:str,
    condition_col:str,
    layer="data",
    features:list[str]=None,
    gene_field="gene"
) -> pd.DataFrame:
    ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    if features is not None:
        ann = _subset_layer(ann, layer, features)
        ann.var_names = features
    else:
        ann.var_names = ann.var[gene_field]
    sc.pp.log1p(ann, layer=layer)
    sc.tl.rank_genes_groups(
        ann,
        groupby=celltype_col,
        method="wilcoxon",
        layer=layer,
        pts=True
    )
    res = ann.uns["rank_genes_groups"]
    output = pd.melt(pd.DataFrame(res["names"]), var_name="celltype", value_name="gene")
    for col in ["pvals", "pvals_adj", "logfoldchanges"]:
        temp = pd.melt(pd.DataFrame(res[col]), var_name="celltype", value_name=col)
        temp.drop(columns={"celltype"}, inplace=True)
        output = output.join(temp, how="inner")
    temp = pd.melt(res["pts"], var_name="celltype", value_name="pts", ignore_index=False)
    temp.index.name = "gene"
    temp.reset_index(inplace=True)
    output = output.merge(temp, on=["gene", "celltype"], how="inner")
    return output

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
        var_name="celltype",
        value_name="avg_exp"
    )

def process_table_to_ic(
    tab:pd.DataFrame,
    table_type:str,
    lr_network:LigandReceptorNetwork,
    senders_oi:list[str]=None,
    receivers_oi:list[str]=None
):
    if table_type == "expression":
        sender_table = tab.rename(columns={
            "celltype": "sender",
            "gene": "ligand",
            "avg_exp": "avg_ligand"
        })
        receiver_table = tab.rename(columns={
            "celltype": "receiver",
            "gene": "receptor",
            "avg_exp": "avg_receptor"
        })
        columns_reorder = [
            "sender",
            "receiver",
            "ligand",
            "receptor",
            "avg_ligand",
            "avg_receptor",
            "ligand_receptor_prod"
        ]
    elif (table_type == "celltype_DE"):
        sender_table = tab.rename(columns={
            "celltype": "sender",
            "gene": "ligand",
            "logfoldchanges": "lfc_ligand",
            "pvals": "pval_ligand",
            "pvals_adj": "pval_adj_ligand",
            "pts": "pct_expressed_sender"
        })
        receiver_table = tab.rename(columns={
            "celltype": "receiver",
            "gene": "receptor",
            "logfoldchanges": "lfc_receptor",
            "pvals": "pval_receptor",
            "pvals_adj": "pval_adj_receptor",
            "pts": "pct_expressed_receiver"
        })
        columns_reorder = [
            "sender",
            "receiver",
            "ligand",
            "receptor",
            "lfc_ligand",
            "lfc_receptor",
            "ligand_receptor_lfc_avg",
            "pval_ligand",
            "pval_adj_ligand",
            "pval_receptor",
            "pval_adj_receptor",
            "pct_expressed_sender",
            "pct_expressed_receiver"
        ]
    elif table_type == "group_DE":
        sender_table = tab.rename(columns={
            "gene": "ligand",
            "logfoldchanges": "lfc_ligand",
            "pval": "pval_ligand",
            "pval_adj": "pval_adj_ligand"
        })
        receiver_table = tab.rename(columns={
            "gene": "receptor",
            "logfoldchanges": "lfc_receptor",
            "pval": "pval_receptor",
            "pval_adj": "pval_adj_receptor"
        })
        columns_reorder = [
            "ligand",
            "receptor",
            "lfc_ligand",
            "lfc_receptor",
            "ligand_receptor_lfc_avg",
            "pval_ligand",
            "pval_adj_ligand",
            "pval_receptor",
            "pval_adj_receptor"
        ]
    else:
        raise ValueError("table_type argument should be 'expression', 'celltype_DE' or 'group_DE'")
    if senders_oi is not None:
        sender_table = sender_table[[sender in senders_oi for sender in sender_table["sender"]]]
    if receivers_oi is not None:
        receiver_table = receiver_table[[receiver in receivers_oi for receiver in receiver_table["receiver"]]]
    sender_receiver_table = (
        pd.DataFrame(lr_network)
        .rename(columns={0: "ligand", 1: "receptor"})
        .merge(sender_table, on="ligand", how="inner")
        .merge(receiver_table, on="receptor", how="inner")
    )
    if table_type == "expression":
        sender_receiver_table["ligand_receptor_prod"] = [
            x * y
            for x, y in zip(
                sender_receiver_table["avg_ligand"],
                sender_receiver_table["avg_receptor"]
            )
        ]
        sender_receiver_table.sort_values(
            by="ligand_receptor_prod",
            ascending=False,
            inplace=True
        )
    else:
        sender_receiver_table["ligand_receptor_lfc_avg"] = [
            (x + y) / 2
            for x, y in zip(
                sender_receiver_table["lfc_ligand"],
                sender_receiver_table["lfc_receptor"]
            )
        ]
        sender_receiver_table.sort_values(
            by="ligand_receptor_lfc_avg",
            ascending=False,
            inplace=True
        )
    return sender_receiver_table[columns_reorder]
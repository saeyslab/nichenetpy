from nichenetpy.extraction import subset_ann, average_expression, _subset_layer
from nichenetpy.normalization import relative_counts, scaling_zscore, scale_quantile_adapted
from nichenetpy.network import LigandReceptorNetwork
from nichenetpy.utils import ligand_activities_df, df_grouped_apply
from nichenetpy.metrics import group_metrics

from anndata import AnnData

import pandas as pd
import numpy as np


def calculate_de(
    ann:AnnData,
    celltype_col:str,
    condition_oi:str,
    condition_col:str,
    layer="data",
    features:list[str]=None
) -> pd.DataFrame:
    '''
    Calculate differential expression of one cell type versus all other cell types using group_metrics.
    If condition_oi is provided, only consider cells from that condition.

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    celltype_col : str
        the column in ann.obs which contains the celltypes
    condition_oi : str
        The condition of interest
    condition_col : str
        the column in ann.obs which contains the conditions
    layer : str
        the layer of the AnnData object to use
    features : list of str
        the genes to consider
    
    Returns
    -------
    pandas.DataFrame
        the differential expression
    '''
    ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    if features is not None:
        ann = _subset_layer(ann, layer, features)
        ann.var_names = features
    group_metrics(
        ann,
        groupby=celltype_col,
        layer=layer
    )
    res = ann.uns["group_metrics"]
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
    '''
    Calculate the average gene expression per cell type.
    If condition_oi is provided, only consider cells from that condition.

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    celltype_col : str
        the column in ann.obs which contains the celltypes
    condition_oi : str
        The condition of interest
    condition_col : str
        the column in ann.obs which contains the conditions
    layer : str
        the layer of the AnnData object to use
    
    Returns
    -------
    pandas.DataFrame
        the average gene expression per cell type
    '''
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
    '''
    First, only keep information of ligands for senders_oi, and information of receptors for receivers_oi.
    Then, combine information for senders and receivers by linking ligands to receptors based on the prior knowledge ligand-receptor network.

    Parameters
    ----------
    tab : pandas.DataFrame
        the table to process
    table_type : str
        "expression", "celltype_DE", or "group_DE"
        indicates whether the table contains expression, celltype markers, or condition-specific information
    lr_network : LigandReceptorNetwork
        prior knowledge Ligand-Receptor network
    senders_oi : list of str
        the sender celltypes of interest
    receivers_oi : list of str
        the receiver celltypes of interest
    
    Returns
    -------
    pandas.DataFrame
        the processed table
    '''
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

def _prioritization(
    de:pd.DataFrame,
    lig_rec:str,
    send_rcvr:str=None
):
    output = de[
        [lig_rec, f"lfc_{lig_rec}", f"pval_{lig_rec}"]
        if send_rcvr is None else
        [send_rcvr, lig_rec, f"lfc_{lig_rec}", f"pval_{lig_rec}"]
    ]
    output.drop_duplicates(inplace=True)
    output[f"lfc_pval_{lig_rec}"] = (
        -1 *
        np.log10(output[f"pval_{lig_rec}"]) *
        output[f"lfc_{lig_rec}"]
    )
    temp = -np.log10(output[f"pval_{lig_rec}"])
    output[f"lfc_pval_{lig_rec}"] = temp * output[f"lfc_{lig_rec}"]
    output[f"pval_adapted_{lig_rec}"] = (
        temp * output[f"lfc_{lig_rec}"].apply(lambda x : -1 if x < 0 else 1)
    )
    temp = output[f"lfc_{lig_rec}"].rank(method="average", na_option="top")
    output[f"scaled_lfc_{lig_rec}"] = temp / temp.max()
    temp = output[f"pval_{lig_rec}"].rank(method="average", na_option="top", ascending=False)
    output[f"scaled_pval_{lig_rec}"] = temp / temp.max()
    temp = output[f"lfc_pval_{lig_rec}"].rank(method="average", na_option="top")
    output[f"scaled_lfc_pval_{lig_rec}"] = temp / temp.max()
    temp = output[f"pval_adapted_{lig_rec}"].rank(method="average", na_option="top")
    output[f"scaled_pval_adapted_{lig_rec}"] = temp / temp.max()
    output.sort_values(by=f"lfc_pval_{lig_rec}", ascending=False, inplace=True)
    return output

def generate_prioritization_table(
    sender_receiver_info:pd.DataFrame,
    sender_receiver_de:pd.DataFrame,
    ligand_activities:pd.DataFrame|dict[str, dict[str, float]]|list[tuple[str, dict[str, float]]],
    lr_condition_de:pd.DataFrame=None,
    prioritizing_weights:dict[str, float]=None
):
    '''
    User can choose the importance attached to each of the following prioritization criteria:
        differential expression of ligand and receptor,
        cell-type specificity of expression of ligand and receptor,
        NicheNet ligand activity

    Parameters
    ----------
    sender_receiver_info : pandas.DataFrame
        processed output of get_avg_exp
    sender_receiver_de : pandas.DataFrame
        processed output of calculate_de
    ligand_activities : pandas.DataFrame
        output of predict_ligand_activities
    lr_condition_de : pandas.DataFrame
        processed output of group_metrics
    prioritizing_weights : dict
        a dictionary indicating the relative weights of each prioritization criterion
        If provided, the dictionary must contain the following names:
            "de_ligand",
            "de_receptor",
            "activity_scaled",
            "exprs_ligand",
            "exprs_receptor",
            "ligand_condition_specificity",
            "receptor_condition_specificity"
    
    Returns
    -------
    pandas.DataFrame
        the processed table
    '''
    pd.options.mode.chained_assignment = None # false positive warnings removal
    if type(ligand_activities) is dict or type(ligand_activities) is list:
        ligand_activities = ligand_activities_df(ligand_activities)
    elif type(ligand_activities) is not pd.DataFrame:
        raise TypeError(f"ligand_activities should be of type pandas.DataFrame, dict or list, was {type(ligand_activities)}")
    if prioritizing_weights is None:
        prioritizing_weights = {
            "de_ligand": 1,
            "de_receptor": 1,
            "activity_scaled": 1,
            "exprs_ligand": 1,
            "exprs_receptor": 1,
            "ligand_condition_specificity": 0,
            "receptor_condition_specificity": 0
        } if lr_condition_de is None else {
            "de_ligand": 1,
            "de_receptor": 1,
            "activity_scaled": 1,
            "exprs_ligand": 1,
            "exprs_receptor": 1,
            "ligand_condition_specificity": 1,
            "receptor_condition_specificity": 1
        }
    else:
        for key in (
            "de_ligand",
            "de_receptor",
            "activity_scaled",
            "exprs_ligand",
            "exprs_receptor",
            "ligand_condition_specificity",
            "receptor_condition_specificity"
        ):
            if key not in lr_condition_de:
                return ValueError(f"{key} key missing in lr_condition_de")
    if "rank" not in ligand_activities.columns:
        ligand_activities["rank"] = ligand_activities[["aupr_corrected"]].rank(method="average", na_option="bottom", ascending=False)
    sender_receiver = sender_receiver_de[["sender", "receiver"]]
    sender_receiver.drop_duplicates(inplace=True)
    sender_ligand_prioritization = _prioritization(
        sender_receiver_de,
        "ligand",
        "sender"
    )
    receiver_receptor_prioritization = _prioritization(
        sender_receiver_de,
        "receptor",
        "receiver"
    )
    if "receiver" in ligand_activities.columns:
        ligand_activity_prioritization = ligand_activities[["aupr_corrected", "rank", "receiver"]]
    else:
        ligand_activity_prioritization = ligand_activities[["aupr_corrected", "rank"]]
    ligand_activity_prioritization.index.name = "ligand"
    ligand_activity_prioritization.reset_index(inplace=True)
    ligand_activity_prioritization.rename(columns={"aupr_corrected": "activity"}, inplace=True)
    ligand_activity_prioritization["activity_zscore"] = scaling_zscore(ligand_activity_prioritization["activity"])
    ligand_activity_prioritization["scaled_activity"] = scale_quantile_adapted(
        ligand_activity_prioritization["activity"].transpose(),
        cutoff=0.01
    ).transpose()
    ligand_activity_prioritization.sort_values(by="activity_zscore", ascending=False, inplace=True)
    ligand_celltype_specificity_prioritization = sender_receiver_info[["sender", "ligand", "avg_ligand"]]
    ligand_celltype_specificity_prioritization.drop_duplicates(inplace=True)
    ligand_celltype_specificity_prioritization = df_grouped_apply(
        ligand_celltype_specificity_prioritization,
        groupby="ligand",
        func=lambda group : scale_quantile_adapted(group["avg_ligand"]),
        dest="scaled_avg_exprs_ligand"
    )
    ligand_celltype_specificity_prioritization.sort_values(
        by="scaled_avg_exprs_ligand",
        ascending=False,
        inplace=True
    )
    receptor_celltype_specificity_prioritization = sender_receiver_info[["receiver", "receptor", "avg_receptor"]]
    receptor_celltype_specificity_prioritization.drop_duplicates(inplace=True)
    receptor_celltype_specificity_prioritization = df_grouped_apply(
        receptor_celltype_specificity_prioritization,
        groupby="receptor",
        func=lambda group : scale_quantile_adapted(group["avg_receptor"]),
        dest="scaled_avg_exprs_receptor"
    )
    receptor_celltype_specificity_prioritization.sort_values(
        by="scaled_avg_exprs_receptor",
        ascending=False,
        inplace=True
    )
    if lr_condition_de is not None:
        ligand_condition_prioritization = _prioritization(lr_condition_de, "ligand")
        ligand_condition_prioritization.index = ligand_condition_prioritization["ligand"]
        ligand_condition_prioritization.drop(columns=["ligand"], inplace=True)
        ligand_condition_prioritization.rename(columns=lambda x : f"{x}_group", inplace=True)
        ligand_condition_prioritization.reset_index(inplace=True)
        receptor_condition_prioritization = _prioritization(lr_condition_de, "receptor")
        receptor_condition_prioritization.index = receptor_condition_prioritization["receptor"]
        receptor_condition_prioritization.drop(columns=["receptor"], inplace=True)
        receptor_condition_prioritization.rename(columns=lambda x : f"{x}_group", inplace=True)
        receptor_condition_prioritization.reset_index(inplace=True)
    group_prioritization = sender_receiver_de.merge(sender_receiver_info, how="inner")
    if prioritizing_weights["de_ligand"] > 0:
        group_prioritization = group_prioritization.merge(sender_ligand_prioritization, how="inner")
    if prioritizing_weights["activity_scaled"] > 0:
        group_prioritization = group_prioritization.merge(ligand_activity_prioritization, how="inner")
    if prioritizing_weights["de_receptor"] > 0:
        group_prioritization = group_prioritization.merge(receiver_receptor_prioritization, how="inner")
    if prioritizing_weights["exprs_ligand"] > 0:
        group_prioritization = group_prioritization.merge(ligand_celltype_specificity_prioritization, how="inner")
    if prioritizing_weights["exprs_receptor"] > 0:
        group_prioritization = group_prioritization.merge(receptor_celltype_specificity_prioritization, how="inner")
    if prioritizing_weights["ligand_condition_specificity"] > 0:
        group_prioritization = group_prioritization.merge(ligand_condition_prioritization, how="inner")
    if prioritizing_weights["receptor_condition_specificity"] > 0:
        group_prioritization = group_prioritization.merge(receptor_condition_prioritization, how="inner")
    sum_prioritization_weights = (
        (
            prioritizing_weights["de_ligand"] +
            prioritizing_weights["de_receptor"] +
            prioritizing_weights["exprs_ligand"] +
            prioritizing_weights["exprs_receptor"]
        ) / 2 +
        prioritizing_weights["activity_scaled"] +
        prioritizing_weights["ligand_condition_specificity"] +
        prioritizing_weights["receptor_condition_specificity"]
    )
    score = 0
    if "scaled_p_val_adapted_ligand" in group_prioritization.columns:
        score += prioritizing_weights["de_ligand"] * group_prioritization["scaled_p_val_adapted_ligand"] / 2
    if "scaled_p_val_adapted_receptor" in group_prioritization.columns:
        score += prioritizing_weights["de_receptor"] * group_prioritization["scaled_p_val_adapted_receptor"] / 2
    if "scaled_activity" in group_prioritization.columns:
        score += prioritizing_weights["activity_scaled"] * group_prioritization["scaled_activity"]
    if "scaled_avg_exprs_ligand" in group_prioritization.columns:
        score += prioritizing_weights["exprs_ligand"] * group_prioritization["scaled_avg_exprs_ligand"] / 2
    if "scaled_avg_exprs_receptor" in group_prioritization.columns:
        score += prioritizing_weights["exprs_receptor"] * group_prioritization["scaled_avg_exprs_receptor"] / 2
    if "scaled_p_val_adapted_ligand_group" in group_prioritization.columns:
        score += prioritizing_weights["ligand_condition_specificity"] * group_prioritization["scaled_p_val_adapted_ligand_group"]
    if "scaled_p_val_adapted_receptor_group" in group_prioritization.columns:
        score += prioritizing_weights["receptor_condition_specificity"] * group_prioritization["scaled_p_val_adapted_receptor_group"]
    score /= sum_prioritization_weights
    group_prioritization["prioritization_score"] = score
    group_prioritization.sort_values(by="prioritization_score", ascending=False, inplace=True)
    group_prioritization["prioritization_rank"] = group_prioritization[["prioritization_score"]].rank(
        method="average",
        na_option="bottom",
        ascending=False
    )
    return group_prioritization
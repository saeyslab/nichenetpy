from nichenetpy.extraction import average_expression
from nichenetpy.normalization import (
    relative_counts,
    scaling_zscore,
    scale_quantile_adapted
)
from nichenetpy.network import LigandReceptorNetwork
from nichenetpy.utils import (
    ligand_activities_df,
    df_grouped_apply,
    rank_genes_groups_to_dataframe
)
from nichenetpy.metrics import group_metrics
from nichenetpy.ann_utils import subset_ann

from anndata import AnnData
from collections.abc import Iterable, Collection
from numbers import Number

import pandas as pd
import numpy as np
import scanpy as sc


def calculate_de(
    ann:AnnData,
    celltype_col:str,
    condition_oi:str,
    condition_col:str,
    layer="data",
    features:Iterable[str]=None,
    min_abs_lfc:float=0,
    min_pct:float=0,
    pval_thresh:float=1,
    use_scanpy:bool=False
) -> pd.DataFrame:
    '''
    Calculate differential expression of one cell type versus all other cell types.
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
    features : Iterable of str
        the genes to consider
    min_abs_lfc : float
        genes with a lfc lower than this value will be excluded from the wilcoxon rank sum test
    min_pct : float
        genes with a pct lower than this value will be excluded from the wilcoxon rank sum test
    pval_thresh : float
        upper bound for the p-values (if p_values for a gene is smaller than this threshold, it is excluded)
    use_scanpy : bool
        if True, use scanpy.rank_genes_groups
    
    Returns
    -------
    pandas.DataFrame
        the differential expression

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type AnnData, was {type(ann)}")
    if type(celltype_col) is not str:
        raise TypeError(f"celltype_col should have type str, was {type(celltype_col)}")
    if type(condition_oi) is not str:
        raise TypeError(f"condition_oi should have type str, was {type(condition_oi)}")
    if type(condition_col) is not str:
        raise TypeError(f"condition_col should have type str, was {type(condition_col)}")
    if type(layer) is not str:
        raise TypeError(f"layer should have type str, was {type(layer)}")
    if not isinstance(features, Iterable):
        raise TypeError(f"features should have type Iterable[str], was {type(features)}")
    if not isinstance(min_abs_lfc, Number):
        raise TypeError(f"min_abs_lfc should have type float, was {type(min_abs_lfc)}")
    if not isinstance(min_pct, Number):
        raise TypeError(f"min_pct should have type float, was {type(min_pct)}")
    if not isinstance(pval_thresh, Number):
        raise TypeError(f"pval_thresh should have type float, was {type(pval_thresh)}")
    if not type(use_scanpy) is bool:
        raise TypeError(f"use_scanpy should have type bool, was {type(use_scanpy)}")
    ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    if use_scanpy:
        sc.tl.rank_genes_groups(
            ann,
            groupby=celltype_col,
            method="wilcoxon",
            layer=layer,
            pts=True
        )
        return rank_genes_groups_to_dataframe(ann, groupby=celltype_col)
    else:
        group_metrics(
            ann,
            groupby=celltype_col,
            layer=layer,
            min_abs_lfc=min_abs_lfc,
            min_pct=min_pct,
            pval_thresh=pval_thresh,
            features=features
        )
        return ann.uns["group_metrics"]

def get_avg_exp(
    ann:AnnData,
    celltype_col:str,
    condition_oi:str=None,
    condition_col:str=None,
    layer:str="counts",
    features:Iterable[str]=None
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
    features : Iterable[str]
        the genes to use, if None, use all genes from the AnnData object
    
    Returns
    -------
    pandas.DataFrame
        the average gene expression per cell type
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type AnnData, was {type(ann)}")
    if type(celltype_col) is not str:
        raise TypeError(f"celltype_col should have type str, was {type(celltype_col)}")
    if type(condition_oi) is not str:
        raise TypeError(f"condition_oi should have type str, was {type(condition_oi)}")
    if type(condition_col) is not str:
        raise TypeError(f"condition_col should have type str, was {type(condition_col)}")
    if type(layer) is not str:
        raise TypeError(f"layer should have type str, was {type(layer)}")
    if features is not None and not isinstance(features, Iterable):
        raise TypeError(f"features should have type Iterable[str], was {type(features)}")
    if condition_col is not None and condition_oi is not None:
        ann = subset_ann(ann, condition_oi, layers=[layer], val_col=condition_col)
    if features is not None:
        ann = subset_ann(ann, genes=features)
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
    senders_oi:Collection[str]=None,
    receivers_oi:Collection[str]=None
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
    senders_oi : Collection of str
        the sender celltypes of interest
    receivers_oi : Collection of str
        the receiver celltypes of interest
    
    Returns
    -------
    pandas.DataFrame
        the processed table
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(tab) is not pd.DataFrame:
        raise TypeError(f"tab should have type pandas.DataFrame, was {type(tab)}")
    if type(table_type) is not str:
        raise TypeError(f"table_type should have type str, was {type(table_type)}")
    if type(lr_network) is not LigandReceptorNetwork:
        raise TypeError(f"lr_network should have type LigandReceptorNetwork, was {type(lr_network)}")
    if senders_oi is not None and not isinstance(senders_oi, Collection):
        raise TypeError(f"senders_oi should have type Collection[str], was {type(senders_oi)}")
    if receivers_oi is not None and not isinstance(receivers_oi, Collection):
        raise TypeError(f"receivers_oi should have type Collection[str], was {type(receivers_oi)}")
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
            "lfc": "lfc_ligand",
            "pval": "pval_ligand",
            "pval_adj": "pval_adj_ligand",
            "pct": "pct_expressed_sender"
        })
        receiver_table = tab.rename(columns={
            "celltype": "receiver",
            "gene": "receptor",
            "lfc": "lfc_receptor",
            "pval": "pval_receptor",
            "pval_adj": "pval_adj_receptor",
            "pct": "pct_expressed_receiver"
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
            "lfc": "lfc_ligand",
            "pval": "pval_ligand",
            "pval_adj": "pval_adj_ligand"
        })
        receiver_table = tab.rename(columns={
            "gene": "receptor",
            "lfc": "lfc_receptor",
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
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if prioritizing weigths does not contain the correct keys
    '''
    if type(sender_receiver_info) is not pd.DataFrame:
        raise TypeError(f"sender_receiver_info should be of type pandas.DataFrame, was {type(sender_receiver_info)}")
    if type(sender_receiver_de) is not pd.DataFrame:
        raise TypeError(f"sender_receiver_de should be of type pandas.DataFrame, was {type(sender_receiver_de)}")
    if lr_condition_de is not None and type(lr_condition_de) is not pd.DataFrame:
        raise TypeError(f"lr_condition_de should be of type pandas.DataFrame, was {type(lr_condition_de)}")
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
        if type(prioritizing_weights) is not dict:
            raise TypeError(f"prioritizing_weights should have type dict, was {type(prioritizing_weights)}")
        for key in (
            "de_ligand",
            "de_receptor",
            "activity_scaled",
            "exprs_ligand",
            "exprs_receptor",
            "ligand_condition_specificity",
            "receptor_condition_specificity"
        ):
            if key not in prioritizing_weights:
                raise ValueError(f"{key} key missing in prioritizing_weights")
    if "rank" not in ligand_activities.columns:
        ligand_activities["rank"] = ligand_activities[["aupr_corrected"]].rank(method="average", na_option="bottom", ascending=False)
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
        cutoff=0.01,
        by_row=True
    ).transpose()
    ligand_activity_prioritization.sort_values(by="activity_zscore", ascending=False, inplace=True)
    ligand_celltype_specificity_prioritization = sender_receiver_info[["sender", "ligand", "avg_ligand"]]
    ligand_celltype_specificity_prioritization.drop_duplicates(inplace=True)
    ligand_celltype_specificity_prioritization = df_grouped_apply(
        ligand_celltype_specificity_prioritization,
        groupby="ligand",
        func=lambda group : scale_quantile_adapted(group["avg_ligand"], by_row=True),
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
        func=lambda group : scale_quantile_adapted(group["avg_receptor"], by_row=True),
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
    if "scaled_pval_adapted_ligand" in group_prioritization.columns:
        score += prioritizing_weights["de_ligand"] * group_prioritization["scaled_pval_adapted_ligand"] / 2
    if "scaled_pval_adapted_receptor" in group_prioritization.columns:
        score += prioritizing_weights["de_receptor"] * group_prioritization["scaled_pval_adapted_receptor"] / 2
    if "scaled_activity" in group_prioritization.columns:
        score += prioritizing_weights["activity_scaled"] * group_prioritization["scaled_activity"]
    if "scaled_avg_exprs_ligand" in group_prioritization.columns:
        score += prioritizing_weights["exprs_ligand"] * group_prioritization["scaled_avg_exprs_ligand"] / 2
    if "scaled_avg_exprs_receptor" in group_prioritization.columns:
        score += prioritizing_weights["exprs_receptor"] * group_prioritization["scaled_avg_exprs_receptor"] / 2
    if "scaled_pval_adapted_ligand_group" in group_prioritization.columns:
        score += prioritizing_weights["ligand_condition_specificity"] * group_prioritization["scaled_pval_adapted_ligand_group"]
    if "scaled_pval_adapted_receptor_group" in group_prioritization.columns:
        score += prioritizing_weights["receptor_condition_specificity"] * group_prioritization["scaled_pval_adapted_receptor_group"]
    score /= sum_prioritization_weights
    group_prioritization["prioritization_score"] = score
    group_prioritization.sort_values(by="prioritization_score", ascending=False, inplace=True)
    group_prioritization["prioritization_rank"] = group_prioritization[["prioritization_score"]].rank(
        method="average",
        na_option="bottom",
        ascending=False
    )
    return group_prioritization
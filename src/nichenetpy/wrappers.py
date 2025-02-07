from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.utils import (
    combine_by_key,
    combine_dicts
)
from nichenetpy.extraction import (
    get_expressed_genes,
    subset_ann,
    get_weighted_ligand_receptor_links,
    get_lfc_celltype
)
from nichenetpy.visualization import (
    prepare_ligand_target_visualization,
    prepare_ligand_receptor_visualization,
    heatmap_2d,
    heatmap_1d
)
from nichenetpy.prioritization import (
    calculate_de,
    get_avg_exp,
    process_table_to_ic,
    generate_prioritization_table
)
from nichenetpy.metrics import group_metrics

from itertools import cycle, chain
from collections.abc import Iterable
from anndata import AnnData

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_geneset_oi(
    ann:AnnData,
    receiver:str,
    condition_oi:str,#TODO
    condition_ref:str,#TODO
    layer:str="data",
    condition_col:str="aggregate",
    max_pval_adj:float=0.05,
    min_abs_lfc:float=0.25,
    min_pct:float=0.05
) -> set[str]:
    '''
    Gets the geneset of interest from an AnnData object. The gene set of interest are genes within the receiver cell type that are likely to be influenced by ligands from the CCC event. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to extract expressed genes from
    receiver : str
        the receiver cell type
    condition_oi : str
        the condition of interest
    condition_ref : str
        the reference condition
    layer : str
        the name of the layer which contains the data matrix
    condition_col : str
        the name of the column in obs which contains the conditions
    max_pval_adj : float
        the upper bound for pval_adj
    min_lfc : float
        the lower bound for lfc,
    min_pct : float
        the lower bound for the pct
    
    Returns
    -------
    list
        the geneset of interest
    '''
    ann_receiver = subset_ann(ann, receiver, layers=[layer])
    group_metrics(
        ann_receiver,
        groupby=condition_col,
        layer=layer,
        min_pct=min_pct,
        min_abs_lfc=min_abs_lfc
    )
    DE_table = ann_receiver.uns["group_metrics"]
    return set(
        DE_table[
            (DE_table[condition_col] == "LCMV") &
            (DE_table["pval_adj"] <= max_pval_adj)
        ]["gene"]
    )

def combine_weighted_ligand_target_links(active_ligand_target_links:Iterable[dict]) -> list[tuple[str, str, float]]:
    '''
    Combines weighted ligand-target links of different ligands. 

    Parameters
    ----------
    active_ligand_target_links : list
        list of ligand-target links as returned by LigandActivityPredictor.get_weighted_ligand_target_links
    
    Returns
    -------
    list
        list of (ligand, target, weight) tuples representing the combined ligand-target links
    '''
    return list(
        chain(
            *(zip(cycle([e["ligand"]]), e["target"], e["weight"]) for e in active_ligand_target_links)
        )
    )

def run_nichenet(
    ann:AnnData,
    predictor:LigandActivityPredictor,
    lr_network:LigandReceptorNetwork,
    receiver:str,
    condition_oi:str,
    condition_ref:str,
    sender_celltypes:Iterable[str]=None,
    get_expressed_genes_pct:float=0.05,
    layer:str="data",
    condition_col:str="aggregate",
    celltype_col:str="celltype",
    max_pval_adj:float=0.05,
    min_abs_lfc:float=0.25,
    min_pct:float=0.05,
    ligands_top_n:int=30,
    targets_top_n:int=100,
    lr_sig:WeightedNetwork=None,
    get_ltl:bool=False,
    get_lfc:bool=False,
    get_prioritization_table:bool=False,#TODO: add output to docs
    case_control:bool=True #TODO: add to docs
):
    '''
    Runs a standard nichenet analysis. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    predictor : LigandActivityPredictor
        the predictor which contains the ligand-target matrix
    lr_network : LigandReceptorNetwork
        the ligand-receptor network containing the ligand-receptor interactions
    receiver : str
        the receiver cell type
    condition_oi : str
        the condition of interest
    condition_ref : str
        the reference condition
    sender_celltypes : Iterable[str]
        the sender cell types for the sender-focused approach, if None only the sender-agnostic analysis is performed
    get_expressed_genes_pct : float
        the minimum percent difference between the percent of cells expressing the gene in the cluster and the percent of cells
    layer : str
        the layer in the AnnData object which contains the data matrix
    condition_col : str
        the name of the column in obs which contains the conditions
    celltype_col : str
        the name of the column in obs which contains the celltypes
    max_pval_adj : float
        the upper bound for pval_adj
    min_abs_lfc : float
        the lower bound for lfc
    min_pct : float
        the lower bound for the pct
    ligands_top_n : int
        the amount of ligands that are considered to be the best upstream ligands
    targets_top_n : int
        the number of target genes to consider per ligand when performing the target gene inference
    lr_sig : WeightedNetwork
        a weighted network containing the ligand-receptor interactions and their weights
        if None, the ligand receptor links are not computed
    get_ltl : bool
        if true, the active ligand-target links are computed and returned
    get_lfc : bool
        if true, the log fold changes are computed and returned
    get_prioritization_table : bool
        if true, the prioritization table is computed and returned
    case_control : bool
        the case_control argument for generate_info_tables
    
    Returns
    -------
    dict
        a dictionary which contains the output of the analysis, it contains the following objects (depending on certain arguments) for the sender-agnostic approach:
            best_upstream_ligands : list of str
                the top scoring ligands in the sender-agnostic approach
            ligand_activities_sorted : dict
                the computed metrics for each ligand in the sender-agnostic approach
            active_ligand_target_links : list of tuple
                list of (ligand, target, weight) tuples representing the ligand-target links in the sender-agnostic approach
            ligand_receptor_links : WeightedNetwork
                the weighted ligand-receptor links in the sender-agnostic approach
            expressed_receptors : set of str
                the expressed receptors
            expressed_genes_receiver : set of str
                expressed genes in the receiver
            prioritization_table : pandas.DataFrame
                Data frame of prioritized sender-ligand-receiver-receptor interactions
        and the following additional objects for the sender-focused approach:
            best_upstream_ligands_focused : list of str
                the top scoring ligands in the sender-focused approach
            ligand_activities_sorted_focused : dict
                the computed metrics for each ligand in the sender-focused approach
            active_ligand_target_links_focused : list of tuple
                list of (ligand, target, weight) tuples representing the ligand-target links in the sender-focused approach
            ligand_receptor_links_focused : WeightedNetwork
                the weighted ligand-receptor links in the sender-focused approach
            ann_focused : AnnData
                the AnnData object used in the sender-focused approach (new object derived from ann)
            lfcs : list of tuple
                the log fold changes as a list of tuples of lists where the first list of each tuple contains the ligands and second list contains the values
            expressed_ligands : set of str
                the expressed ligands
    '''
    output = dict()
    expressed_genes_receiver = set(
        get_expressed_genes(receiver, ann, pct=get_expressed_genes_pct, celltype_col=celltype_col)
    )
    output["expressed_genes_receiver"] = expressed_genes_receiver
    expressed_receptors = lr_network.get_receptors().intersection(expressed_genes_receiver)
    output["expressed_receptors"] = expressed_receptors
    potential_ligands = set(
        key for key, group in lr_network.item_iter()
        if len(group.intersection(expressed_receptors)) > 0
    )
    geneset = get_geneset_oi(
        ann,
        receiver=receiver,
        condition_oi=condition_oi,
        condition_ref=condition_ref,
        layer=layer,
        condition_col=condition_col,
        max_pval_adj=max_pval_adj,
        min_abs_lfc=min_abs_lfc,
        min_pct=min_pct
    )
    geneset.intersection_update(predictor.get_genes())
    ligand_activities = predictor.predict_ligand_activities(
        geneset=geneset,
        background_expressed_genes=expressed_genes_receiver,
        potential_ligands=potential_ligands
    )
    ligand_activities_sorted = sorted(ligand_activities.items(), key=lambda x : x[1]["aupr_corrected"], reverse=True)
    output["ligand_activities_sorted"] = ligand_activities_sorted
    best_upstream_ligands = [e[0] for e in ligand_activities_sorted[:ligands_top_n]]
    output["best_upstream_ligands"] = best_upstream_ligands
    if get_ltl:
        output["active_ligand_target_links"] = combine_weighted_ligand_target_links((
            predictor.get_weighted_ligand_target_links(ligand, geneset, n=targets_top_n)
            for ligand in best_upstream_ligands
        ))
    if lr_sig is not None:
        output["ligand_receptor_links"] = get_weighted_ligand_receptor_links(
            best_upstream_ligands,
            expressed_receptors,
            lr_network,
            lr_sig
        )
    if sender_celltypes is not None:
        list_expressed_genes_sender = [
            get_expressed_genes(
                ct,
                ann,
                pct=get_expressed_genes_pct,
                celltype_col=celltype_col
            ) for ct in sender_celltypes
        ]
        expressed_genes_sender = set(e for l in list_expressed_genes_sender for e in l)
        expressed_ligands = lr_network.get_ligands().intersection(expressed_genes_sender)
        output["expressed_ligands"] = expressed_ligands
        potential_ligands_focused = potential_ligands.intersection(expressed_genes_sender)
        ligand_activities_focused = dict(
            (key, val) for key, val in ligand_activities.items() if key in potential_ligands_focused
        )
        ligand_activities_sorted_focused = sorted(
            ligand_activities_focused.items(),
            key=lambda x : x[1]["aupr_corrected"],
            reverse=True
        )
        output["ligand_activities_sorted_focused"] = ligand_activities_sorted_focused
        best_upstream_ligands_focused = [e[0] for e in ligand_activities_sorted_focused[:ligands_top_n]]
        output["best_upstream_ligands_focused"] = best_upstream_ligands_focused
        if get_ltl:
            output["active_ligand_target_links_focused"] = combine_weighted_ligand_target_links((
                predictor.get_weighted_ligand_target_links(ligand, geneset, n=targets_top_n)
                for ligand in best_upstream_ligands_focused
            ))
        if lr_sig is not None:
            output["ligand_receptor_links_focused"] = get_weighted_ligand_receptor_links(
                best_upstream_ligands_focused,
                expressed_receptors,
                lr_network,
                lr_sig
            )
        ann_focused = subset_ann(ann, sender_celltypes, layers=[layer])
        ann_focused.var = ann.var
        ann_focused.X = ann_focused.layers[layer]
        output["ann_focused"] = ann_focused
        if get_lfc:
            output["lfcs"] = [
                get_lfc_celltype(
                    ann,
                    celltype,
                    condition_col=condition_col,
                    condition_oi=condition_oi,
                    condition_ref=condition_ref,
                    layer=layer,
                    celltype_col=celltype_col,
                    features=best_upstream_ligands_focused
                )
                for celltype in sender_celltypes
            ]
    if get_prioritization_table:
        if sender_celltypes is None:
            return ValueError("sender_celltypes needs to be provided if get_prioritization_table is True")
        lr_network_filtered = lr_network.subset_sep(expressed_ligands, expressed_receptors)
        info_tables = generate_info_tables(
            ann,
            celltype_col,
            sender_celltypes,
            [receiver],
            lr_network_filtered,
            condition_col,
            condition_oi,
            condition_ref,
            case_control
        )
        output["prioritization_table"] = generate_prioritization_table(
            info_tables["sender_receiver_info"],
            info_tables["sender_receiver_de"],
            ligand_activities_sorted,
            info_tables["lr_condition_de"]
        )
    return output

def create_ligand_activity_hist(
    ligand_activities_sorted:Iterable[tuple],
    xtitle:str="ligand activity",
    ytitle:str="# ligands",
    figsize:tuple[float, float]=(6, 6)
):
    '''
    Creates a ligand activity histogram. 

    Parameters
    ----------
    ligand_activities_sorted : Iterable of str
        the computed metrics for each ligand
    xtitle : str
        the title of the x-axis
    ytitle : str
        the title of the y-axis
    figsize : tuple of float
        the size of the figure
    '''
    plt.subplots(figsize=figsize)
    vals = [e[1]["aupr_corrected"] for e in ligand_activities_sorted]
    plt.hist(vals, bins=40, edgecolor="black")
    plt.vlines(x=vals[29], ymin=0, ymax=120, color="red", linestyles="dashed")
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()

def create_ligand_activity_heatmap(
    ligand_activities_sorted:Iterable[tuple],
    title:str="ligand_activity",
    cbar_label:str="AUPR",
    cmap:str="YlOrRd",
    figsize:tuple[float, float]=(6, 6)
):
    '''
    Creates a ligand activity heatmap. 

    Parameters
    ----------
    ligand_activities_sorted : Iterable of tuple
        the computed metrics for each ligand
    title : str
        the title of the plot
    cbar_label : str
        the label of the color bar
    cmap : str
        the color map
    figsize : tuple of float
        the size of the figure
    '''
    ligands, metrics = zip(*ligand_activities_sorted)
    _, ax = heatmap_1d(
        [e["aupr_corrected"] for e in metrics],
        labels=ligands,
        title=title,
        cbar_label=cbar_label,
        cmap=cmap,
        figsize=figsize
    )
    ax.invert_yaxis()
    plt.show()

def create_regulatory_potential_heatmap(
    predictor:LigandActivityPredictor,
    active_ligand_target_links:list[tuple[str, str, float]],
    xtitle="predicted target genes",
    ytitle="prioritized ligands",
    cbar_label="regulatory potential",
    cmap="Blues",
    figsize:tuple[float, float]=(6, 6)
):
    '''
    Creates a regulatory potential heatmap. 

    Parameters
    ----------
    predictor : LigandActivityPredictor
        the predictor which contains the ligand-target matrix
    active_ligand_target_links : list of tuple
        list of (ligand, target, weight) tuples representing the ligand-target links
    xtitle : str
        the title of the x-axis
    ytitle : str
        the title of the y-axis
    cbar_label : str
        the label of the color bar
    cmap : str
        the color map
    figsize : tuple of float
        the size of the figure
    '''
    ligand_target_vis, targets, ligands = prepare_ligand_target_visualization(
        predictor,
        active_ligand_target_links,
        cutoff=0.33
    )
    heatmap_2d(
        ligand_target_vis.transpose(),
        xlabels=targets,
        ylabels=ligands,
        xtitle=xtitle,
        ytitle=ytitle,
        cbar_label=cbar_label,
        cmap=cmap,
        figsize=figsize
    )
    plt.show()

def create_prior_interaction_potential_heatmap(
    ligand_receptor_links:WeightedNetwork,
    xtitle="receptors",
    ytitle="ligands",
    cbar_label="prior interaction potential",
    cmap="Oranges",
    figsize:tuple[float, float]=(6, 6)
):
    '''
    Creates a prior interaction potential heatmap. 

    Parameters
    ----------
    ligand_receptor_links : WeightedNetwork
        the weighted ligand-receptor links in the sender-agnostic approach
    xtitle : str
        the title of the x-axis
    ytitle : str
        the title of the y-axis
    cbar_label : str
        the label of the color bar
    cmap : str
        the color map
    figsize : tuple of float
        the size of the figure
    '''
    mat, ligands, receptors = prepare_ligand_receptor_visualization(ligand_receptor_links)
    heatmap_2d(
        mat,
        xlabels=receptors,
        ylabels=ligands,
        xtitle=xtitle,
        ytitle=ytitle,
        cbar_label=cbar_label,
        cmap=cmap,
        figsize=figsize
    )
    plt.show()

def create_lfc_heatmap(
    sender_celltypes:list[str],
    ligand_activities:dict[str, dict[str, float]],
    lfcs:list[tuple[list[str], list[float]]],
    xtitle="cell types",
    ytitle="prioritized ligands",
    cbar_label="LFC",
    cmap="seismic",
    figsize:tuple[float, float]=(6, 6)
):
    '''
    Creates an LFC heatmap. 

    Parameters
    ----------
    sender_celltypes : list of str
        the sender cell types
    ligand_activities : dict
        the computed metrics for each ligand
    lfcs : list of tuple
        the log fold changes as a list of tuples of lists where the first list of each tuple contains the ligands and second list contains the values
    xtitle : str
        the title of the x-axis
    ytitle : str
        the title of the y-axis
    cbar_label : str
        the label of the color bar
    cmap : str
        the color map
    figsize : tuple of float
        the size of the figure
    '''
    lfcs = combine_by_key(*lfcs)
    # sort by ligand activity
    ligands, vals = zip(*(
        (ligand, metrics_vals[1])
        for ligand, metrics_vals in
        sorted(
            combine_dicts(ligand_activities, lfcs).items(),
            key=lambda x : x[1][0]["aupr_corrected"],
            reverse=True
        )
    ))
    _, ax = heatmap_2d(
        np.vstack(vals),
        xlabels=sender_celltypes,
        ylabels=ligands,
        xtitle=xtitle,
        ytitle=ytitle,
        cbar_label=cbar_label,
        cbar_position="right",
        cbar_orientation="vertical",
        cmap=cmap,
        figsize=figsize
    )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    plt.show()

def generate_info_tables(
    ann:AnnData,
    celltype_col:str,
    senders_oi:list[str],
    receivers_oi:list[str],
    lr_network_filtered:LigandReceptorNetwork,
    condition_col:str,
    condition_oi:str,
    condition_ref:str,#TODO
    case_control:bool=False
) -> dict[str, pd.DataFrame]:
    '''
    Calculate differential expression, average expression, and condition specificity of ligands and receptors. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    celltype_col : str
        the column in ann.obs which contains the celltypes
    senders_oi : list of str
        the sender celltypes of interest
    receivers_oi : list of str
        the receiver celltypes of interest
    lr_network_filtered : LigandReceptorNetwork
        the filtered ligand-receptor network
    condition_col : str
        the column in ann.obs which contains the conditions
    condition_oi : str
        the condition of interest
    condition_ref : str
        the reference condition
    case_control : bool
        if True, calculate condition specificity, else only calculate cell type specificity.
    
    Returns
    -------
    dict
        dictionary containing the three dataframes:
            "sender_receiver_de",
            "sender_receiver_info",
            "group_DE"
    '''
    output = {
        "sender_receiver_de": process_table_to_ic(
            calculate_de(
                ann,
                celltype_col,
                condition_oi,
                condition_col,
                features=lr_network_filtered.get_ligands().union(lr_network_filtered.get_receptors())
            ),
            "celltype_DE",
            lr_network_filtered,
            senders_oi,
            receivers_oi
        ),
        "sender_receiver_info": process_table_to_ic(
            get_avg_exp(
                ann,
                celltype_col,
                condition_oi,
                condition_col
            ),
            "expression",
            lr_network_filtered
        )
    }
    if case_control:
        group_metrics(
            ann,
            groupby=condition_col,
            group_oi=condition_oi,
            group_ref=condition_ref
        )
        res = ann.uns["group_metrics"]
        output["lr_condition_de"] = process_table_to_ic(
            res[["gene", "lfc", "pval", "pval_adj"]],
            "group_DE",
            lr_network_filtered
        )
    return output
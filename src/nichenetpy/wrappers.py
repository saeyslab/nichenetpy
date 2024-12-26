from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.utils import (
    combine_by_key,
    combine_dicts
)
from nichenetpy.extraction import (
    get_expressed_genes,
    subset_ann_celltype,
    get_weighted_ligand_receptor_links,
    get_lfc_celltype
)
from nichenetpy.visualization import (
    prepare_ligand_target_visualization,
    prepare_ligand_receptor_visualization,
    heatmap_2d,
    heatmap_1d
)

from itertools import cycle, chain
from collections.abc import Iterable
from anndata import AnnData

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


def get_geneset_oi(
    ann:AnnData,
    receiver:str,
    condition_oi:str,
    condition_ref:str,
    layer:str="data",
    gene_field:str="gene",
    condition_col:str="aggregate",
    method:str="wilcoxon",
    max_pval_adj:float=0.05,
    min_log2FC:float=0.25
) -> list[str]:
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
    gene_field : str
        the name of the column in ann.var which contains the gene symbols
    condition_col : str
        the name of the column in obs which contains the conditions
    method : str
        the method to use in rank_genes_groups
    max_pval_adj : float
        the upper bound for pval_adj
    min_log2FC : float
        te lower bound for log2FC
    
    Returns
    -------
    list
        the geneset of interest
    '''
    ann_receiver = subset_ann_celltype(ann, receiver, layers=[layer])
    ann_receiver.var_names = ann.var[gene_field]
    sc.pp.log1p(ann_receiver, layer=layer)
    sc.tl.rank_genes_groups(
        ann_receiver,
        groupby=condition_col,
        method=method,
        layer=layer,
        groups=[condition_oi], 
        reference=condition_ref
    )
    return [
        gene for gene, pval_adj, log2FC in
        zip(
            [e[0] for e in ann_receiver.uns["rank_genes_groups"]["names"]],
            [e[0] for e in ann_receiver.uns["rank_genes_groups"]["pvals_adj"]],
            [e[0] for e in ann_receiver.uns["rank_genes_groups"]["logfoldchanges"]]
        ) if pval_adj <= max_pval_adj and abs(log2FC) >= min_log2FC
    ]

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
    lr_sig:WeightedNetwork,
    receiver:str,
    condition_oi:str,
    condition_ref:str,
    sender_celltypes:Iterable[str]=None,
    get_expressed_genes_pct:float=0.05,
    layer:str="data",
    gene_field:str="gene",
    condition_col:str="aggregate",
    rank_method:str="wilcoxon",
    max_pval_adj:float=0.05,
    min_log2FC:float=0.25,
    ligands_top_n:int=30
):
    expressed_genes_receiver = set(get_expressed_genes(receiver, ann, pct=get_expressed_genes_pct))
    all_receptors = lr_network.get_receptors()
    expressed_receptors = all_receptors.intersection(expressed_genes_receiver)
    potential_ligands = set(
        key for key, group in lr_network.item_iter()
        if len(group.intersection(expressed_receptors)) > 0
    )
    geneset = get_geneset_oi(
        ann,
        receiver,
        condition_oi,
        condition_ref,
        layer,
        gene_field,
        condition_col,
        rank_method,
        max_pval_adj,
        min_log2FC
    )
    ligand_activities = predictor.predict_ligand_activities(
        geneset=geneset,
        background_expressed_genes=expressed_genes_receiver,
        potential_ligands=potential_ligands
    )
    ligand_activities_sorted = sorted(ligand_activities.items(), key=lambda x : x[1]["aupr_corrected"], reverse=True)
    best_upstream_ligands = [e[0] for e in ligand_activities_sorted[:ligands_top_n]]
    active_ligand_target_links = combine_weighted_ligand_target_links((
        predictor.get_weighted_ligand_target_links(ligand, geneset, n=100)
        for ligand in best_upstream_ligands
    ))
    ligand_receptor_links = get_weighted_ligand_receptor_links(
        best_upstream_ligands,
        expressed_receptors,
        lr_network,
        lr_sig
    )
    if sender_celltypes is not None:
        list_expressed_genes_sender = [get_expressed_genes(ct, ann, pct=get_expressed_genes_pct) for ct in sender_celltypes]
        expressed_genes_sender = set(e for l in list_expressed_genes_sender for e in l)
        potential_ligands_focused = potential_ligands.intersection(expressed_genes_sender)
        ligand_activities_focused = dict(
            (key, val) for key, val in ligand_activities.items() if key in potential_ligands_focused
        )
        ligand_activities_sorted_focused = sorted(
            ligand_activities_focused.items(),
            key=lambda x : x[1]["aupr_corrected"],
            reverse=True
        )
        best_upstream_ligands_focused = [e[0] for e in ligand_activities_sorted_focused[:ligands_top_n]]
        active_ligand_target_links_focused = combine_weighted_ligand_target_links((
            predictor.get_weighted_ligand_target_links(ligand, geneset, n=100)
            for ligand in best_upstream_ligands_focused
        ))
        ligand_receptor_links_focused = get_weighted_ligand_receptor_links(
            best_upstream_ligands_focused,
            expressed_receptors,
            lr_network,
            lr_sig
        )
        ann_focused = subset_ann_celltype(ann, sender_celltypes, layers=[layer])
        ann_focused.var = ann.var
        ann_focused.X = ann_focused.layers[layer]
        lfcs = [
            get_lfc_celltype(
                ann,
                celltype,
                "aggregate",
                condition_oi="LCMV",
                condition_ref="SS",
                layer="data",
                features=best_upstream_ligands_focused
            )
            for celltype in sender_celltypes
        ]
        return {
            "best_upstream_ligands": best_upstream_ligands,
            "ligand_activities_sorted": ligand_activities_sorted,
            "active_ligand_target_links": active_ligand_target_links,
            "ligand_receptor_links": ligand_receptor_links,
            "best_upstream_ligands_focused": best_upstream_ligands_focused,
            "ligand_activities_sorted_focused": ligand_activities_sorted_focused,
            "active_ligand_target_links_focused": active_ligand_target_links_focused,
            "ligand_receptor_links_focused": ligand_receptor_links_focused,
            "ann_focused": ann_focused,
            "lfcs": lfcs
        }
    else:
        return {
            "best_upstream_ligands": best_upstream_ligands,
            "ligand_activities_sorted": ligand_activities_sorted,
            "active_ligand_target_links": active_ligand_target_links,
            "ligand_receptor_links": active_ligand_target_links
        }

def create_ligand_activity_hist(
    ligand_activities_sorted:Iterable[str],
    figsize:tuple[float, float]=(6, 6)
):
    plt.subplots(figsize=figsize)
    vals = [e[1]["aupr_corrected"] for e in ligand_activities_sorted]
    plt.hist(vals, bins=40, edgecolor="black")
    plt.vlines(x=vals[29], ymin=0, ymax=120, color="red", linestyles="dashed")
    plt.xlabel("ligand activity")
    plt.ylabel("# ligands")
    plt.show()

def create_ligand_activity_heatmap(
    ligand_activities_sorted:Iterable[str],
    figsize:tuple[float, float]=(6, 6)
):
    ligands, metrics = zip(*ligand_activities_sorted)
    _, ax = heatmap_1d(
        [e["aupr_corrected"] for e in metrics],
        labels=ligands,
        title="ligand activity",
        cbar_label="AUPR",
        cmap="YlOrRd",
        figsize=figsize
    )
    ax.invert_yaxis()
    plt.show()

def create_regulatory_potential_heatmap(
    predictor:LigandActivityPredictor,
    active_ligand_target_links:list[tuple[str, str, float]],
    figsize:tuple[float, float]=(6, 6)
):
    ligand_target_vis, targets, ligands = prepare_ligand_target_visualization(
        predictor,
        active_ligand_target_links,
        cutoff=0.33
    )
    heatmap_2d(
        ligand_target_vis.transpose(),
        xlabels=targets,
        ylabels=ligands,
        xtitle="predicted target genes",
        ytitle="prioritized ligands",
        cbar_label="regulatory potential",
        cmap="Blues",
        figsize=figsize
    )
    plt.show()

def create_prior_interaction_potential_heatmap(
    ligand_receptor_links:WeightedNetwork,
    figsize:tuple[float, float]=(6, 6)
):
    mat, ligands, receptors = prepare_ligand_receptor_visualization(ligand_receptor_links)
    heatmap_2d(
        mat,
        xlabels=receptors,
        ylabels=ligands,
        xtitle="receptors",
        ytitle="ligands",
        cbar_label="prior interaction potential",
        cmap="Oranges",
        figsize=figsize
    )
    plt.show()

def create_lfc_heatmap(
    sender_celltypes:list[str],
    ligand_activities:dict[str, dict[str, float]],
    lfcs:list[tuple[list[str], list[float]]],
    figsize:tuple[float, float]=(6, 6)
):
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
        xtitle="cell types",
        ytitle="prioritized ligands",
        cbar_label="LFC",
        cbar_position="right",
        cbar_orientation="vertical",
        cmap="seismic",
        figsize=figsize
    )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    plt.show()
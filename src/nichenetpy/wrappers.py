from collections.abc import Callable, Iterable, Sequence
from itertools import chain, cycle, repeat
from numbers import Number
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pycirclize import Circos
from scipy.stats import fisher_exact

from nichenetpy.ann_utils import subset_ann
from nichenetpy.extraction import get_expressed_genes, get_lfc_celltype, get_weighted_ligand_receptor_links
from nichenetpy.metrics import group_metrics
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.normalization import scaling_modified_zscore
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.prioritization import calculate_de, generate_prioritization_table, get_avg_exp, process_table_to_ic
from nichenetpy.typing import GeneSymbol, nichenet_matrix, target_eval_metric_t
from nichenetpy.utils import combine_by_key, combine_dicts, df_grouped_apply, rank_genes_groups_to_dataframe
from nichenetpy.visualization import (
    heatmap_1d,
    heatmap_2d,
    prepare_ligand_receptor_visualization,
    prepare_ligand_target_visualization,
)


def get_geneset_oi(
    ann: AnnData,
    receiver: str,
    condition_col: str,
    condition_oi: str,
    layer: str = "data",
    max_pval_adj: float = 0.05,
    min_abs_lfc: float = 0.25,
    min_pct: float = 0.05,
    use_scanpy: bool = False,
    scanpy_corr_method: Literal["bonferroni"] | Literal["benjamini-hochberg"] = "benjamini-hochberg",
    scanpy_tie_correct: bool = False,
    celltype_col: str = "celltype",
    lfc_denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None = np.expm1,  # pyright: ignore[reportArgumentType] # this works fine
) -> set[str]:
    """
    Gets the geneset of interest from an AnnData object. The gene set of interest are genes within the receiver cell type that are likely to be influenced by ligands from the CCC event.

    Parameters
    ----------
    ann : AnnData
        the AnnData object to extract expressed genes from
    receiver : str
        the receiver cell type
    condition_col : str
        the name of the column in obs which contains the conditions
    condition_oi : str
        the condition of interest
    layer : str
        the name of the layer which contains the data matrix
    max_pval_adj : float
        the upper bound for pval_adj
    min_abs_lfc : float
        the lower bound for the absolute log fold change
    min_pct : float
        genes are only considered if they're expressed in at least this fraction of the cells of the condition of
        interest or of the other conditions (with use_scanpy: of the condition of interest)
    use_scanpy : bool
        if true, use scanpy.tl.rank_genes_groups to compute the metrics
    scanpy_corr_method : str
        corr_method argument passed to scanpy.tl.rank_genes_groups
    scanpy_tie_correct : bool
        tie_correct argument passed to scanpy.tl.rank_genes_groups
    celltype_col : str
        the column in the anndata object which contains the celltype
    lfc_denormalize : Callable or None
        a denormalization function to apply prior to the calculation of the log fold changes

        only used with use_scanpy=False

    Returns
    -------
    set
        the geneset of interest

    Notes
    -----
    With use_scanpy=True, scanpy_corr_method="bonferroni" and scanpy_tie_correct=True
    only the log fold changes will be different compared to use_scanpy=False
    """
    ann_receiver = subset_ann(ann, val=receiver, layers=[layer], val_col=celltype_col)
    if use_scanpy:  # lfc will be different
        sc.tl.rank_genes_groups(
            ann_receiver,
            groupby=condition_col,
            layer=layer,
            pts=True,
            method="wilcoxon",
            corr_method=scanpy_corr_method,
            tie_correct=scanpy_tie_correct,
        )
        DE_table = rank_genes_groups_to_dataframe(ann_receiver, condition_col)
        DE_table = DE_table[
            (DE_table["pval_adj"] <= max_pval_adj)
            & (abs(DE_table["lfc"]) >= min_abs_lfc)
            & (DE_table["pct"] >= min_pct)
        ]
    else:
        group_metrics(
            ann_receiver,
            groupby=condition_col,
            layer=layer,
            min_pct=min_pct,
            min_abs_lfc=min_abs_lfc,
            lfc_denormalize=lfc_denormalize,
        )
        DE_table = ann_receiver.uns["group_metrics"]
    return set(DE_table[(DE_table[condition_col] == condition_oi) & (DE_table["pval_adj"] <= max_pval_adj)]["gene"])


def combine_weighted_ligand_target_links(
    active_ligand_target_links: Iterable[dict],
) -> list[tuple[GeneSymbol, GeneSymbol, float]]:
    """
    Combines weighted ligand-target links of different ligands.

    Parameters
    ----------
    active_ligand_target_links : list
        list of ligand-target links as returned by LigandActivityPredictor.get_weighted_ligand_target_links

    Returns
    -------
    list
        list of (ligand, target, weight) tuples representing the combined ligand-target links
    """
    return list(
        chain(
            *(
                zip(cycle([e["ligand"]]), e["target"], e["weight"], strict=False)
                for e in active_ligand_target_links
                if e["target"] is not None
            )
        )
    )


def run_nichenet(
    ann: AnnData,
    predictor: LigandActivityPredictor[str],
    lr_network: LigandReceptorNetwork[str],
    receiver: str,
    condition_col: str,
    condition_oi: str,
    condition_ref: str,
    sender_celltypes: Iterable[str] | None = None,
    layer: str = "data",
    celltype_col: str = "celltype",
    max_pval_adj: float = 0.05,
    min_abs_lfc: float = 0.25,
    expression_pct: float = 0.1,
    ligands_top_n: int = 30,
    targets_top_n: int = 200,
    lr_sig: WeightedNetwork[str] | None = None,
    get_ltl: bool = False,
    get_lfc: bool = False,
    get_prioritization_table: bool = False,
    case_control: bool = True,
    use_scanpy: bool = False,
    lfc_denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None = np.expm1,  # pyright: ignore[reportArgumentType] # this works fine
    target_evaluation_metric: target_eval_metric_t = "aupr_corrected",
) -> dict[str, Any]:
    """
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
    condition_col : str
        the name of the column in obs which contains the conditions
    condition_oi : str
        the condition of interest
    condition_ref : str
        the reference condition
    sender_celltypes : Iterable[str] or None
        the sender cell types for the sender-focused approach, if None only the sender-agnostic analysis is performed
    layer : str
        the layer in the AnnData object which contains the data matrix
    celltype_col : str
        the name of the column in obs which contains the celltypes
    max_pval_adj : float
        the upper bound for pval_adj
    min_abs_lfc : float
        the lower bound for the absolute log fold change of the genes of interest
    expression_pct : float
        the minimum fraction of the cells of a cell type that must express a gene for it to be considered expressed
        (receiver and sender genes); also used as `min_pct` for the differential expression: genes are only tested if
        they're expressed in at least this fraction of the cells of one of the compared groups
    ligands_top_n : int
        the amount of ligands that are considered to be the best upstream ligands
    targets_top_n : int
        the number of target genes to consider per ligand when performing the target gene inference
    lr_sig : WeightedNetwork or None
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
    use_scanpy : bool
        if true, use scanpy.tl.rank_genes_groups to compute the metrics (lfc, p-values, pct)
    lfc_denormalize : Callable or None
        a denormalization function to apply prior to the calculation of the log fold changes
    target_evaluation_metric : string
        the target prediction evaluation metric to compute (ligands are ranked based on this metric)

    Returns
    -------
    dict
        a dictionary which contains the output of the analysis, it contains the following objects (depending on certain arguments)

        for the sender-agnostic approach:

            best_upstream_ligands : list of gene_t
                the top scoring ligands in the sender-agnostic approach
            ligand_activities_sorted : dict
                the computed metrics for each ligand in the sender-agnostic approach
            active_ligand_target_links : list of tuple
                list of (ligand, target, weight) tuples representing the ligand-target
                links in the sender-agnostic approach
            ligand_receptor_links : WeightedNetwork
                the weighted ligand-receptor links in the sender-agnostic approach
            expressed_receptors : set of gene_t
                the expressed receptors
            expressed_genes_receiver : set of gene_t
                expressed genes in the receiver
            background_expressed_genes : set of gene_t
                the background expressed genes
            prioritization_table : pandas.DataFrame
                Data frame of prioritized sender-ligand-receiver-receptor interactions
            geneset_oi : set of gene_t
                the geneset of interest

        and the following additional objects for the sender-focused approach:

            best_upstream_ligands_focused : list of gene_t
                the top scoring ligands in the sender-focused approach
            ligand_activities_sorted_focused : dict
                the computed metrics for each ligand in the sender-focused approach
            active_ligand_target_links_focused : list of tuple
                list of (ligand, target, weight) tuples representing the ligand-target links
                in the sender-focused approach
            ligand_receptor_links_focused : WeightedNetwork
                the weighted ligand-receptor links in the sender-focused approach
            ann_focused : AnnData
                the AnnData object used in the sender-focused approach (new object derived from ann)
            lfcs : list of tuple
                the log fold changes as a list of tuples of lists where the first list of each tuple contains
                the ligands and second list contains the values
            expressed_ligands : set of gene_t
                the expressed ligands

        if get_prioritization_table is True, additionally

            prioritization_table : pandas.DataFrame

    Raises
    ------
    ValueError
        if get_prioritization_table is True and sender_celltypes is not provided
    """
    output = {}
    expressed_genes_receiver = set(
        get_expressed_genes(receiver, ann, pct=expression_pct, celltype_col=celltype_col, layer=layer)
    )
    output["expressed_genes_receiver"] = expressed_genes_receiver
    expressed_receptors = lr_network.get_receptors().intersection(expressed_genes_receiver)
    output["expressed_receptors"] = expressed_receptors
    background_expressed_genes = predictor.get_genes().intersection(expressed_genes_receiver)
    output["background_expressed_genes"] = background_expressed_genes
    potential_ligands = {
        key for key, group in lr_network.item_iter() if len(group.intersection(expressed_receptors)) > 0
    }
    geneset = get_geneset_oi(
        ann,
        receiver=receiver,
        condition_oi=condition_oi,
        layer=layer,
        condition_col=condition_col,
        max_pval_adj=max_pval_adj,
        min_abs_lfc=min_abs_lfc,
        min_pct=expression_pct,
        use_scanpy=use_scanpy,
        celltype_col=celltype_col,
        lfc_denormalize=lfc_denormalize,
    )
    geneset.intersection_update(predictor.get_genes())
    output["geneset_oi"] = geneset
    background_expressed_genes = predictor.get_genes().intersection(expressed_genes_receiver)
    ligand_activities = predictor.predict_ligand_activities(
        geneset=geneset,
        background_expressed_genes=background_expressed_genes,
        potential_ligands=potential_ligands,
        target_evaluation_metrics=(target_evaluation_metric,),
    )
    ligand_activities_sorted = sorted(ligand_activities.items(), key=lambda x: (-x[1][target_evaluation_metric], x[0]))
    output["ligand_activities_sorted"] = ligand_activities_sorted
    best_upstream_ligands = [e[0] for e in ligand_activities_sorted[:ligands_top_n]]
    output["best_upstream_ligands"] = best_upstream_ligands
    if get_ltl:
        output["active_ligand_target_links"] = combine_weighted_ligand_target_links(
            predictor.get_weighted_ligand_target_links(ligand, geneset, n=targets_top_n)
            for ligand in best_upstream_ligands
        )
    if lr_sig is not None:
        output["ligand_receptor_links"] = get_weighted_ligand_receptor_links(
            best_upstream_ligands, expressed_receptors, lr_network, lr_sig
        )
    if sender_celltypes is not None:
        expressed_genes_sender = {
            gene
            for ct in sender_celltypes
            for gene in get_expressed_genes(ct, ann, pct=expression_pct, celltype_col=celltype_col, layer=layer)
        }
        expressed_ligands = lr_network.get_ligands().intersection(expressed_genes_sender)
        output["expressed_ligands"] = expressed_ligands
        potential_ligands_focused = potential_ligands.intersection(expressed_genes_sender)
        ligand_activities_focused = {
            key: val for key, val in ligand_activities.items() if key in potential_ligands_focused
        }
        ligand_activities_sorted_focused = sorted(
            ligand_activities_focused.items(), key=lambda x: (-x[1][target_evaluation_metric], x[0])
        )
        output["ligand_activities_sorted_focused"] = ligand_activities_sorted_focused
        best_upstream_ligands_focused = [e[0] for e in ligand_activities_sorted_focused[:ligands_top_n]]
        output["best_upstream_ligands_focused"] = best_upstream_ligands_focused
        if get_ltl:
            output["active_ligand_target_links_focused"] = combine_weighted_ligand_target_links(
                predictor.get_weighted_ligand_target_links(ligand, geneset, n=targets_top_n)
                for ligand in best_upstream_ligands_focused
            )
        if lr_sig is not None:
            output["ligand_receptor_links_focused"] = get_weighted_ligand_receptor_links(
                best_upstream_ligands_focused, expressed_receptors, lr_network, lr_sig
            )
        ann_focused = subset_ann(ann, val=sender_celltypes, layers=[layer], val_col=celltype_col)
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
                    features=best_upstream_ligands_focused,
                )
                for celltype in sender_celltypes
            ]
    if get_prioritization_table:
        if sender_celltypes is None:
            raise ValueError("sender_celltypes needs to be provided if get_prioritization_table is True")
        if not isinstance(sender_celltypes, Sequence):
            sender_celltypes = list(sender_celltypes)
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
            case_control,
            use_scanpy=use_scanpy,
        )
        output["prioritization_table"] = generate_prioritization_table(
            info_tables["sender_receiver_info"],
            info_tables["sender_receiver_de"],
            ligand_activities_sorted,  # type: ignore # false positive
            info_tables["lr_condition_de"] if case_control else None,
            target_evaluation_metric=target_evaluation_metric,
        )
    return output


def run_nichenet_cluster_de(
    ann: AnnData,
    predictor: LigandActivityPredictor[str],
    lr_network: LigandReceptorNetwork[str],
    receiver_affected: str,
    receiver_reference: str,
    sender_celltypes: Iterable[str] | None = None,
    layer: str = "data",
    celltype_col: str = "celltype",
    max_pval_adj: float = 0.05,
    lfc_cutoff: float = 0.25,
    expression_pct: float = 0.1,
    ligands_top_n: int | None = 30,
    targets_top_n: int = 200,
    lfc_denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None = np.expm1,  # pyright: ignore[reportArgumentType] # this works fine
    geneset: Literal["DE"] | Literal["up"] | Literal["down"] = "DE",
    target_evaluation_metric: target_eval_metric_t = "aupr_corrected",
) -> dict[str, Any]:
    """
    Runs a standard nichenet analysis.

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    predictor : LigandActivityPredictor
        the predictor which contains the ligand-target matrix
    lr_network : LigandReceptorNetwork
        the ligand-receptor network containing the ligand-receptor interactions
    receiver_affected : str
        Name of cluster identity/identities of "affected" cells that were presumably affected by intercellular communication with other cells
    receiver_reference : str
        Name of cluster identity/identities of "steady-state" cells, before they are affected by intercellular communication with other cells
    sender_celltypes : Iterable[str] or None
        The sender celltypes or None for all celltypes
    layer : str
        the layer in the AnnData object which contains the data matrix
    celltype_col : str
        the name of the column in obs which contains the celltypes
    max_pval_adj : float
        the upper bound for pval_adj
    lfc_cutoff: float
        the cutoff for the logfold change
    expression_pct : float
        the minimum fraction of the cells of a cell type that must express a gene for it to be considered expressed
        (receiver and sender genes); also used as `min_pct` for the differential expression: genes are only tested if
        they're expressed in at least this fraction of the cells of one of the compared groups
    ligands_top_n : int
        the amount of ligands that are considered to be the best upstream ligands
    targets_top_n : int
        the number of target genes to consider per ligand when performing the target gene inference
    lfc_denormalize : Callable or None
        a denormalization function to apply prior to the calculation of the log fold changes
    geneset : str
        geneset Indicate whether to consider all DE genes between condition 1 and 2 ("DE"), or only genes upregulated in condition 1 ("up"), or only genes downregulad in condition 1 ("down").
    target_evaluation_metric : string
        the target prediction evaluation metric to compute (ligands are ranked based on this metric)

    Returns
    -------
    dict
        a dictionary which contains the output of the analysis, it contains the following objects

        for the sender-agnostic approach:

            best_upstream_ligands : list of gene_t
                the top scoring ligands in the sender-agnostic approach
            ligand_activities : dict
                the computed metrics for each ligand
            active_ligand_target_links : list of tuple
                list of (ligand, target, weight) tuples representing the ligand-target
                links in the sender-agnostic approach
            expressed_genes_sender : set of gene_t
                expressed genes in the sender
            expressed_genes_receiver_ref : set of gene_t
                expressed genes in the reference receiver
            expressed_genes_receiver_aff : set of gene_t
                expressed genes in the affected receiver
            background_expressed_genes : set of gene_t
                the background expressed genes
            geneset_oi : set of gene_t
                the geneset of interest

    Raises
    ------
    ValueError
        if geneset is not "DE", "up" or "down"
    """
    if type(geneset) is not str:
        raise TypeError(f"geneset should have type str, was {type(geneset)}")
    elif geneset not in ("DE", "up", "down"):
        raise ValueError(f"geneset should be 'DE', 'up' or 'down', was {geneset}")
    expressed_genes_receiver_ref = set(
        get_expressed_genes(receiver_reference, ann, pct=expression_pct, celltype_col=celltype_col, layer=layer)
    )
    expressed_genes_receiver_aff = set(
        get_expressed_genes(receiver_affected, ann, pct=expression_pct, celltype_col=celltype_col, layer=layer)
    )
    expressed_genes_receiver = expressed_genes_receiver_ref.union(expressed_genes_receiver_aff)
    if sender_celltypes is None:
        expressed_genes_sender = set(ann.var_names).union(predictor.row_names, predictor.col_names)
    else:
        expressed_genes_sender = {
            gene
            for ct in sender_celltypes
            for gene in get_expressed_genes(ct, ann, pct=expression_pct, celltype_col=celltype_col, layer=layer)
        }
    ann_filtered = subset_ann(ann, val=(receiver_reference, receiver_affected), layers=[layer], val_col=celltype_col)
    if geneset == "DE":
        group_metrics(
            ann_filtered,
            groupby=celltype_col,
            layer=layer,
            min_pct=expression_pct,
            min_abs_lfc=lfc_cutoff,
            lfc_denormalize=lfc_denormalize,
        )
        DE_table = ann_filtered.uns["group_metrics"]
        geneset_oi = set(DE_table[(DE_table["pval_adj"] <= max_pval_adj)]["gene"])
    else:
        group_metrics(
            ann_filtered,
            groupby=celltype_col,
            group_oi=receiver_affected,
            group_ref=receiver_reference,
            layer=layer,
            min_pct=expression_pct,
            min_abs_lfc=0,
            lfc_denormalize=lfc_denormalize,
        )
        DE_table = ann_filtered.uns["group_metrics"]
        geneset_oi = set(
            DE_table[
                (DE_table["pval_adj"] <= max_pval_adj)
                & ((DE_table["lfc"] >= lfc_cutoff) if geneset == "up" else (DE_table["lfc"] <= -lfc_cutoff))
            ]["gene"]
        )
    geneset_oi.intersection_update(predictor.row_names)
    background_expressed_genes = predictor.get_genes().intersection(expressed_genes_receiver)
    expressed_ligands = lr_network.get_ligands().intersection(expressed_genes_sender)
    expressed_receptors = lr_network.get_receptors().intersection(expressed_genes_receiver)
    potential_ligands = {
        key
        for key, group in lr_network.item_iter()
        if key in expressed_ligands and len(group.intersection(expressed_receptors)) > 0
    }
    ligand_activities_dct = predictor.predict_ligand_activities(
        geneset=geneset_oi,
        background_expressed_genes=background_expressed_genes,
        potential_ligands=potential_ligands,
        target_evaluation_metrics=(target_evaluation_metric,),
    )
    ligand_activities = sorted(ligand_activities_dct.items(), key=lambda x: (-x[1][target_evaluation_metric], x[0]))
    best_upstream_ligands = [
        e[0] for e in (ligand_activities if ligands_top_n is None else ligand_activities[:ligands_top_n])
    ]
    active_ligand_target_links = combine_weighted_ligand_target_links(
        predictor.get_weighted_ligand_target_links(ligand, geneset_oi, n=targets_top_n)
        for ligand in best_upstream_ligands
    )
    return {
        "expressed_genes_receiver_ref": expressed_genes_receiver_ref,
        "expressed_genes_receiver_aff": expressed_genes_receiver_aff,
        "expressed_genes_sender": expressed_genes_sender,
        "geneset_oi": geneset_oi,
        "background_expressed_genes": background_expressed_genes,
        "ligand_activities": ligand_activities_dct,
        "best_upstream_ligands": best_upstream_ligands,
        "active_ligand_target_links": active_ligand_target_links,
    }


def create_ligand_activity_hist(
    ligand_activities_sorted: Iterable[tuple],
    xtitle: str = "ligand activity",
    ytitle: str = "# ligands",
    figsize: tuple[float, float] = (6, 6),
    target_evaluation_metric: target_eval_metric_t = "aupr_corrected",
    top_n: int = 30,
) -> tuple[Figure, Axes]:
    """
    Creates a ligand activity histogram with a dashed line at the activity of the top_n-th ligand.

    Parameters
    ----------
    ligand_activities_sorted : Iterable of tuple
        the computed metrics for each ligand, sorted from the most to the least active ligand
    xtitle : str
        the title of the x-axis
    ytitle : str
        the title of the y-axis
    figsize : tuple of float
        the size of the figure
    target_evaluation_metric : string
        the target prediction evaluation metric to use
    top_n : int
        the dashed line marks the activity of the top_n-th ligand (the last ligand if there are fewer)

    Returns
    -------
    Figure
        the figure
    Axes
        the axes

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if there are no ligands or top_n is smaller than 1
    """
    if type(top_n) is not int:
        raise TypeError(f"top_n should have type int, was {type(top_n)}")
    if top_n < 1:
        raise ValueError(f"top_n should be at least 1, was {top_n}")
    vals = [e[1][target_evaluation_metric] for e in ligand_activities_sorted]
    if len(vals) == 0:
        raise ValueError("ligand_activities_sorted is empty")
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=40, edgecolor="black")
    # axvline spans the full height of the plot (the line used to stop at a fixed count of 120)
    ax.axvline(x=vals[min(top_n, len(vals)) - 1], color="red", linestyle="dashed")
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    return (fig, ax)


def create_ligand_activity_heatmap(
    ligand_activities_sorted: Iterable[tuple[str, str]],
    title: str = "ligand activity",
    cbar_label: str = "AUPR",
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = (6, 6),
    target_evaluation_metric: target_eval_metric_t = "aupr_corrected",
) -> tuple[Figure, Axes]:
    """
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
    target_evaluation_metric : string
        the target prediction evaluation metric to use

    Returns
    -------
    Figure
        the figure
    Axes
        the axes
    """
    ligands, metrics = zip(*ligand_activities_sorted, strict=True)
    fig, ax = heatmap_1d(
        [e[target_evaluation_metric] for e in metrics],
        labels=ligands,
        title=title,
        cbar_label=cbar_label,
        cmap=cmap,
        figsize=figsize,
    )
    ax.invert_yaxis()
    return (fig, ax)


def create_regulatory_potential_heatmap(
    predictor: LigandActivityPredictor[str],
    active_ligand_target_links: Sequence[tuple[str, str, float]],
    xtitle="predicted target genes",
    ytitle="prioritized ligands",
    cbar_label="regulatory potential",
    cmap="Blues",
    figsize: tuple[float, float] = (6, 6),
) -> tuple[Figure, Axes]:
    """
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

    Returns
    -------
    Figure
        the figure
    Axes
        the axes
    """
    ligand_target_vis, targets, ligands = prepare_ligand_target_visualization(
        predictor, active_ligand_target_links, cutoff=0.33
    )
    return heatmap_2d(
        ligand_target_vis.transpose(),
        xlabels=targets,
        ylabels=ligands,
        xtitle=xtitle,
        ytitle=ytitle,
        cbar_label=cbar_label,
        cmap=cmap,
        figsize=figsize,
    )


def create_prior_interaction_potential_heatmap(
    ligand_receptor_links: WeightedNetwork[str],
    xtitle="receptors",
    ytitle="ligands",
    cbar_label="prior interaction potential",
    cmap="Oranges",
    figsize: tuple[float, float] = (6, 6),
) -> tuple[Figure, Axes]:
    """
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

    Returns
    -------
    Figure
        the figure
    Axes
        the axes
    """
    mat, ligands, receptors = prepare_ligand_receptor_visualization(ligand_receptor_links)
    return heatmap_2d(
        mat,
        xlabels=receptors,
        ylabels=ligands,
        xtitle=xtitle,
        ytitle=ytitle,
        cbar_label=cbar_label,
        cmap=cmap,
        figsize=figsize,
    )


def create_lfc_heatmap(
    sender_celltypes: list[str],
    ligand_activities: dict[GeneSymbol, dict[str, float]],
    lfcs: list[tuple[list[GeneSymbol], list[float]]],
    xtitle="cell types",
    ytitle="prioritized ligands",
    cbar_label="LFC",
    cmap="seismic",
    figsize: tuple[float, float] = (6, 6),
    target_evaluation_metric: target_eval_metric_t = "aupr_corrected",
) -> tuple[Figure, Axes]:
    """
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
    target_evaluation_metric : string
        the target prediction evaluation metric to use

    Returns
    -------
    Figure
        the figure
    Axes
        the axes
    """
    lfcs_dct = combine_by_key(*lfcs)
    # sort by ligand activity
    ligands, vals = zip(
        *(
            (ligand, metrics_vals[1])
            for ligand, metrics_vals in sorted(
                combine_dicts(ligand_activities, lfcs_dct).items(),
                key=lambda x: x[1][0][target_evaluation_metric],
                reverse=True,
            )
        ),
        strict=True,
    )
    fig, ax = heatmap_2d(
        np.vstack(vals),
        xlabels=sender_celltypes,
        ylabels=ligands,
        xtitle=xtitle,
        ytitle=ytitle,
        cbar_label=cbar_label,
        cbar_position="right",
        cbar_orientation="vertical",
        cmap=cmap,
        figsize=figsize,
    )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    return (fig, ax)


def _create_circos_plot(
    circos_links: Iterable[tuple[tuple[str, str], tuple[str, str]]],
    colors: dict[str, str],
    inter_space: float = 5,
    intra_space: float = 1,
    opacity: Iterable[float] | None = None,
    separate_sender_receiver: bool = True,
    sender_receiver_space: float = 0,
) -> tuple[Figure, Axes]:
    circos_links = list(circos_links)
    # opacity is used twice (its maximum and the value of each link), so it's stored in a list (it may be a generator)
    weights = [1.0] * len(circos_links) if opacity is None else list(opacity)
    if len(weights) != len(circos_links):
        raise ValueError(f"opacity should have a value for each of the {len(circos_links)} links, had {len(weights)}")
    max_weight = max(weights, default=1.0)

    # Each rectangle is identified by (role, group, gene) and each sector by (role, group), with role "send" or "rec"
    # (or "" if senders and receivers aren't separated). The names are kept as tuples, so groups and genes can contain
    # any character (they used to be joined with "_" and split again, which failed for names containing "_").
    def sector_name(role: str, group: str) -> str:
        return f"{role}_{group}" if role else group

    send_role, rec_role = ("send", "rec") if separate_sender_receiver else ("", "")
    link_count_out: dict[tuple[str, str, str], int] = {}
    link_count_in: dict[tuple[str, str, str], int] = {}
    links = []
    # count the links of each rectangle; the n-th link of a rectangle starts at position n - 1 within the rectangle
    for ((sender_group, sender), (receiver_group, receiver)), weight in zip(circos_links, weights, strict=True):
        key_out = (send_role, sender_group, sender)
        key_in = (rec_role, receiver_group, receiver)
        link_count_out[key_out] = link_count_out.get(key_out, 0) + 1
        link_count_in[key_in] = link_count_in.get(key_in, 0) + 1
        links.append((key_out, key_in, weight / max_weight, link_count_out[key_out], link_count_in[key_in]))
    # the genes of each sector (in order of appearance, a set had a different order in every Python session) and the
    # size of each sector (its links + the space between its genes)
    sector2group: dict[str, str] = {}
    sector_genes: dict[str, list[str]] = {}
    group_sizes: dict[str, float] = {}
    for (role, group, gene), count in chain(link_count_in.items(), link_count_out.items()):
        name = sector_name(role, group)
        sector2group[name] = group
        if name in group_sizes:
            sector_genes[name].append(gene)
            group_sizes[name] += count + intra_space
        else:
            sector_genes[name] = [gene]
            group_sizes[name] = count
    if separate_sender_receiver:
        # pycirclize doesn't let you order the sectors, they're drawn in the order of group_sizes:
        # the receiver sectors ("rec_...") first, then the sender sectors ("send_...")
        group_sizes = dict(sorted(group_sizes.items(), key=lambda x: x[0]))
        if sender_receiver_space > 0:
            # extra space at the start of the first receiver sector and the end of the last receiver sector
            rec_sector_count = len({group for _, group, _ in link_count_in})
            keys = list(group_sizes.keys())
            for i in (0, rec_sector_count - 1):
                group_sizes[keys[i]] += sender_receiver_space
    circos = Circos(
        sectors=group_sizes,
        space=(
            inter_space
            * (360 / (sum(group_sizes.values()) + (len(group_sizes) - 1) * inter_space + 2 * sender_receiver_space))
        ),
    )
    link_count = {
        (sector_name(role, group), gene): count
        for (role, group, gene), count in chain(link_count_in.items(), link_count_out.items())
    }
    rects: dict[str, dict[str, float]] = {}
    # create a track for each sector and add the rectangles
    start = sender_receiver_space if separate_sender_receiver else 0
    for sector in circos.sectors:
        name = sector.name
        sub_rects: dict[str, float] = {}
        rects[name] = sub_rects
        track = sector.add_track((95, 100))
        for gene in sector_genes[name]:
            end = start + link_count[(name, gene)]
            track.rect(start, end, fc=colors[sector2group[name]])
            sub_rects[gene] = start
            track.text(
                gene, (end + start) / 2, r=track.r_center + track.r_size + len(gene) + 4, size=6, orientation="vertical"
            )
            start = end + intra_space
        start = 0
    # link the rectangles
    for (s_role, s_group, sender), (r_role, r_group, receiver), weight, send_pos, rec_pos in links:
        sender_sector = sector_name(s_role, s_group)
        receiver_sector = sector_name(r_role, r_group)
        circos.link(
            (sender_sector, rects[sender_sector][sender] + send_pos - 1, rects[sender_sector][sender] + send_pos),
            (
                receiver_sector,
                rects[receiver_sector][receiver] + rec_pos - 1,
                rects[receiver_sector][receiver] + rec_pos,
            ),
            direction=1,
            color=colors[s_group],
            alpha=weight,
        )
    # add a legend to the plot
    fig = circos.plotfig()
    circos.ax.legend(
        handles=[Patch(color=color, label=celltype) for celltype, color in colors.items()],
        bbox_to_anchor=(0, 1.1),
        loc="right",
        ncols=1,
    )
    return (fig, circos.ax)


def create_ligand_receptor_links_prioritization_circos_plot(
    senders: Iterable[str],
    receivers: Iterable[str],
    ligands: Iterable[str],
    receptors: Iterable[str],
    colors: dict[str, str],
    inter_space: float = 5,
    intra_space: float = 1,
    opacity: Iterable[float] | None = None,
    sender_receiver_space: float = 0,
) -> tuple[Figure, Axes]:
    """
    Creates a circos plot showing the links between ligands and receptors.

    Parameters
    ----------
    senders : Iterable of str
        the sender celltypes
    receivers : Iterable of str
        the receiver celltypes
    ligands : Iterable of str
        the ligands
    receptors : Iterable of str
        the receptors
    colors : dict
        color mapping for the ligands, should include a 'General' and 'target' mapping as well
    inter_space : float
        space between celltypes
    intra_space : float
        space between genes
    opacity : Iterable of float or None
        the opacity of each link
    sender_receiver_space : float
        the extra space between sender and receiver sector groups

    Returns
    -------
    matplotlib.Figure
        the plotted figure
    matplotlib.Axes
        the axes

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if not isinstance(senders, Iterable):
        raise TypeError(f"senders should have type Iterable[str], was {type(senders)}")
    if not isinstance(receivers, Iterable):
        raise TypeError(f"receivers should have type Iterable[str], was {type(receivers)}")
    if not isinstance(ligands, Iterable):
        raise TypeError(f"ligands should have type Iterable[gene_t], was {type(ligands)}")
    if not isinstance(receptors, Iterable):
        raise TypeError(f"receptors should have type Iterable[gene_t], was {type(receptors)}")
    if type(colors) is not dict:
        raise TypeError(f"colors should have type dict[gene_t, str], was {type(colors)}")
    if not isinstance(inter_space, Number):
        raise TypeError(f"inter_space should have type float, was {type(inter_space)}")
    if not isinstance(intra_space, Number):
        raise TypeError(f"intra_space should have type float, was {type(intra_space)}")
    if opacity is not None and not isinstance(opacity, Iterable):
        raise TypeError(f"opacity should have type Iterable[float], was {type(opacity)}")
    return _create_circos_plot(
        zip(zip(senders, ligands, strict=True), zip(receivers, receptors, strict=True), strict=True),
        colors=colors,
        inter_space=inter_space,
        intra_space=intra_space,
        opacity=opacity,
        sender_receiver_space=sender_receiver_space,
    )


def create_ligand_links_circos_plot(
    circos_links: pd.DataFrame,
    colors: dict[str, str],
    dest_name: str,
    inter_space: float = 5,
    intra_space: float = 1,
    opacity: Iterable[float] | None = None,
    sender_receiver_space: float = 0,
) -> tuple[Figure, Axes]:
    """
    Creates a circos plot showing the links between ligands and their targets or receptors.

    Parameters
    ----------
    circos_links : pandas.DataFrame
        dataframe with one row per link and the columns 'ligand', 'ligand_type' (the cell type the ligand is assigned
        to, e.g. by visualization.assign_ligands_to_celltype) and `dest_name` (the receiving genes)
    colors : dict
        color mapping for the ligand types and `dest_name`, e.g. {'General': ..., 'NK': ..., 'target': ...}
    dest_name : str
        the column with the receiving genes, also used as the name (and color key) of the receiving sector, e.g.
        'target' or 'receptor'
    inter_space : float
        space between celltypes
    intra_space : float
        space between genes
    opacity : Iterable of float or None
        the opacity of each link (relative to the largest value), e.g. circos_links['weight']
    sender_receiver_space : float
        the extra space between sender and receiver sector groups

    Returns
    -------
    matplotlib.Figure
        the plotted figure
    matplotlib.Axes
        the axes

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if opacity doesn't have one value per link
    """
    if type(circos_links) is not pd.DataFrame:
        raise TypeError(f"circos_links should have type pandas.DataFrame, was {type(circos_links)}")
    if type(colors) is not dict:
        raise TypeError(f"colors should have type dict[gene_t, str], was {type(colors)}")
    if type(dest_name) is not str:
        raise TypeError(f"dest_name should have type str, was {type(dest_name)}")
    if not isinstance(inter_space, Number):
        raise TypeError(f"inter_space should have type float, was {type(inter_space)}")
    if not isinstance(intra_space, Number):
        raise TypeError(f"intra_space should have type float, was {type(intra_space)}")
    if opacity is not None and not isinstance(opacity, Iterable):
        raise TypeError(f"opacity should have type Iterable[float], was {type(opacity)}")
    return _create_circos_plot(
        zip(
            zip(circos_links["ligand_type"].tolist(), circos_links["ligand"].tolist(), strict=True),
            zip(repeat(dest_name, len(circos_links)), circos_links[dest_name].tolist(), strict=True),
            strict=True,
        ),
        colors,
        inter_space,
        intra_space,
        opacity=opacity,
        sender_receiver_space=sender_receiver_space,
    )


def generate_info_tables[GeneSymbol](
    ann: AnnData,
    celltype_col: str,
    senders_oi: Sequence[str],
    receivers_oi: Sequence[str],
    lr_network_filtered: LigandReceptorNetwork[GeneSymbol],
    condition_col: str,
    condition_oi: str,
    condition_ref: str,
    case_control: bool = False,
    use_scanpy: bool = False,
) -> dict[str, pd.DataFrame]:
    """
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
        if True, calculate condition specificity, else only calculate cell type specificity
    use_scanpy : bool
        if true, use scanpy.rank_genes_groups to compute the metrics

    Returns
    -------
    dict
        dictionary containing the three dataframes:
        "sender_receiver_de",
        "sender_receiver_info",
        "lr_condition_de"

    Notes
    -----
    "lr_condition_de" is only present if case_control is True
    """
    output = {
        "sender_receiver_de": process_table_to_ic(
            calculate_de(
                ann,
                celltype_col,
                condition_oi,
                condition_col,
                features=lr_network_filtered.get_ligands().union(lr_network_filtered.get_receptors()),
                use_scanpy=use_scanpy,
            ),
            "celltype_DE",
            lr_network_filtered,
            senders_oi,
            receivers_oi,
            celltype_col=celltype_col,
        ),
        "sender_receiver_info": process_table_to_ic(
            get_avg_exp(ann, celltype_col, condition_oi, condition_col),
            "expression",
            lr_network_filtered,
            celltype_col=celltype_col,
        ),
    }
    if case_control:
        group_metrics(ann, groupby=condition_col, group_oi=condition_oi, group_ref=condition_ref)
        res = ann.uns["group_metrics"]
        output["lr_condition_de"] = process_table_to_ic(
            res[["gene", "lfc", "pval", "pval_adj"]], "group_DE", lr_network_filtered, celltype_col=celltype_col
        )
    return output


def infer_supporting_datasources(
    tf_signaling: pd.DataFrame,
    tf_regulatory: pd.DataFrame,
    lr_network: pd.DataFrame,
    sig_network: pd.DataFrame,
    gr_network: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the data sources that support the specific interactions in the extracted ligand-target signaling subnetwork.

    Parameters
    ----------
    tf_signaling : pandas.DataFrame
        dataframe which contains weighted ligand-receptor and signaling interactions  (from, to, weight)
    tf_regulatory : pandas.DataFrame
        dataframe which contains weighted gene regulatory interactions (from, to, weight)
    lr_network : pandas.DataFrame
        dataframe which contains ligand-receptor interactions
    sig_network : pandas.DataFrame
        dataframe which contains signaling interactions
    gr_network : pandas.DataFrame
        dataframe which contains gene regulatory interactions

    Returns
    -------
    pandas.DataFrame
        the integrated weighted ligand-signaling and gene regulatory network data frames

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(tf_signaling) is not pd.DataFrame:
        raise TypeError(f"tf_signaling should have type pandas.DataFrame, was {type(tf_signaling)}")
    if type(tf_regulatory) is not pd.DataFrame:
        raise TypeError(f"tf_regulatory should have type pandas.DataFrame, was {type(tf_regulatory)}")
    if type(lr_network) is not pd.DataFrame:
        raise TypeError(f"lr_network should have type pandas.DataFrame, was {type(lr_network)}")
    if type(sig_network) is not pd.DataFrame:
        raise TypeError(f"sig_network should have type pandas.DataFrame, was {type(sig_network)}")
    if type(gr_network) is not pd.DataFrame:
        raise TypeError(f"gr_network should have type pandas.DataFrame, was {type(gr_network)}")
    signaling_filtered = tf_signaling[["from", "to"]]
    signaling_filtered = signaling_filtered.merge(pd.concat((lr_network, sig_network)), on=("from", "to"), how="inner")
    signaling_filtered["layer"] = "ligand_signaling"
    regulatory_filtered = tf_regulatory[["from", "to"]]
    regulatory_filtered = regulatory_filtered.merge(gr_network, on=("from", "to"), how="inner")
    regulatory_filtered["layer"] = "regulatory"
    return pd.concat((regulatory_filtered, signaling_filtered))


def calculate_fraction_top_predicted(
    affected_gene_predictions: pd.DataFrame, quantile_cutoff: float = 0.95
) -> pd.DataFrame:
    """
    Determine the fraction of genes belonging to the geneset or background to the top-predicted genes.

    Parameters
    ----------
    affected_gene_predictions : pandas.DataFrame
        dataframe which contains "response" and "prediction" columns
    quantile_cutoff : float
        Quantile of which genes should be considered as top-predicted targets.
        Default: 0.95

    Returns
    -------
    pandas.DataFrame
        A dataframe indicating the number of genes belonging to the gene set of interest or background (true_target column),
        the number and fraction of genes of these groups that were part of the top predicted targets in a specific
        cross-validation round.

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(affected_gene_predictions) is not pd.DataFrame:
        raise TypeError(
            f"affected_gene_predictions should have type pandas.DataFrame, was {type(affected_gene_predictions)}"
        )
    if not isinstance(quantile_cutoff, Number):
        raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
    prediction = affected_gene_predictions["prediction"]
    predicted_positive = (
        affected_gene_predictions[["response", "prediction"]]
        .rename(columns={"prediction": "positive_prediction"})[prediction >= np.quantile(prediction, quantile_cutoff)]
        .groupby("response")
        .count()
    )
    predicted_positive.index.name = "true_target"
    predicted_positive.reset_index(inplace=True)
    all = (
        affected_gene_predictions[["response", "prediction"]]
        .rename(columns={"response": "true_target", "prediction": "n"})
        .groupby("true_target")
        .count()
    )
    all.reset_index(inplace=True)
    all = all.merge(predicted_positive, on="true_target", how="inner")
    all["fraction_positive_predicted"] = all["positive_prediction"] / all["n"]
    return all


def calculate_fraction_top_predicted_fisher(affected_gene_predictions: pd.DataFrame, quantile_cutoff: float = 0.95):
    """
    Performs a Fisher's exact test to determine whether genes belonging to the gene set of interest are more likely to be part
    of the top-predicted targets.

    Parameters
    ----------
    affected_gene_predictions : pandas.DataFrame
        dataframe which contains "response" and "prediction" columns
    quantile_cutoff : float
        Quantile of which genes should be considered as top-predicted targets.
        Default: 0.95

    Returns
    -------
    object
        summary of the Fisher's exact test, has attributes "statistic" and "pvalue" (see scipy.stats.fisher_exact)

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(affected_gene_predictions) is not pd.DataFrame:
        raise TypeError(
            f"affected_gene_predictions should have type pandas.DataFrame, was {type(affected_gene_predictions)}"
        )
    if not isinstance(quantile_cutoff, Number):
        raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
    prediction = affected_gene_predictions["prediction"]
    predicted_positive = (
        affected_gene_predictions[["response", "prediction"]]
        .rename(columns={"prediction": "positive_prediction"})[prediction >= np.quantile(prediction, quantile_cutoff)]
        .groupby("response")
        .count()
    )
    predicted_positive.reset_index(inplace=True)
    all = (
        affected_gene_predictions[["response", "prediction"]]
        .rename(columns={"prediction": "n"})
        .groupby("response")
        .count()
    )
    all.reset_index(inplace=True)
    df = all.merge(predicted_positive, on="response", how="left")
    df["positive_prediction"] = np.nan_to_num(df["positive_prediction"])
    true_res = df[df["response"] == 1]
    false_res = df[df["response"] == 0]
    tp = true_res["positive_prediction"].iloc[0]
    fp = false_res["positive_prediction"].iloc[0]
    fn = true_res["n"].iloc[0] - true_res["positive_prediction"].iloc[0]
    tn = false_res["n"].iloc[0] - false_res["positive_prediction"].iloc[0]
    return fisher_exact(np.array([[tp, fp], [fn, tn]]), alternative="greater")


def get_top_predicted_genes(affected_gene_predictions: pd.DataFrame, quantile_cutoff: float = 0.95) -> pd.DataFrame:
    """
    Find which genes were among the top-predicted targets genes in a specific cross-validation round and see whether these
    genes belong to the gene set of interest as well.

    Parameters
    ----------
    affected_gene_predictions : pandas.DataFrame
        dataframe which contains "response" and "prediction" columns
    quantile_cutoff : float
        Quantile of which genes should be considered as top-predicted targets.
        Default: 0.95

    Returns
    -------
    pandas.DataFrame
        A dataframe indicating for every gene whether it belongs to the geneset and whether it belongs to the top-predicted genes
        in a specific cross-validation round.

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(affected_gene_predictions) is not pd.DataFrame:
        raise TypeError(
            f"affected_gene_predictions should have type pandas.DataFrame, was {type(affected_gene_predictions)}"
        )
    if not isinstance(quantile_cutoff, Number):
        raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
    prediction = affected_gene_predictions["prediction"]
    predicted_positive = affected_gene_predictions.copy()
    predicted_positive["predicted_top_target"] = prediction >= np.quantile(prediction, quantile_cutoff)
    predicted_positive.rename(columns={"response": "true_target"}, inplace=True)
    return predicted_positive[["gene", "true_target", "predicted_top_target"]]


def normalize_single_cell_ligand_activities(ligand_activities: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize single-cell ligand activities to make ligand activities over different cells comparable.

    Parameters
    ----------
    ligand_activities : pandas.DataFrame
        Output from the function "predict_single_cell_ligand_activities"

    Returns
    -------
    pandas.DataFrame
        A dataframe giving the normalized ligand activity scores for single cells.
        Following columns in the tibble: cell, ligand, pearson, which is the normalized ligand activity value.

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(ligand_activities) is not pd.DataFrame:
        raise TypeError(f"ligand_activities should have type pandas.DataFrame, was {type(ligand_activities)}")
    single_ligand_activities_aupr_norm = df_grouped_apply(
        ligand_activities[["cell", "ligand", "aupr"]],
        groupby="cell",
        func=lambda x: scaling_modified_zscore(np.array(x["aupr"])),
        dest="aupr",
    )
    return single_ligand_activities_aupr_norm.pivot(index="cell", columns="ligand", values="aupr")

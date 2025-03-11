from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.utils import (
    combine_by_key,
    combine_dicts
)
from nichenetpy.extraction import (
    get_expressed_genes,
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
from nichenetpy.ann_utils import subset_ann

from itertools import cycle, chain
from collections.abc import Iterable
from anndata import AnnData
from pycirclize import Circos
from pycirclize.utils import ColorCycler
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from scipy.stats import fisher_exact
from numbers import Number

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_geneset_oi(
    ann:AnnData,
    receiver:str,
    condition_col:str,
    condition_oi:str,
    layer:str="data",
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
    condition_col : str
        the name of the column in obs which contains the conditions
    condition_oi : str
        the condition of interest
    layer : str
        the name of the layer which contains the data matrix
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
    ann_receiver = subset_ann(
        ann,
        val=receiver,
        layers=[layer],
        val_col="celltype"
    )
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
            (DE_table[condition_col] == condition_oi) &
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
    condition_col:str,
    condition_oi:str,
    condition_ref:str,
    sender_celltypes:Iterable[str]=None,
    get_expressed_genes_pct:float=0.05,
    layer:str="data",
    celltype_col:str="celltype",
    max_pval_adj:float=0.05,
    min_abs_lfc:float=0.25,
    min_pct:float=0.05,
    ligands_top_n:int=30,
    targets_top_n:int=100,
    lr_sig:WeightedNetwork=None,
    get_ltl:bool=False,
    get_lfc:bool=False,
    get_prioritization_table:bool=False,
    case_control:bool=True
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
    condition_col : str
        the name of the column in obs which contains the conditions
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
            geneset_oi : set of str
                the geneset of interest
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
        if get_prioritization_table is True, additionally
            prioritization_table : pandas.DataFrame
    
    Raises
    ------
    ValueError
        if get_prioritization_table is True and sender_celltypes is not provided
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
        layer=layer,
        condition_col=condition_col,
        max_pval_adj=max_pval_adj,
        min_abs_lfc=min_abs_lfc,
        min_pct=min_pct
    )
    geneset.intersection_update(predictor.get_genes())
    output["geneset_oi"] = geneset
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
        ann_focused = subset_ann(
            ann,
            val=sender_celltypes,
            layers=[layer],
            val_col="celltype"
        )
        ann_focused.X = ann_focused.layers[layer]
        output["ann_focused"] = ann_focused
        if get_lfc:
            output["lfcs"] = [
                get_lfc_celltype(
                    ann,
                    celltype,
                    condition_col=condition_col,
                    condition_oi=condition_oi,
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
    title:str="ligand activity",
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

def create_ligand_receptor_links_circos_plot(
    senders:Iterable[str],
    receivers:Iterable[str],
    ligands:Iterable[str],
    receptors:Iterable[str]
) -> Figure:
    '''
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
    
    Returns
    -------
    matplotlib.Figure
        the plotted figure
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if not isinstance(senders, Iterable):
        raise TypeError(f"senders should have type Iterable[str], was {type(senders)}")
    if not isinstance(receivers, Iterable):
        raise TypeError(f"receivers should have type Iterable[str], was {type(receivers)}")
    if not isinstance(ligands, Iterable):
        raise TypeError(f"ligands should have type Iterable[str], was {type(ligands)}")
    if not isinstance(receptors, Iterable):
        raise TypeError(f"receptors should have type Iterable[str], was {type(receptors)}")
    celltypes = set(chain(senders, receivers))
    link_count_in = dict()
    link_count_out = dict()
    links = []
    for sender, receiver, ligand, receptor in zip(senders, receivers, ligands, receptors):
        key_out = f"{sender}_{ligand}_out"
        if key_out in link_count_out:
            link_count_out[key_out] += 1
        else:
            link_count_out[key_out] = 1
        key_in = f"{receiver}_{receptor}_in"
        if key_in in link_count_in:
            link_count_in[key_in] += 1
        else:
            link_count_in[key_in] = 1
        links.append((
            key_out,
            key_in,
            link_count_out[key_out],
            link_count_in[key_in]
        ))
    circos = Circos(
    sectors={
            key: count
            for key, count in chain(link_count_in.items(), link_count_out.items())
        },
        space=1
    )
    ColorCycler.set_cmap("Set1")
    colors = dict(zip(celltypes, ColorCycler.get_color_list(len(celltypes))))
    for sector in circos.sectors:
        celltype, gene, _ = sector.name.split("_")
        sector.text(gene, size=10, orientation="vertical")
        track = sector.add_track((95, 100))
        track.axis(fc=colors[celltype])
    for sender, receiver, send_pos, rec_pos in links:
        celltype = sender.split("_")[0]
        circos.link_line(
            (sender, send_pos - 0.5),
            (receiver, rec_pos - 0.5,),
            direction=1,
            color=colors[celltype]
        )
    fig = circos.plotfig()
    circos.ax.legend(
        handles=[
            Patch(color=color, label=celltype)
            for celltype, color in colors.items()
        ],
        bbox_to_anchor=(0, 1.1),
        loc="right",
        ncols=1,
    )
    return fig

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

def infer_supporting_datasources(
    tf_signaling:pd.DataFrame,
    tf_regulatory:pd.DataFrame,
    lr_network:pd.DataFrame,
    sig_network:pd.DataFrame,
    gr_network:pd.DataFrame
):
    '''
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
    sig_network : pandas.DataFrame
    gr_network : pandas.DataFrame

    Returns
    -------
    tuple of pandas.DataFrame
        the integrated weighted ligand-signaling and gene regulatory network data frames

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
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
    return pd.concat((
        regulatory_filtered,
        signaling_filtered
    ))

def calculate_fraction_top_predicted(
    affected_gene_predictions:pd.DataFrame,
    quantile_cutoff:float=0.95
) -> pd.DataFrame:
    '''
    Determine the fraction of genes belonging to the geneset or background and to the top-predicted genes.

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
    '''
    if type(affected_gene_predictions) is not pd.DataFrame:
        raise TypeError(f"affected_gene_predictions should have type pandas.DataFrame, was {type(affected_gene_predictions)}")
    if not isinstance(quantile_cutoff, Number):
        raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
    prediction = affected_gene_predictions["prediction"]
    predicted_positive = affected_gene_predictions[["response", "prediction"]].rename(columns={"prediction": "positive_prediction"})[
        prediction >= np.quantile(prediction, quantile_cutoff)
    ].groupby("response").count()
    predicted_positive.index.name = "true_target"
    predicted_positive.reset_index(inplace=True)
    all = affected_gene_predictions[["response", "prediction"]].rename(
        columns={"response": "true_target", "prediction": "n"}
    ).groupby("true_target").count()
    all.reset_index(inplace=True)
    all = all.merge(predicted_positive, on="true_target", how="inner")
    all["fraction_positive_predicted"] = all["positive_prediction"]/all["n"]
    return all

def calculate_fraction_top_predicted_fisher(
    affected_gene_predictions:pd.DataFrame,
    quantile_cutoff:float=0.95
) -> object:
    '''
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
    '''
    if type(affected_gene_predictions) is not pd.DataFrame:
        raise TypeError(f"affected_gene_predictions should have type pandas.DataFrame, was {type(affected_gene_predictions)}")
    if not isinstance(quantile_cutoff, Number):
        raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
    prediction = affected_gene_predictions["prediction"]
    predicted_positive = affected_gene_predictions[["response", "prediction"]].rename(columns={"prediction": "positive_prediction"})[
        prediction >= np.quantile(prediction, quantile_cutoff)
    ].groupby("response").count()
    predicted_positive.reset_index(inplace=True)
    all = affected_gene_predictions[["response", "prediction"]].rename(columns={"prediction": "n"}).groupby("response").count()
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

def get_top_predicted_genes(
    affected_gene_predictions:pd.DataFrame,
    quantile_cutoff:float=0.95
) -> pd.DataFrame:
    '''
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
    '''
    if type(affected_gene_predictions) is not pd.DataFrame:
        raise TypeError(f"affected_gene_predictions should have type pandas.DataFrame, was {type(affected_gene_predictions)}")
    if not isinstance(quantile_cutoff, Number):
        raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
    prediction = affected_gene_predictions["prediction"]
    predicted_positive = affected_gene_predictions.copy()
    predicted_positive["predicted_top_target"] = (prediction >= np.quantile(prediction, quantile_cutoff))
    predicted_positive.rename(columns={"response": "true_target"}, inplace=True)
    return predicted_positive[["gene", "true_target", "predicted_top_target"]]
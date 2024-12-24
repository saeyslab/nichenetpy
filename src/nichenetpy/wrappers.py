from nichenetpy.extraction import subset_ann_celltype

from anndata import AnnData
from itertools import chain, cycle

import scanpy as sc


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

def combine_weighted_ligand_target_links(active_ligand_target_links:list[dict]) -> list[tuple[str, str, float]]:
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
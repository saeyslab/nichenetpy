from nichenetpy.metrics import calculate_metrics

import numpy as np


class LigandActivityPredictor:
    '''
    This class facilitates the computation of ligand activities using a ligand-target matrix. 

    Parameters
    ----------
    ligand_target_matrix : numpy.ndarray
        a (ngenes X nligands) matrix describing the potential that a ligand may regulate a target gene
    row_names : list of str
        list of names of the rows/genes
    col_names : list of str
        list of names of the columns/ligands
    
    Attributes
    ----------
    ligand_target_matrix : numpy.ndarray
        a (ngenes X nligands) matrix describing the potential that a ligand may regulate a target gene
    row_names : list of str
        list of names of the rows/genes
    col_names : list of str
        list of names of the columns/ligands
    ligand2index : dict
        mapping of ligand names to indices
    gene2index : dict
        mapping of gene names to indices
    '''
    def __init__(
        self,
        ligand_target_matrix:np.ndarray,
        row_names:list[str],
        col_names:list[str],
    ) -> None:
        self.ligand_target_matrix = ligand_target_matrix
        self.row_names = row_names
        self.col_names = col_names
        self.ligand2index = dict(zip(self.col_names, range(len(self.col_names))))
        self.gene2index = dict(zip(self.row_names, range(len(self.row_names))))
    
    def get_ligands(self) -> set[str]:
        '''
        Get the ligands from the ligand-target matrix. 

        Returns
        -------
        set
            all ligands in the ligand-target matrix
        '''
        return set(self.ligand2index.keys())
    
    def get_genes(self) -> set[str]:
        '''
        Get the genes from the ligand-target matrix. 

        Returns
        -------
        set
            all geness in the ligand-target matrix
        '''
        return set(self.gene2index.keys())

    def predict_ligand_activities(
        self,
        geneset:list[str],
        background_expressed_genes:list[str],
        potential_ligands:list[str]
    ) -> dict[str, dict[str, float]]:
        '''
        Predict activities of ligands in regulating expression of a gene set of interest. Ligand activities are defined as how well they predict the observed transcriptional response (i.e. gene set) according to the NicheNet model.

        Parameters
        ----------
        geneset : list of str
            the gene symbols of genes of which the expression is potentially affected by ligands from the interacting cell
        background_expressed_genes : list of str
            the gene symbols of the background, non-affected, genes (can contain the symbols of the affected genes as well)
        potential ligands : list of str
            the gene symbols of the potentially active ligands for which you want to compute ligand activities

        Returns
        -------
        dict
            nested dictionary which contains the ligand activity for each ligand
        '''
        output = dict()

        # create the expected gene expression response vector
        response = dict((gene, 0) for gene in background_expressed_genes if gene not in geneset)
        for gene in geneset:
            response[gene] = 1
        
        # create the prediction model vector
        predictions = ([
            dict(zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index[ligand]]))
            for ligand in potential_ligands
        ])

        # compute the metrics for each ligand
        for ligand, prediction in zip(potential_ligands, predictions):
            # we need to match the predictions with the responses so we intersect and sort by key
            common_keys = prediction.keys() & response.keys()
            pred = [tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])]
            resp = [tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])]
            output[ligand] = calculate_metrics(pred, resp)
        return output
    
    def get_weighted_ligand_target_links(self, ligand:str, geneset:set[str], n:int=250) -> dict:
        '''
        Infer active ligand target links between possible ligands and genes belonging to a gene set of interest: consider the intersect between the top n targets of a ligand and the gene set.

        Parameters
        ----------
        ligand : str
            the gene symbol of the potentially active ligand for which you want to find target genes
        geneset : list of str
            the gene symbols of genes for which the expression is potentially affected by ligands from the interacting cell
        n : int
            the top n of targets per ligand that will be considered, defaults to 250

        Returns
        -------
        dict
            dictionary which contains:
                the input ligand, the target genes and the regulatory potential scores between the ligand and each target
        '''
        targets = set(
            e[0] for e in sorted(
                zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index[ligand]]),
                key=lambda x : x[1],
                reverse=True
            )[:n]
        ).intersection(geneset)
        targets = sorted(targets)
        if len(targets) == 0:
            return {
                "ligand": ligand,
                "target": None,
                "weight": None
            }
        else:
            return {
                "ligand": ligand,
                "target": targets,
                "weight": [self.ligand_target_matrix[self.gene2index[target]][self.ligand2index[ligand]] for target in targets]
            }
    
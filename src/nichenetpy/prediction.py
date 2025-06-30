from nichenetpy.metrics import calculate_ligand_importance_metrics
from nichenetpy.utils import subset_matrix, combine_dicts

from collections.abc import Collection, Iterable
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from itertools import chain
from functools import reduce
from numbers import Number

import numpy as np
import pandas as pd


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
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Attributes
    ----------
    ligand_target_matrix : numpy.ndarray
        a (ngenes X nligands) matrix describing the potential that a ligand may regulate a target gene
    row_names : list of str
        list of names of the rows/genes
    col_names : list of str
        list of names of the columns/ligands
    _ligand2index : dict
        mapping of ligand names to indices
    _gene2index : dict
        mapping of gene names to indices
    '''
    def __init__(
        self,
        ligand_target_matrix:np.ndarray,
        row_names:list[str]|tuple[str],
        col_names:list[str]|tuple[str],
    ) -> None:
        if type(ligand_target_matrix) is not np.ndarray:
            raise TypeError(f"ligand_target_matrix should have type numpy.ndarray, was {type(ligand_target_matrix)}")
        if type(row_names) is not list and type(row_names) is not tuple:
            raise TypeError(f"row_names should have type list[str] or tuple[str], was {type(row_names)}")
        if type(col_names) is not list and type(col_names) is not tuple:
            raise TypeError(f"col_names should have type list[str] or tuple[str], was {type(col_names)}")
        self.ligand_target_matrix = ligand_target_matrix
        self.row_names = row_names
        self.col_names = col_names
        self._ligand2index = dict(zip(self.col_names, range(len(self.col_names))))
        self._gene2index = dict(zip(self.row_names, range(len(self.row_names))))
    
    def ligand2index(self, ligand:str):
        '''
        Get the index of the ligand in the ligand-target matrix

        Returns
        -------
        int
            the index
        
        Raises
        ------
        ValueError
            if the ligand is not present in the ligand-target matrix
        '''
        if ligand not in self._ligand2index:
            raise ValueError(f"{ligand} not in ligand_target_matrix")
        return self._ligand2index[ligand]
    
    def gene2index(self, gene:str):
        '''
        Get the index of the gene in the ligand-target matrix

        Returns
        -------
        int
            the index
        
        Raises
        ------
        ValueError
            if the gene is not present in the ligand-target matrix
        '''
        if gene not in self._gene2index:
            raise ValueError(f"{gene} not in ligand_target_matrix")
        return self._gene2index[gene]
    
    def get_ligands(self) -> set[str]:
        '''
        Get the ligands from the ligand-target matrix. 

        Returns
        -------
        set
            all ligands in the ligand-target matrix
        '''
        return set(self._ligand2index.keys())
    
    def get_genes(self) -> set[str]:
        '''
        Get the genes from the ligand-target matrix. 

        Returns
        -------
        set
            all genes in the ligand-target matrix
        '''
        return set(self._gene2index.keys())
    
    def evaluate_target_prediction(
        self,
        ligand:str,
        response:dict[str, int]
    ):
        '''
        Evaluate how well the model (i.e. the inferred ligand-target probability scores) is able to predict the observed response
        to a ligand (e.g. the set of DE genes after treatment of cells by a ligand). It shows several classification evaluation
        metrics for the prediction. 

        Parameters
        ----------
        ligand : str
            the ligand of interest
        response : dict[str, int]
            a dictionary indicating whether a target is a true target of the possibly active ligand

        Returns
        -------
        dict
            the evaluation metrics
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        ValueError
            if the arguments are invalid
        '''
        if type(ligand) is not str:
            raise TypeError(f"ligand should have type str, was {type(ligand)}")
        if type(response) is not dict:
            raise TypeError(f"response should have type dict, was {type(response)}")
        # create the prediction model vector
        prediction = dict(zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index(ligand)]))
        # we need to match the predictions with the responses so we intersect and sort by key
        common_keys = prediction.keys() & response.keys()
        pred = [tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])]
        resp = [tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])]
        return calculate_ligand_importance_metrics(pred, resp)

    def predict_ligand_activities(
        self,
        geneset:Collection[str],
        background_expressed_genes:Iterable[str],
        potential_ligands:Iterable[str]
    ) -> dict[str, dict[str, float]]:
        '''
        Predict activities of ligands in regulating expression of a gene set of interest.
        Ligand activities are defined as how well they predict the observed transcriptional response (i.e. gene set) according
        to the NicheNet model.

        Parameters
        ----------
        geneset : Collection of str
            the genes of which the expression is potentially affected by ligands from the interacting cell
        background_expressed_genes : Iterable of str
            the background, non-affected, genes (can contain the symbols of the affected genes as well)
        potential_ligands : Iterable of str
            the potentially active ligands for which you want to compute ligand activities

        Returns
        -------
        dict
            the ligand activity for each ligand
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        ValueError
            if the arguments are invalid
        '''
        if not isinstance(geneset, Collection):
            raise TypeError(f"geneset should have type Collection, was {type(geneset)}")
        if not isinstance(background_expressed_genes, Iterable):
            raise TypeError(f"background_expressed_genes should have type Iterable, was {type(background_expressed_genes)}")
        if not isinstance(potential_ligands, Iterable):
            raise TypeError(f"potential_ligands should have type Iterable, was {type(potential_ligands)}")
        output = dict()
        # create the expected gene expression response vector
        response = dict((gene, 0) for gene in background_expressed_genes if gene not in geneset)
        for gene in geneset:
            response[gene] = 1
        # compute the metrics for each ligand
        for ligand in potential_ligands:
            output[ligand] = self.evaluate_target_prediction(ligand, response)
        return output
    
    def predict_single_cell_ligand_activities(
        self,
        cells:Collection[str],
        expression_scaled:np.ndarray,
        expression_scaled_rows:Iterable[str],
        expression_scaled_cols:Iterable[str],
        potential_ligands:Collection[str],
        quantile_cutoff:float=0.975,
    ) -> dict[tuple[str, str], dict[str, float]]:
        '''
        Predict activities of ligands in regulating expression of a gene set of interest.
        Ligand activities are defined as how well they predict the observed transcriptional response (i.e. gene set) according
        to the NicheNet model.

        Parameters
        ----------
        cells : Collection of str
            the cells for which the ligand activities should be calculated
        expression_scaled : np.ndarray
            scaled expression matrix of single-cells
            (scaled such that high values indicate that a gene is stronger expressed in that cell compared to others)
        expression_scaled_rows : Iterable of str
            the names of the rows of expression_scaled
        expression_scaled_cols : Iterable of str
            the names of the columns of expression_scaled
        potential_ligands : Collection of str
            the genes of the potentially active ligands for which you want to define ligand activities
        quantile_cutoff : float
            the cutoff value used to compute the response vector

        Returns
        -------
        dict
            the ligand activity for each ligand
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        ValueError
            if the arguments are invalid
        '''
        if not isinstance(cells, Collection):
            raise TypeError(f"cells should have type Collection[str], was {type(cells)}")
        if type(expression_scaled) is not np.ndarray:
            raise TypeError(f"expression_scaled should have type numpy.ndarray, was {type(expression_scaled)}")
        if not isinstance(expression_scaled_rows, Iterable):
            raise TypeError(f"expression_scaled_rows should have type Iterable[str], was {type(expression_scaled_rows)}")
        if not isinstance(expression_scaled_cols, Iterable):
            raise TypeError(f"expression_scaled_cols should have type Iterable[str], was {type(expression_scaled_cols)}")
        if not isinstance(potential_ligands, Collection):
            raise TypeError(f"potential_ligands should have type Collection[str], was {type(potential_ligands)}")
        if not isinstance(quantile_cutoff, Number):
            raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
        output = dict()
        row2id = dict(zip(expression_scaled_rows, range(len(expression_scaled_rows))))
        for cell in cells:
            if cell not in row2id:
                raise ValueError(f"{cell} not in ligand_target_matrix")
            response = expression_scaled[row2id[cell], :]
            qt = np.quantile(response, quantile_cutoff)
            response = dict(zip(
                expression_scaled_cols,
                (1 if e >= qt else 0 for e in response)
            ))
            for ligand in potential_ligands:
                prediction = dict(zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index(ligand)]))
                common_keys = prediction.keys() & response.keys()
                pred = [tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])]
                resp = [tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])]
                output[(cell, ligand)] = calculate_ligand_importance_metrics(pred, resp)
        return output
    
    def get_weighted_ligand_target_links(
        self,
        ligand:str,
        geneset:set[str],
        n:int=250
    ) -> dict:
        '''
        Infer active ligand target links between possible ligands and genes belonging to a gene set of interest: consider the intersect between the top n targets of a ligand and the gene set.

        Parameters
        ----------
        ligand : str
            the gene symbol of the potentially active ligand for which you want to find target genes
        geneset : set of str
            the genes for which the expression is potentially affected by ligands from the interacting cell
        n : int
            the top n of targets per ligand that will be considered, defaults to 250

        Returns
        -------
        dict
            dictionary which contains:
                the input ligand, the target genes and the regulatory potential scores between the ligand and each target
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        ValueError
            if the arguments are invalid
        '''
        if type(ligand) is not str:
            raise TypeError(f"ligand should have type str, was {type(ligand)}")
        if type(geneset) is not set:
            raise TypeError(f"geneset should have type set, was {type(geneset)}")
        if type(n) is not int:
            raise TypeError(f"n should have type int, was {type(n)}")
        if n < self.ligand_target_matrix.shape[1]:
            top_n_score = sorted(
                self.ligand_target_matrix[:, self.ligand2index(ligand)],
                reverse=True
            )[n-1]
        else:
            top_n_score = min(
                sorted(
                    self.ligand_target_matrix[:, self.ligand2index(ligand)],
                    reverse=True
                )[:n]
            )
        targets = sorted(
            set(
                e[0]
                for e in zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index(ligand)])
                if e[1] >= top_n_score
            ).intersection(geneset)
        )
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
                "weight": [self.ligand_target_matrix[self.gene2index(target)][self.ligand2index(ligand)] for target in targets]
            }

def assess_rf_class_probabilities(
    folds:int,
    geneset:set[str],
    background_expressed_genes:set[str],
    ligands_oi:Iterable[str],
    predictor:LigandActivityPredictor,
    ntrees:int=1000
):
    '''
    Assess probability that a target gene belongs to the geneset based on a multi-ligand random forest model (with cross-validation).
    Target genes and background genes will be split in different groups in a stratified way.

    Parameters
    ----------
    folds : int
        integer describing how many folds should be used
    geneset : set of str
        the genes for which the expression is potentially affected by ligands from the interacting cell
    background_expressed_genes : set of str
        the background, non-affected, genes (can contain the symbols of the affected genes as well)
    ligands_oi : Iterable of str
        the ligands you want to build the multi-ligand random forest with
    predictor : LigandActivityPredictor
        the ligand-activity predictor which contains the ligand-target matrix
    ntrees : int
        the amount of trees in the random forest

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns: "gene", "response", "prediction".
        Response indicates whether the gene belongs to the geneset of interest, prediction gives the probability this gene
        belongs to the geneset according to the random forest model.

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    '''
    if type(folds) is not int:
        raise TypeError(f"folds should have type int, was {type(folds)}")
    if type(geneset) is not set:
        raise TypeError(f"geneset should have type set[str], was {type(geneset)}")
    if type(background_expressed_genes) is not set:
        raise TypeError(f"background_expressed_genes should have type set[str], was {type(background_expressed_genes)}")
    if not isinstance(ligands_oi, Iterable):
        raise TypeError(f"ligands_oi should have type Iterable[str], was {type(ligands_oi)}")
    if type(predictor) is not LigandActivityPredictor:
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(LigandActivityPredictor)}")
    if type(ntrees) is not int:
        raise TypeError(f"ntrees should have type int, was {type(ntrees)}")
    geneset.intersection_update(predictor.row_names)
    background_expressed_genes.intersection_update(predictor.row_names)
    background_expressed_genes = np.array([[e] for e in background_expressed_genes.difference(geneset)])
    geneset = np.array([[e] for e in geneset])
    kf = KFold(n_splits=folds, shuffle=True)
    geneset_predictions_all = []
    for beg_split, geneset_split in zip(kf.split(background_expressed_genes), kf.split(geneset)):
        geneset_train, geneset_test = geneset_split
        beg_train, beg_test = beg_split
        row_names, res = zip(
            *chain(
                ((str(geneset[id][0]), 1) for id in geneset_train),
                ((str(background_expressed_genes[id][0]), 0) for id in beg_train)
            )
        )
        pred_mat = subset_matrix(
            predictor.ligand_target_matrix,
            rows=[predictor.gene2index(gene) for gene in row_names],
            cols=[predictor.ligand2index(ligand) for ligand in ligands_oi]
        )
        rf = RandomForestClassifier(n_estimators=ntrees)
        rf.fit(X=pred_mat, y=res)
        row_names, res = zip(
            *chain(
                ((str(geneset[id][0]), 1) for id in geneset_test),
                ((str(background_expressed_genes[id][0]), 0) for id in beg_test)
            )
        )
        pred_mat = subset_matrix(
            predictor.ligand_target_matrix,
            rows=[predictor.gene2index(gene) for gene in row_names],
            cols=[predictor.ligand2index(ligand) for ligand in ligands_oi]
        )
        pred = rf.apply(pred_mat)
        score = [
            sum(
                (
                    rf.estimators_[j].classes_[np.argmax(rf.estimators_[j].tree_.value[pred[i, j]])]
                    for j in range(pred.shape[1])
                )
            ) / pred.shape[1] for i in range(pred.shape[0])
        ]
        geneset_predictions_all.append(
            pd.DataFrame({
                "gene": row_names,
                "response": res,
                "prediction": score
            })
        )
    return pd.concat(geneset_predictions_all)
from nichenetpy.metrics import (
    calculate_prediction_evaluation_metrics,
    calculate_aupr,
    calculate_auroc
)
from nichenetpy.utils import subset_matrix
from nichenetpy.typing import (
    nichenet_matrix,
    gene_t
)

from collections.abc import Collection, Iterable
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from itertools import chain, repeat
from numbers import Number
from scipy.sparse import (
    csr_matrix,
    csc_matrix
)
from scipy.stats import pearsonr
from random import uniform

import numpy as np
import pandas as pd
import warnings


class LigandActivityPredictor:
    '''
    This class facilitates the computation of ligand activities using a ligand-target matrix. 

    Parameters
    ----------
    ligand_target_matrix : numpy.ndarray
        a (ngenes X nligands) matrix describing the potential that a ligand may regulate a target gene
    row_names : list of gene_t
        list of names of the rows/genes
    col_names : list of gene_t
        list of names of the columns/ligands
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Attributes
    ----------
    ligand_target_matrix : numpy.ndarray
        a (ngenes X nligands) matrix describing the potential that a ligand may regulate a target gene
    row_names : list of gene_t
        list of names of the rows/genes
    col_names : list of gene_t
        list of names of the columns/ligands
    _ligand2index : dict
        mapping of ligand names to indices
    _gene2index : dict
        mapping of gene names to indices
    
    Notes
    -----
    since a NicheNet analysis needs to index the columns of the ligand-target matrix many times, it is recommended to use a column-major memory layout
    '''
    def __init__(
        self,
        ligand_target_matrix:nichenet_matrix,
        row_names:list[gene_t]|tuple[gene_t],
        col_names:list[gene_t]|tuple[gene_t]
    ) -> None:
        self.ligand_target_matrix = ligand_target_matrix
        if type(ligand_target_matrix) is csr_matrix:
            warnings.warn("a scipy.csr_matrix was passed, this will result in extremely slow column indexing and is therefore not supported, the matrix is automatically converted to column-major format")
            if self.matrix_density() > 0.5:
                self.ligand_target_matrix = self.ligand_target_matrix.toarray(order="F")
            else:
                self.ligand_target_matrix = csc_matrix(self.ligand_target_matrix)
        elif type(ligand_target_matrix) is csc_matrix:
            density = self.matrix_density()
            if density > 0.5:
                warnings.warn(f"a scipy.csc_matrix was passed with a density of {density}, the reduction in memory consumption may not be worth the increased time to index columns, consider using a column-major numpy.ndarray in stead")
        elif type(ligand_target_matrix) is np.ndarray:
            if ligand_target_matrix.flags.c_contiguous:
                warnings.warn("a row-major numpy.ndarray was passed, consider converting to a column-major numpy.ndarray for faster column indexing")
        else:
            raise TypeError(f"ligand_target_matrix should have type numpy.ndarray, scipy.csr_matrix or scipy.csc_matrix, was {type(ligand_target_matrix)}")
        if type(row_names) is not list and type(row_names) is not tuple:
            raise TypeError(f"row_names should have type list[gene_t] or tuple[gene_t], was {type(row_names)}")
        if type(col_names) is not list and type(col_names) is not tuple:
            raise TypeError(f"col_names should have type list[gene_t] or tuple[gene_t], was {type(col_names)}")
        self.row_names = row_names
        self.col_names = col_names
        self._ligand2index = dict(zip(self.col_names, range(len(self.col_names))))
        self._gene2index = dict(zip(self.row_names, range(len(self.row_names))))
    
    def matrix_density(self) -> float:
        '''
        compute the ratio of non-zero elements in the ligand-target matrix

        Returns
        -------
        float
            the ratio of non-zero elements in the ligand-target matrix
        '''
        return (
            (np.count_nonzero(self.ligand_target_matrix) if type(self.ligand_target_matrix) is np.ndarray else self.ligand_target_matrix.getnnz())
            /
            (self.ligand_target_matrix.shape[0] * self.ligand_target_matrix.shape[1])
        )
    
    def ligand2index(self, ligand:gene_t):
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
    
    def gene2index(self, gene:gene_t):
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
    
    def get_ligands(self) -> set[gene_t]:
        '''
        Get the ligands from the ligand-target matrix. 

        Returns
        -------
        set
            all ligands in the ligand-target matrix
        '''
        return set(self._ligand2index.keys())
    
    def get_genes(self) -> set[gene_t]:
        '''
        Get the genes from the ligand-target matrix. 

        Returns
        -------
        set
            all genes in the ligand-target matrix
        '''
        return set(self._gene2index.keys())
    
    def get_col(
        self,
        index:int|gene_t,
        is_index:bool=True
    ) -> np.ndarray:
        '''
        index the columns of the ligand-target matrix, the key may also be a gene symbol

        Parameters
        ----------
        index : int or gene_t
            the index or gene symbol
        is_index : bool
            whether index is an index or gene symbol

        Returns
        -------
        dict
            the corresponding column
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        IndexError
            if the index is invalid
        '''
        if type(index) is str:
            is_index = False
        if not is_index:
            try:
                index = self.ligand2index(index)
            except KeyError:
                raise IndexError(f"{index} is not a gene symbol with a corresponding column in the ligand-target matrix")
        elif type(index) is not int:
            raise TypeError(f"index should have type int if is_index, was {type(index)}")
        if index >= self.ligand_target_matrix.shape[1]:
            raise IndexError(f"column index out of bounds, {index} for ligand-target matrix of shape {self.ligand_target_matrix.shape}")
        if type(self.ligand_target_matrix) is np.ndarray:
            return self.ligand_target_matrix[:, index]
        else: # csc_matrix
            indices_non_zero = self.ligand_target_matrix.indices[
                self.ligand_target_matrix.indptr[index]:self.ligand_target_matrix.indptr[index+1]
            ]
            values_non_zero = self.ligand_target_matrix.data[
                self.ligand_target_matrix.indptr[index]:self.ligand_target_matrix.indptr[index+1]
            ]
            col = np.zeros(shape=(self.ligand_target_matrix.shape[0],))
            for i, v in zip(indices_non_zero, values_non_zero):
                col[i] = v
            return np.array(col)
    
    def evaluate_target_prediction(
        self,
        ligand:gene_t,
        response:dict[gene_t, int]
    ):
        '''
        Evaluate how well the model (i.e. the inferred ligand-target probability scores) is able to predict the observed response
        to a ligand (e.g. the set of DE genes after treatment of cells by a ligand). It shows several classification evaluation
        metrics for the prediction. 

        Parameters
        ----------
        ligand : gene_t
            the ligand of interest
        response : dict[gene_t, int]
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
        if not isinstance(ligand, gene_t):
            raise TypeError(f"ligand should have type gene_t, was {type(ligand)}")
        if type(response) is not dict:
            raise TypeError(f"response should have type dict, was {type(response)}")
        # create the prediction model vector
        prediction = dict(zip(self.row_names, self.get_col(ligand, is_index=False)))
        # we need to match the predictions with the responses so we intersect and sort by key
        common_keys = prediction.keys() & response.keys()
        pred = np.array([tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])])
        resp = np.array([tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])])
        return calculate_prediction_evaluation_metrics(pred, resp)

    def predict_ligand_activities(
        self,
        geneset:Collection[gene_t],
        background_expressed_genes:Iterable[gene_t],
        potential_ligands:Iterable[gene_t]
    ):
        '''
        Predict activities of ligands in regulating expression of a gene set of interest.
        Ligand activities are defined as how well they predict the observed transcriptional response (i.e. gene set) according
        to the NicheNet model.

        Parameters
        ----------
        geneset : Collection of gene_t
            the genes of which the expression is potentially affected by ligands from the interacting cell
        background_expressed_genes : Iterable of gene_t
            the background, non-affected, genes (can contain the symbols of the affected genes as well)
        potential_ligands : Iterable of gene_t
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
        if len(geneset) < 20 or len(geneset) > 2000:
            warnings.warn(f"It is recommended that the size of the gene set of interest is between 20 and 2000, was {len(geneset)}")
        if len(background_expressed_genes) < 10 * len(geneset):
            warnings.warn(f"The background set is small compared to the geneset of interest, a background size of at least {10 * len(geneset)} is recommended, was {len(background_expressed_genes)}")
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
        expression_scaled_cols:Iterable[gene_t],
        potential_ligands:Collection[gene_t],
        quantile_cutoff:float=0.975,
        calc_aupr:bool=True,
        calc_auroc:bool=True,
        calc_pearson:bool=True
    ):
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
        expression_scaled_cols : Iterable of gene_t
            the names of the columns of expression_scaled
        potential_ligands : Collection of gene_t
            the genes of the potentially active ligands for which you want to define ligand activities
        quantile_cutoff : float
            the cutoff value used to compute the response vector
        calc_aupr : bool
            whether or not to compute the aupr score
        calc_auroc : bool
            whether or not to compute the auroc score
        calc_pearson : bool
            whether or not to compute the pearson score

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
            raise TypeError(f"expression_scaled_cols should have type Iterable[gene_t], was {type(expression_scaled_cols)}")
        if not isinstance(potential_ligands, Collection):
            raise TypeError(f"potential_ligands should have type Collection[gene_t], was {type(potential_ligands)}")
        if not isinstance(quantile_cutoff, Number):
            raise TypeError(f"quantile_cutoff should have type float, was {type(quantile_cutoff)}")
        if type(calc_aupr) is not bool:
            raise TypeError(f"calc_aupr should have type bool, was {type(calc_aupr)}")
        if type(calc_auroc) is not bool:
            raise TypeError(f"calc_auroc should have type bool, was {type(calc_auroc)}")
        if type(calc_pearson) is not bool:
            raise TypeError(f"calc_pearson should have type bool, was {type(calc_pearson)}")
        row2id = dict(zip(expression_scaled_rows, range(len(expression_scaled_rows))))
        if calc_aupr:
            aupr = []
            aupr_corrected = []
        if calc_auroc:
            auroc = []
        if calc_pearson:
            pearson = []
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
                prediction = dict(zip(self.row_names, self.get_col(ligand, is_index=False)))
                common_keys = prediction.keys() & response.keys()
                pred = np.array([tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])])
                resp = np.array([tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])])
                if calc_aupr:
                    aupr.append(calculate_aupr(resp, pred))
                    aupr_corrected.append(aupr[-1] - sum(resp)/len(resp))
                if calc_auroc:
                    auroc.append(calculate_auroc(resp, pred))
                if calc_pearson:
                    pearson.append(pearsonr(resp, pred).statistic)
        output = {
            "cell": chain(*(repeat(cell, len(potential_ligands)) for cell in cells)),
            "ligand": chain(*repeat(potential_ligands, len(cells)))
        }
        if calc_aupr:
            output["aupr"] = aupr
            output["aupr_corrected"] = aupr_corrected
        if calc_auroc:
            output["auroc"] = auroc
        if calc_pearson:
            output["pearson"] = pearson
        return pd.DataFrame(output)
    
    def get_weighted_ligand_target_links(
        self,
        ligand:gene_t,
        geneset:Iterable[gene_t],
        n:int=250
    ) -> dict:
        '''
        Infer active ligand target links between possible ligands and genes belonging to a gene set of interest: consider the intersect between the top n targets of a ligand and the gene set.

        Parameters
        ----------
        ligand : gene_t
            the gene symbol of the potentially active ligand for which you want to find target genes
        geneset : Iterable of gene_t
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
        if not isinstance(ligand, gene_t):
            raise TypeError(f"ligand should have type gene_t, was {type(ligand)}")
        if not isinstance(geneset, Iterable):
            raise TypeError(f"geneset should have type Iterable, was {type(geneset)}")
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
    
    def replace_zero_col_by_noisy_scores(self):
        '''
        Replace zero columns with a very low noisy random score. 
        '''
        m = min(e for r in self.ligand_target_matrix for e in r if e > 0)
        for j in range(self.ligand_target_matrix.shape[1]):
            if sum(self.ligand_target_matrix[:, j]) == 0:
                for i in range(self.ligand_target_matrix.shape[0]):
                    self.ligand_target_matrix[i, j] = uniform(0, m)
    
    def gene_presence(self, genes:Iterable[gene_t]) -> float:
        '''
        calculate the ratio of genes that are present in the ligand-target matrix

        Parameters
        ----------
        genes : Iterable of gene_t
            the genes to check for

        Returns
        -------
        float
            the ratio of genes that are present in the ligand-target matrix
        '''
        return len(self.get_genes().intersection(genes))/len(genes)
    
    def ligand_presence(self, ligands:Iterable[gene_t]) -> float:
        '''
        calculate the ratio of ligands that are present in the ligand-target matrix

        Parameters
        ----------
        ligands : Iterable of gene_t
            the ligands to check for

        Returns
        -------
        float
            the ratio of ligands that are present in the ligand-target matrix
        '''
        return len(self.get_ligands().intersection(ligands))/len(ligands)

def assess_rf_class_probabilities(
    folds:int,
    geneset:set[gene_t],
    background_expressed_genes:set[gene_t],
    ligands_oi:Iterable[gene_t],
    predictor:LigandActivityPredictor,
    ntrees:int=1000
):
    '''
    Assess probability that a target gene belongs to the geneset based on a multi-ligand random forest model (with cross-validation).
    Target genes and background genes will be split in different groups in a stratified way.

    Parameters
    ----------
    folds : int
        how many folds should be used
    geneset : set of gene_t
        the genes for which the expression is potentially affected by ligands from the interacting cell
    background_expressed_genes : set of gene_t
        the background, non-affected, genes (can contain the symbols of the affected genes as well)
    ligands_oi : Iterable of gene_t
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
        raise TypeError(f"geneset should have type set[gene_t], was {type(geneset)}")
    if type(background_expressed_genes) is not set:
        raise TypeError(f"background_expressed_genes should have type set[gene_t], was {type(background_expressed_genes)}")
    if not isinstance(ligands_oi, Iterable):
        raise TypeError(f"ligands_oi should have type Iterable[gene_t], was {type(ligands_oi)}")
    if not isinstance(predictor, LigandActivityPredictor):
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
                ((geneset[id][0], 1) for id in geneset_train),
                ((background_expressed_genes[id][0], 0) for id in beg_train)
            )
        )
        pred_mat = subset_matrix(
            predictor.ligand_target_matrix,
            rows=[predictor.gene2index(gene) for gene in row_names],
            cols=[predictor.ligand2index(ligand) for ligand in ligands_oi]
        )
        # random forest trained on some genes
        # predicts if the gene is differentially expressed based on the regulatory potential scores of the ligands of interest
        rf = RandomForestClassifier(n_estimators=ntrees)
        rf.fit(X=pred_mat, y=res)
        row_names, res = zip(
            *chain(
                ((geneset[id][0], 1) for id in geneset_test),
                ((background_expressed_genes[id][0], 0) for id in beg_test)
            )
        )
        pred_mat = subset_matrix(
            predictor.ligand_target_matrix,
            rows=[predictor.gene2index(gene) for gene in row_names],
            cols=[predictor.ligand2index(ligand) for ligand in ligands_oi]
        )
        # prediction for the remaining genes
        pred = rf.apply(pred_mat)
        # the amount of trees that predict that the gene is differentially expressed (for each gene)
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
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.metrics import calculate_prediction_evaluation_metrics
from nichenetpy.utils import is_ligand_active
from nichenetpy.typing import gene_t

from collections.abc import (
    Iterable,
    ItemsView
)
from itertools import repeat
from re import search
from warnings import warn

import pandas as pd


class EvaluationData:
    '''
    Data which can be used for model evaluation. Each item needs to contain

        * a ligand

        * the genes that were regulated by the ligand

    Parameters
    ----------
    obj : dict or Iterable of dict or pandas.DataFrame
        Iterable of a ligand with it's corresponding genes and optionally a name/key for the data element. 
        Needs to contain at least a mapping for `key_name`, `ligand_name` and `de_genes_name`
    key_name : string
        the name of the key field in the data elements
    ligand_name : string
        the name of the ligand in the data elements
    de_genes_name : string
        the name of the target genes in the data elements

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    
    Attributes
    ----------
    _data : dict
        the data elements
    _key_name : string
        the name of the key field in the data elements
    _ligand_name : string
        the name of the ligand in the data elements
    _de_genes_name : string
        the name of the target genes in the data elements
    '''
    def __init__(
        self,
        obj:dict[str, dict]|Iterable[dict]|ItemsView|pd.DataFrame|None=None,
        key_name:str="name",
        ligand_name:str="from",
        de_genes_name:str="response"
    ):
        self._data = dict()
        self._key_name = key_name
        self._ligand_name = ligand_name
        self._de_genes_name = de_genes_name
        if obj is not None:
            if type(obj) is dict:
                for key, val in obj.items():
                    self[key] = val
            elif type(obj) is pd.DataFrame:
                for key, val in zip(obj[key_name], (dict(zip(obj.columns, row)) for row in zip(*(obj[col] for col in obj.columns)))):
                    self[key] = val
            elif isinstance(obj, ItemsView):
                for key, val in obj:
                    self[key] = val
            elif isinstance(obj, Iterable):
                for item in obj:
                    self.add(item)
            else:
                raise TypeError(f"obj should have type dict, pandas.DataFrame, ItemsView or Iterable, was {type(obj)}")

    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key:str, val:dict):
        if type(key) is not str:
            raise TypeError(f"key should have type str, was {type(key)}")
        if type(val) is not dict:
            raise TypeError(f"val should have type dict, was {type(val)}")
        if self._ligand_name not in val:
            raise ValueError(f"item should have a '{self._ligand_name}' key")
        if type(val[self._ligand_name]) is not str and type(val[self._ligand_name]) is not list:
            raise ValueError(f"'{self._ligand_name}' does not map to a string or list of strings")
        if self._de_genes_name not in val:
            raise ValueError(f"item should have a '{self._de_genes_name}' key")
        if type(val[self._de_genes_name]) is not dict:
            raise ValueError(f"'{self._de_genes_name}' does not map to a dict")
        val[self._key_name] = key
        self._data[key] = val

    def add(self, item:dict):
        '''
        Add a dictionary which maps the following keys

            * `key_name` -> the name of the data element

            * `ligand_name` -> the ligand

            * `de_genes_name` -> the target genes

        Parameters
        ----------
        item : dict
            the data element to add
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        ValueError
            if the arguments are invalid
        '''
        if type(item) is not dict:
            raise TypeError(f"item should have type dict, was {type(item)}")
        if self._key_name not in item:
            raise ValueError(f"item should have a '{self._key_name}' key")
        if type(item[self._key_name]) is not str:
            raise ValueError(f"'{self._key_name}' does not map to a string")
        if self._ligand_name not in item:
            raise ValueError(f"item should have a '{self._ligand_name}' key")
        if not isinstance(item[self._ligand_name], gene_t) and type(item[self._ligand_name]) is not list:
            raise ValueError(f"'{self._ligand_name}' does not map to a gene symbol or list")
        if self._de_genes_name not in item:
            raise ValueError(f"item should have a '{self._de_genes_name}' key")
        if type(item[self._de_genes_name]) is not dict:
            raise ValueError(f"'{self._de_genes_name}' does not map to a dict")
        self._data[item[self._key_name]] = item
    
    def add_all(self, items:Iterable[dict]):
        '''
        Adds dictionaries which map the following keys

            * `key_name` -> the name of the data element

            * `ligand_name` -> the ligand

            * `de_genes_name` -> the target genes

        Parameters
        ----------
        item : Iterable of dict
            the data elements to add
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        ValueError
            if the arguments are invalid
        '''
        for item in items:
            self.add(item)
    
    def __contains__(self, key):
        return key in self._data
    
    def __iter__(self):
        return iter(self._data.values())
    
    def __len__(self):
        return len(self._data)
    
    def keys(self):
        '''
        returns the keys of the data elements
        
        Returns
        -------
        Iterable
            the keys of the data elements
        '''
        return self._data.keys()
    
    def values(self):
        '''
        returns the data elements
        
        Returns
        -------
        Iterable
            the data elements
        '''
        return self._data.values()
    
    def items(self):
        '''
        returns the data elements along with the corresponding keys
        
        Returns
        -------
        Iterable of tuple
            (key, value) tuples where the values are the data elements
        '''
        return self._data.items()
    
    def get_ligands(
        self,
        combination:bool=True
    ):
        '''
        Extract all ligands from the evaluation data. 

        Parameters
        ----------
        combination : bool
            whether to include combinations of ligands in the output

        Returns
        -------
        set
            a set which contains all ligands present in the evaluation data
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        '''
        if type(combination) is not bool:
            raise TypeError(f"combination should have type bool, was {type(combination)}")
        output = set()
        for setting in self.values():
            if isinstance(setting["from"], gene_t):
                output.add(setting["from"])
            else:
                if combination:
                    output.add(tuple(setting["from"]))
                for ligand in setting["from"]:
                    output.add(ligand)
        return output
    
    def to_dataframe(self) -> pd.DataFrame:
        '''
        Convert the evaluation data to a pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            the evaluation data as a dataframe
        '''
        keys = set(next(iter(self._data.values())).keys())
        columns_dct = {
            k: []
            for k in keys
        }
        for v in self._data.values():
            for k in keys:
                columns_dct[k].append(v[k])
        return pd.DataFrame(columns_dct)
    
    def get_applicable_evaluation_datasets(
        self,
        predictor:LigandActivityPredictor,
        combination:bool=True
    ):
        '''
        get the subset of applicable evaluation data
        (evaluation data where the ligand is present in the ligand-target matrix and there is at least one true sample for a gene that is present in the ligand-target matrix)

        Parameters
        ----------
        predictor : LigandActivityPredictor
            the ligand activity predictor
        combination : bool
            whether or not to allow combinations of ligands

        Yields
        -------
        str
            the key of the dataset
        dict
            the applicable dataset
        '''
        if not isinstance(predictor, LigandActivityPredictor):
            raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
        ligands = predictor.get_ligands()
        pred_genes = predictor.get_genes()
        for k, gs in self._data.items():
            ligand = gs["from"]
            if type(ligand) is list or type(ligand) is tuple:
                ligand = ("-".join(ligand) if type(ligand[0]) is str else ligand) if combination else None
            if ligand is not None and ligand in ligands:
                res = iter(gs[self._de_genes_name].items())
                is_app = False
                try:
                    while not is_app:
                        gene, val = next(res)
                        is_app = gene in pred_genes and val
                except StopIteration:
                    pass
                if is_app:
                    yield (k, gs)

_metrics = ("aupr", "aupr_corrected", "auroc", "pearson")

def get_single_ligand_importances(
    predictor:LigandActivityPredictor,
    evaluation_data:Iterable[dict],
    all_ligands:Iterable[gene_t]
) -> pd.DataFrame:
    '''
    Get ligand importance measures for ligands based on how well a single, individual, ligand can predict
    an observed response. Assess how well every ligand of interest is able to predict the observed transcriptional
    response in a particular dataset, according to the ligand-target model. It can be assumed that the ligand
    that best predicts the observed response, is more likely to be the true ligand.

    Parameters
    ----------
    predictor : LigandActivityPredictor
        the ligand activity predictor
    evaluation_data : dict
        An Iterable of dictionaries that have the following keys: 
        
            name: the name of the setting

            ligand: the name of the ligand which is known to be active in the setting of interest
        
            from:  the name of the ligand of which the predictive performance need to be assessed
        
            response:   the observed target response, indicates for a gene whether it was a target
                        or not in the setting of interest
    all_ligands : Iterable of gene_t
        the possible ligands that will be considered for the ligand activity state prediction

    Returns
    -------
    pandas.DataFrame
        A data frame with for each ligand - data set combination, classification evaluation metrics indicating
        how well the query ligand predicts the response in the particular dataset. Evaluation metrics are
        the same as in evaluate_target_prediction. In addition to the metrics, the name of the particular setting,
        the name of the query ligand(test_ligand), the name of the true active ligand (if known: ligand).
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if not isinstance(predictor, LigandActivityPredictor):
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if not isinstance(evaluation_data, Iterable):
        raise TypeError(f"evaluation_data should have type Iterable[dict], was {type(evaluation_data)}")
    # compute metrics for each ligand/dataset combination
    ligand_importances = pd.DataFrame(
        dict(zip(
            _metrics,
            zip(*(
                list(zip(*sorted(
                    predictor.evaluate_target_prediction(ligand, setting["response"]).items(),
                    key=lambda x : x[0]
                )))[1]
                for setting in evaluation_data for ligand in all_ligands
            ))
        ))
    )
    ligand_importances["setting"], ligand_importances["test_ligand"], ligand_importances["true_ligand"] = (
        zip(*((setting["name"], ligand, setting["from"]) for setting in evaluation_data for ligand in all_ligands))
    )
    return ligand_importances

def evaluate_single_importances_ligand_prediction(
    importances:pd.DataFrame,
    group:str,
    allow_nan:bool=False
) -> pd.DataFrame:
    '''
    Evaluate how well a single ligand importance metric is able to predict the true activity state of a ligand.
    For this it is assumed, that ligand importance measures for truely active ligands will be higher than for
    non-active ligands. Several classification evaluation metrics for the prediction are calculated and variable
    importance scores can be extracted to rank the different importance measures in order of importance for ligand
    activity state prediction.

    Parameters
    ----------
    importances : pandas.DataFrame
        A data frame containing at least folowing variables: setting, test_ligand, ligand and one or more feature importance scores.
        test_ligand denotes the name of a possibly active ligand, true_ligand the name of the truely active ligand.
    group : str
        the setting of interest
    allow_nan : bool
        if True, return nan values in case a specific metric is undefined, if False the errors are not caught

    Returns
    -------
    pandas.DataFrame
        A data frame containing classification evaluation metrics for the ligand activity state prediction.
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(importances) is not pd.DataFrame:
        raise TypeError(f"importances should have type pandas.DataFrame, was {type(importances)}")
    if type(group) is not str:
        raise TypeError(f"group should have type str, was {type(group)}")
    importances = importances[importances["setting"] == group]
    added = is_ligand_active(importances)
    # use ligand importances as prediction (each metric in turn) and true ligand as response
    output = pd.DataFrame(
        dict(zip(
            _metrics,
            zip(*(
                list(zip(*sorted(
                    calculate_prediction_evaluation_metrics(list(importances[metric]), added, allow_nan=allow_nan).items(),
                    key=lambda x : x[0]
                )))[1]
                for metric in _metrics
            ))
        ))
    )
    output["group"] = list(repeat(group, len(_metrics)))
    output["ligand"] = list(repeat(importances["true_ligand"].iloc[1], len(_metrics)))
    output["metric"] = _metrics
    return output.reindex(["metric", "group", "ligand", "aupr", "aupr_corrected", "auroc", "pearson"], axis=1)
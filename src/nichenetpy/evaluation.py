from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.metrics import calculate_prediction_evaluation_metrics
from nichenetpy.utils import is_ligand_active

from collections.abc import Iterable
from itertools import repeat

import pandas as pd


class EvaluationData:
    '''
    Data which can be used for model evaluation. Each item needs to contain
        - a ligand
        - the genes that were regulated by the ligand

    Parameters
    ----------
    obj : Iterable of dict
        iterable of a ligand with it's corresponding genes and optionally a name/key for the data element
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
        obj:dict[str, dict]|Iterable[dict]=None,
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
            elif isinstance(obj, Iterable):
                for item in obj:
                    self.add(item)

    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key:str, val:dict):
        if type(key) is not str:
            raise TypeError(f"key should have type str, was {type(key)}")
        if type(val) is not dict:
            raise TypeError(f"val should have type dict, was {type(val)}")
        if self._ligand_name not in val:
            raise ValueError(f"item should have a {self._ligand_name} key")
        if type(val[self._ligand_name]) is not str:
            raise ValueError(f"'{self._ligand_name}' does not map to a string")
        if self._de_genes_name not in val:
            raise ValueError(f"item should have a {self._de_genes_name} key")
        if type(val[self._de_genes_name]) is not dict:
            raise ValueError(f"'{self._de_genes_name}' does not map to a dict")
        val[self._key_name] = key
        self._data[key] = val

    def add(self, item:dict):
        '''
        Add a dictionary which maps the following keys
            - _key_name -> the name of the data element
            - _ligand_name -> the ligand
            - _de_genes_name -> the target genes

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
            raise ValueError(f"item should have a {self._key_name} key")
        if type(item[self._key_name]) is not str:
            raise ValueError(f"'{self._key_name}' does not map to a string")
        if self._ligand_name not in item:
            raise ValueError(f"item should have a {self._ligand_name} key")
        if type(item[self._ligand_name]) is not str:
            raise ValueError(f"'{self._ligand_name}' does not map to a string")
        if self._de_genes_name not in item:
            raise ValueError(f"item should have a {self._de_genes_name} key")
        if type(item[self._de_genes_name]) is not dict:
            raise ValueError(f"'{self._de_genes_name}' does not map to a dict")
        self._data[item[self._key_name]] = item
    
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

def get_single_ligand_importances(
    predictor:LigandActivityPredictor,
    evaluation_data:Iterable[dict],
    all_ligands:Iterable[str]
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
    all_ligands : Iterable of str
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
    if type(predictor) is not LigandActivityPredictor:
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if not isinstance(evaluation_data, Iterable):
        raise TypeError(f"evaluation_data should have type Iterable[dict], was {type(evaluation_data)}")
    # compute metrics for multiple prediction/response pairs and store them in a dataframe
    ligand_importances = pd.DataFrame(
        dict(zip(
            ("aupr", "aupr_corrected", "auroc", "pearson"),
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
    group:str
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
    metrics = ("aupr", "aupr_corrected", "auroc", "pearson")
    added = is_ligand_active(importances)
    # compute metrics for multiple prediction/response pairs and store them in a dataframe
    output = pd.DataFrame(
        dict(zip(
            metrics,
            zip(*(
                list(zip(*sorted(
                    calculate_prediction_evaluation_metrics(list(importances[metric]), added).items(),
                    key=lambda x : x[0]
                )))[1]
                for metric in metrics
            ))
        ))
    )
    output["group"] = list(repeat(group, len(metrics)))
    output["ligand"] = list(repeat(importances["true_ligand"].iloc[1], len(metrics)))
    output["metric"] = metrics
    return output.reindex(["metric", "group", "ligand", "aupr", "aupr_corrected", "auroc", "pearson"], axis=1)
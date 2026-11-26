# type: ignore

"""
Here we test the EvaluationData class.
"""

from collections import OrderedDict

from nichenetpy.evaluation import EvaluationData


def test_empty_evaluation_data_to_dataframe():
    # no rows, the key, ligand and DE genes columns
    df = EvaluationData().to_dataframe()
    assert len(df) == 0
    assert df.columns.tolist() == ["name", "from", "response"]
    df = EvaluationData(key_name="key", ligand_name="ligand", de_genes_name="de").to_dataframe()
    assert df.columns.tolist() == ["key", "ligand", "de"]


def test_evaluation_data_dict_subclass():
    # a dict subclass (e.g. an OrderedDict) is accepted as a dict everywhere
    setting = OrderedDict({"name": "s1", "from": "L1", "response": OrderedDict({"g0": 1, "g1": 0})})
    expected = EvaluationData([{"name": "s1", "from": "L1", "response": {"g0": 1, "g1": 0}}]).to_dataframe()
    for data in (
        EvaluationData(OrderedDict({"s1": setting})),
        EvaluationData([setting]),
    ):
        assert data.to_dataframe().to_dict() == expected.to_dict()
    data = EvaluationData()
    data["s1"] = setting
    assert data.to_dataframe().to_dict() == expected.to_dict()
    data = EvaluationData()
    data.add(setting)
    assert data.to_dataframe().to_dict() == expected.to_dict()

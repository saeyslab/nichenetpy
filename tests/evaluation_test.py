# type: ignore

"""
Here we test the EvaluationData class.
"""

from nichenetpy.evaluation import EvaluationData


def test_empty_evaluation_data_to_dataframe():
    # no rows, the key, ligand and DE genes columns
    df = EvaluationData().to_dataframe()
    assert len(df) == 0
    assert df.columns.tolist() == ["name", "from", "response"]
    df = EvaluationData(key_name="key", ligand_name="ligand", de_genes_name="de").to_dataframe()
    assert df.columns.tolist() == ["key", "ligand", "de"]

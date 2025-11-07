from nichenetpy.visualization import (
    _construct_ligand_signaling_df,
    get_ligand_signaling_path
)

import pandas as pd

from common import (
    equals_iter,
    get_model_pickle,
    get_ltf_matrix,
)


ligand_signaling_df_0 = [
    ("IL1B", 0.2114539724, "IL1B", "TNFRSF11B", 1.3534754, 0.2861977536),
    ("ESR1", 0.0008947402, "IL1B", "TNFRSF11B", 0.7991201, 0.0007150048),
    ("RELA", 0.0017290112, "IL1B", "TNFRSF11B", 0.2799552, 0.0004840456),
    ("STAT1", 0.0016298949, "IL1B", "TNFRSF11B", 0.2895576, 0.0004719485),
    ("IL1B", 0.2114539724, "IL1B", "CSF1", 1.2809936, 0.2708711828),
    ("RELA", 0.0017290112, "IL1B", "CSF1", 1.1674674, 0.0020185642),
    ("STAT1", 0.0016298949, "IL1B", "CSF1", 0.9315365, 0.0015183067),
    ("NFKB1", 0.0019036851, "IL1B", "CSF1", 0.7667786, 0.0014597049),
    ("IL1B", 0.2114539724, "IL1B", "NFKB2", 1.2431144, 0.2628614787),
    ("RELA", 0.0017290112, "IL1B", "NFKB2", 1.8179507, 0.0031432571),
    ("NFKB1", 0.0019036851, "IL1B", "NFKB2", 0.7991353, 0.0015213020),
    ("JUN", 0.0010959084, "IL1B", "NFKB2", 1.2643184, 0.0013855770)
]

tf_regulatory_0 = [
    ("ESR1", "TNFRSF11B", 1.1609087),
    ("IL1B", "CSF1", 1.4395296),
    ("IL1B", "NFKB2", 1.4176277),
    ("IL1B", "TNFRSF11B", 1.4814388),
    ("JUN", "CSF1", 0.9110804),
    ("JUN", "NFKB2", 1.4298879),
    ("JUN", "TNFRSF11B", 0.8768344),
    ("NFKB1", "CSF1", 1.1422087),
    ("NFKB1", "NFKB2", 1.1609175),
    ("RELA", "CSF1", 1.3738883),
    ("RELA", "NFKB2", 1.7500000),
    ("RELA", "TNFRSF11B", 0.8607258),
    ("STAT1", "CSF1", 1.2374723),
    ("STAT1", "NFKB2", 0.7500000),
    ("STAT1", "TNFRSF11B", 0.8662780),
]

tf_signaling_0 = [
    ("ESR1", "JUN", 1.4795796),
    ("ESR1", "MAPK1", 1.0675241),
    ("ESR1", "NFKB1", 1.2431740),
    ("ESR1", "RELA", 1.2354086),
    ("ESR1", "STAT1", 0.8828321),
    ("IL1B", "ESR1", 0.8860990),
    ("IL1B", "JUN", 0.9884542),
    ("IL1B", "MAPK1", 1.0168572),
    ("IL1B", "MAPK8", 1.2375072),
    ("IL1B", "NFKB1", 1.4873439),
    ("IL1B", "RELA", 1.5401602),
    ("IL1B", "STAT1", 1.3739715),
    ("JUN", "ESR1", 1.0618713),
    ("JUN", "MAPK1", 1.2637184),
    ("MAPK1", "ESR1", 1.7452258),
    ("MAPK1", "JUN", 1.7041166),
    ("MAPK1", "MAPK8", 1.3174674),
    ("MAPK1", "NFKB1", 1.0314593),
    ("MAPK1", "RELA", 1.3338606),
    ("MAPK1", "STAT1", 1.5579205),
    ("MAPK8", "ESR1", 0.8468483),
    ("MAPK8", "JUN", 1.7500000),
    ("MAPK8", "MAPK1", 1.5247501),
    ("MAPK8", "RELA", 0.7500000),
    ("MAPK8", "STAT1", 1.3865783),
    ("NFKB1", "JUN", 0.8517341),
    ("NFKB1", "MAPK1", 0.8616534),
    ("NFKB1", "MAPK8", 0.8903254),
    ("NFKB1", "RELA", 1.6845898),
    ("NFKB1", "STAT1", 0.8764611),
    ("RELA", "JUN", 1.0652949),
    ("RELA", "MAPK1", 0.7882357),
    ("RELA", "NFKB1", 1.3088079),
    ("STAT1", "ESR1", 0.8459594),
    ("STAT1", "JUN", 0.9047243),
    ("STAT1", "MAPK1", 1.1774413),
    ("STAT1", "RELA", 1.1152577),
]

def construct_ligand_signaling_df_template(ligands_oi, targets_oi, res):
    model = get_model_pickle("human")
    gr = pd.DataFrame(model["gr"]._mapping, columns=["from", "to", "weight"])
    ltf_matrix = get_ltf_matrix("human")
    row_names = ltf_matrix["row_names"]
    col_names = ltf_matrix["col_names"]
    ltf_matrix = ltf_matrix["mat"]
    combined_df = _construct_ligand_signaling_df(
        ligands_oi,
        targets_oi,
        row_names,
        col_names,
        gr,
        ltf_matrix,
        4
    )
    assert equals_iter(
        combined_df.to_numpy(),
        res
    )

def test_construct_ligand_signaling_df_0():
    construct_ligand_signaling_df_template(
        ["IL1B"],
        ["TNFRSF11B", "CSF1", "NFKB2"],
        ligand_signaling_df_0
    )

def get_ligand_signaling_path_template(ligands_oi, targets_oi, exp_tf_signaling, exp_tf_regulatory):
    model = get_model_pickle("human")
    lr_sig = pd.DataFrame(model["lr_sig"]._mapping, columns=["from", "to", "weight"])
    gr = pd.DataFrame(model["gr"]._mapping, columns=["from", "to", "weight"])
    ltf_matrix = get_ltf_matrix("human")
    row_names = ltf_matrix["row_names"]
    col_names = ltf_matrix["col_names"]
    ltf_matrix = ltf_matrix["mat"]
    tf_signaling, tf_regulatory = get_ligand_signaling_path(
        ltf_matrix,
        ligands_oi,
        targets_oi,
        row_names,
        col_names,
        lr_sig,
        gr,
        minmax_scaling=True
    )
    assert equals_iter(
        tf_signaling.to_numpy(),
        exp_tf_signaling
    )
    assert equals_iter(
        tf_regulatory.to_numpy(),
        exp_tf_regulatory
    )

def test_construct_ligand_signaling_df_0():
    get_ligand_signaling_path_template(
        ["IL1B"],
        ["TNFRSF11B", "CSF1", "NFKB2"],
        tf_signaling_0,
        tf_regulatory_0
    )
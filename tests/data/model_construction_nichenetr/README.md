# Model construction vs nichenetr

Reference values for the personalized PageRank (PPR) cases of `test_model_construction` in
`tests/full_nichenet_test.py`, which checks that `model_construction.construct_ligand_target_matrix` gives the same
ligand-target matrix as nichenetr's `construct_ligand_target_matrix` on the real human networks.

> **Note:** `generate_reference.R` is not run by the test suite. The reference values are written in the test itself
> (`MC_top_10_ligand_target_matrix_PPR_0`, `_1` and `_2`), so R and nichenetr are not needed to run it. The script
> documents where the values come from and is only run by hand, to regenerate them after changing the data or the
> settings.

| File | Content |
|---|---|
| `generate_reference.R` | computes the reference values with nichenetr and prints them as the Python lists in the test |

The networks aren't in this folder: they're the files the test downloads from Zenodo (record 20395877) into
`tests/data/tutorial_files/model_construction`.

`generate_reference.R` builds the weighted networks (hub corrections 0.115 for the ligand-signaling network and
0.0803 for the gene regulatory network) and the ligand-target matrix with PPR (`damping_factor` 0.789, `ltf_cutoff`
0.926) for the three cases of the test, and prints the first 10 targets:

- `PPR_0`: all data sources with weight 1, the ligand TNF and the combination TNF-IL6;
- `PPR_1`: the optimized source weights (`avg_weight` of `optimized_source_weights.csv`), TNF;
- `PPR_2`: only the data sources of type literature, comprehensive_db and ChIP (`annotation_data_sources.csv`) with
  weight 1, TNF.

## Regenerating the reference

Run from this directory, after the test has downloaded the networks:

    Rscript generate_reference.R

or pass the folder with the network files:

    Rscript generate_reference.R path/to/model_construction

Then copy the printed lists into `tests/full_nichenet_test.py`. The current values were generated with R 4.5.1,
nichenetr 2.2.1.1 and igraph 2.1.4.

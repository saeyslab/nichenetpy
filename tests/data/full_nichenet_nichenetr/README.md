# Full NicheNet analyses vs nichenetr

Reference values for `test_steps`, `test_wrapper`, `test_steps_prioritization` and
`test_target_prediction_evaluation_geneset` in `tests/full_nichenet_test.py`. These tests mirror the notebooks `steps`,
`wrapper`, `steps_prioritization` and `target_prediction_evaluation_geneset` in `docs/notebooks`, and check that they
give the same results as nichenetr on the same data.

> **Note:** `generate_reference.R` is not run by the test suite. The reference values are written in the test itself
> (the `BASIC_*`, `PRIOR_*` and `TPEG_*` constants and the values in the asserts), so R and nichenetr are not needed to
> run it. The script documents where the values come from and is only run by hand, to regenerate them after changing
> the data or the settings.

| File | Content |
|---|---|
| `generate_reference.R` | computes the reference values with nichenetr and prints them as the Python code in the test |

The data isn't in this folder (about 800 MB). Download these files into a directory of your choice:

| File | Zenodo record |
|---|---|
| `seuratObj.rds` | [3531889](https://zenodo.org/records/3531889) (the same data as `annData3531889.h5`, which the tests use) |
| `ligand_target_matrix_nsga2r_final_mouse.rds`, `lr_network_mouse_21122021.rds`, `weighted_networks_nsga2r_final_mouse.rds` | [7074291](https://zenodo.org/records/7074291) (the nichenetr v2 mouse model, the same model as `nichenet_mouse.pkl`) |

As the notebooks, the script first converts gene aliases to gene symbols. nichenetr's `alias_to_symbol_seurat` returns
Seurat v5 objects unchanged, so the script renames the expression matrices the same way (with
`convert_alias_to_symbols`; names that would become the same symbol keep their original name) and creates a new Seurat
object. This gives exactly the same gene names as `GeneAliasInfo.alias_to_symbol` with `geneinfo_alias_mouse.csv` (1350
of the 13541 genes are renamed).

Then it follows the steps of the tests:

- `test_steps` / `test_wrapper`: expressed genes (`pct = 0.05`), potential ligands, the gene set of interest (CD8 T, LCMV
  vs SS, `FindMarkers` with `min.pct = 0.05`, adjusted p-value <= 0.05 and |log2 fold change| >= 0.25), ligand
  activities, the ligand-target links (`n = 100`) and ligand-receptor links of the top 30 ligands, all ligands and only
  the ligands expressed by the senders, and the log fold changes of the top 30 ligands in the senders;
- `test_steps_prioritization`: `generate_info_tables` and `generate_prioritization_tables` for CD8 T, and for the
  receivers CD8 T, CD4 T and Treg combined;
- `test_target_prediction_evaluation_geneset`: 2 rounds of `assess_rf_class_probabilities` with 3 folds and their
  evaluation, with `set.seed(1)` (the test allows for variance, as the Python random forests differ).

## Regenerating the reference

Run from this directory:

    Rscript generate_reference.R <data directory>

Then copy the printed values into `tests/full_nichenet_test.py`. With `Rscript generate_reference.R <data directory>
FALSE` the gene aliases aren't converted; this reproduces the values the tests had before the conversion was added (a
check that R and Python start from the same data). The current values were generated with R 4.5.1, nichenetr 2.2.1.1
and Seurat 5.3.0.

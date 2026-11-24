# Gene set of interest vs nichenetr

Reference values for `test_run_nichenet_geneset_oi_same_as_nichenetr` in `tests/get_geneset_oi_test.py`, which checks
that `wrappers.run_nichenet` computes the same gene set of interest as nichenetr's `nichenet_seuratobj_aggregate`: the
receiver's condition of interest compared with the reference condition.

> **Note:** `generate_reference.R` is not run by the test suite. The reference values are written in the test itself
> (`NICHENETR_GENESET_OI`), so R and nichenetr are not needed to run it. The script documents where the values come
> from and is only run by hand, to regenerate them after changing the data or the settings.

| File | Content |
|---|---|
| `generate_reference.R` | computes the reference gene sets with nichenetr and prints them as the Python dict in the test |

The data isn't in this folder: the script uses the data of `../group_metrics_seurat` (40 genes, 180 cells), with all
cells as one receiver cell type and the groups A, B and C as conditions. Three conditions are needed: with two, the
reference condition is the same as all other conditions.

`generate_reference.R` runs `nichenet_seuratobj_aggregate` (with `sender = "undefined"`) for the condition pairs A vs B,
A vs C, B vs C and C vs A, each with `expression_pct` / `lfc_cutoff` 0.1 / 0.25, 0.3 / 0.25 and 0.1 / 0.5, and prints
the `geneset_oi` of each run. It uses a small dummy model: the ligands gene30 to gene34, their receptors gene8, gene28,
gene38, gene5 and gene20, and a ligand-target matrix computed from a formula, so the test can build the same model. The
gene set only depends on the model through its genes (all 40 genes).

## Regenerating the reference

Run from this directory:

    Rscript generate_reference.R

Then copy the printed dict into `tests/get_geneset_oi_test.py`. The current values were generated with R 4.5.1,
nichenetr 2.2.1.1 and Seurat 5.3.0.

# Wilcoxon limma vs R

Reference data for `tests/wilcoxon_limma_test.py`, which checks that `metrics.group_metrics(wilcoxon_limma=True)`
gives the same p-values as limma, the test behind Seurat's `FindMarkers(test.use = "wilcox_limma")`.

> **Note:** `generate_reference.R` is not run by the test suite. The tests only read `pvals_limma.csv`, so R and
> limma are not needed to run them. The script documents where the reference values come from and is only run by
> hand, to regenerate `pvals_limma.csv` after changing the data.

| File | Content |
|---|---|
| `data.csv` | dummy expression data, genes x cells (25 genes, 90 cells) |
| `meta.csv` | the group of each cell (A: 30, B: 35, C: 25 cells) |
| `generate_reference.R` | computes the reference p-values with limma |
| `pvals_limma.csv` | the reference p-values (output of `generate_reference.R`) |

`generate_reference.R` tests every gene for each group vs the rest with `limma::rankSumTestWithCorrelation`. It
uses the two-sided p-value as in Seurat: `min(2 * min(lower, upper), 1)`.

## Regenerating the reference

Run from this directory:

    Rscript generate_reference.R

The current file was generated with R 4.5.1 and limma 3.64.3 (identical to Seurat 5.3.0
`FindMarkers(test.use = "wilcox_limma")`).

# group_metrics vs Seurat

Reference data for `tests/group_metrics_seurat_test.py`, which checks that `metrics.group_metrics` gives the same
results as Seurat.

> **Note:** `generate_reference.R` is not run by the test suite. The tests only read `seurat_markers.csv`, so R and
> Seurat are not needed to run them. The script documents where the reference values come from and is only run by
> hand, to regenerate `seurat_markers.csv` after changing the data or the settings.

| File | Content |
|---|---|
| `data.csv` | dummy log-normalized expression data, genes x cells (40 genes, 180 cells) |
| `meta.csv` | the group of each cell (A, B and C, 60 cells each) |
| `generate_reference.R` | computes the reference results with Seurat |
| `seurat_markers.csv` | the reference results (output of `generate_reference.R`) |

`generate_reference.R` runs Seurat's Wilcoxon test for three `min.pct` / `logfc.threshold` settings:

- `FindMarkers` for A vs B and C vs A (`group_oi` and `group_ref`)
- `FindMarkers` for each group vs A (`group_ref` only)
- `FindAllMarkers`, each group vs the rest (neither `group_oi` nor `group_ref`)

For each comparison, gene and group it stores the p-value, log fold change, `pct` and `pct_ref`.

## Regenerating the reference

Run from this directory:

    Rscript generate_reference.R

The current file was generated with R 4.5.1 and Seurat 5.3.0, without presto (Seurat then uses `wilcox.test`).

# Reference p-values for tests/wilcoxon_limma_test.py
# limma::rankSumTestWithCorrelation for every gene, each group vs the rest, two-sided as in Seurat's
# FindMarkers(test.use = "wilcox_limma"): min(2 * min(lower, upper), 1)
# usage (from this directory): Rscript generate_reference.R
# generated with R 4.5.1, limma 3.64.3 (identical to Seurat 5.3.0 FindMarkers(test.use = "wilcox_limma"))
suppressPackageStartupMessages(library(limma))
data <- as.matrix(read.csv("data.csv", row.names = 1, check.names = FALSE))  # genes x cells
meta <- read.csv("meta.csv", row.names = 1)
pvals <- do.call(rbind, lapply(sort(unique(meta$group)), function(g) {
  idx <- which(meta$group == g)
  data.frame(
    gene = rownames(data),
    group = g,
    pval = sapply(rownames(data), function(x) min(2 * min(rankSumTestWithCorrelation(index = idx, statistics = data[x, ])), 1))
  )
}))
write.csv(pvals, "pvals_limma.csv", row.names = FALSE)

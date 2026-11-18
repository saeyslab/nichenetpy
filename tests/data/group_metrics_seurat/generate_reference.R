# Reference results for tests/group_metrics_seurat_test.py
# Seurat's FindMarkers (two groups, and each group vs reference group A) and FindAllMarkers (one group vs the rest)
# with the Wilcoxon test, for several min.pct / logfc.threshold settings, without any other filter
# usage (from this directory): Rscript generate_reference.R
# generated with R 4.5.1, Seurat 5.3.0 (without presto, i.e. using wilcox.test)
suppressPackageStartupMessages(library(Seurat))
data <- as.matrix(read.csv("data.csv", row.names = 1, check.names = FALSE))  # genes x cells, log-normalized
meta <- read.csv("meta.csv", row.names = 1)
obj <- CreateSeuratObject(counts = round(expm1(data)), meta.data = meta, min.cells = 0, min.features = 0)
obj <- SetAssayData(obj, layer = "data", new.data = data)
Idents(obj) <- "group"
settings <- list(c(min_pct = 0.1, lfc = 0), c(min_pct = 0.3, lfc = 0), c(min_pct = 0.1, lfc = 0.5))
res <- list()
for (s in settings) {
  for (pair in list(c("A", "B"), c("C", "A"))) {
    m <- FindMarkers(obj, ident.1 = pair[1], ident.2 = pair[2], test.use = "wilcox", min.pct = s[["min_pct"]],
                     logfc.threshold = s[["lfc"]], min.cells.feature = 0, min.cells.group = 0, verbose = FALSE)
    res[[length(res) + 1]] <- data.frame(mode = paste0(pair[1], "_vs_", pair[2]), min_pct = s[["min_pct"]],
                                         lfc_thr = s[["lfc"]], gene = rownames(m), group = pair[1], pval = m$p_val,
                                         lfc = m$avg_log2FC, pct = m$pct.1, pct_ref = m$pct.2)
  }
  # group_ref without group_oi (no single Seurat function): each other group vs the reference group with FindMarkers
  for (g in setdiff(sort(unique(meta$group)), "A")) {
    m <- FindMarkers(obj, ident.1 = g, ident.2 = "A", test.use = "wilcox", min.pct = s[["min_pct"]],
                     logfc.threshold = s[["lfc"]], min.cells.feature = 0, min.cells.group = 0, verbose = FALSE)
    res[[length(res) + 1]] <- data.frame(mode = "vs_A", min_pct = s[["min_pct"]], lfc_thr = s[["lfc"]],
                                         gene = rownames(m), group = g, pval = m$p_val, lfc = m$avg_log2FC,
                                         pct = m$pct.1, pct_ref = m$pct.2)
  }
  m <- FindAllMarkers(obj, test.use = "wilcox", min.pct = s[["min_pct"]], logfc.threshold = s[["lfc"]], only.pos = FALSE,
                      return.thresh = 1, min.cells.feature = 0, min.cells.group = 0, verbose = FALSE)
  res[[length(res) + 1]] <- data.frame(mode = "one_vs_rest", min_pct = s[["min_pct"]], lfc_thr = s[["lfc"]],
                                       gene = m$gene, group = as.character(m$cluster), pval = m$p_val,
                                       lfc = m$avg_log2FC, pct = m$pct.1, pct_ref = m$pct.2)
}
write.csv(do.call(rbind, res), "seurat_markers.csv", row.names = FALSE)

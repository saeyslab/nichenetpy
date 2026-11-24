# Reference gene sets of interest of nichenetr's nichenet_seuratobj_aggregate for tests/get_geneset_oi_test.py:
# the receiver's condition of interest compared with a reference condition, for several conditions and settings.
# It uses the data of ../group_metrics_seurat (one receiver cell type, the groups A, B and C as conditions) and a small
# dummy model (the gene set doesn't depend on the model, except that it only keeps the genes of the model).
# Run from this directory: Rscript generate_reference.R
# It prints the reference as the Python dict in the test.
suppressPackageStartupMessages({
  library(nichenetr)
  library(Seurat)
  library(dplyr)
})
d <- file.path("..", "group_metrics_seurat")
data <- as.matrix(read.csv(file.path(d, "data.csv"), row.names = 1, check.names = FALSE))  # genes x cells
meta <- read.csv(file.path(d, "meta.csv"), row.names = 1)
meta$celltype <- "R"
obj <- CreateSeuratObject(counts = round(expm1(data)), meta.data = meta, min.cells = 0, min.features = 0)
# nichenetr needs a sparse data layer
obj <- SetAssayData(obj, layer = "data", new.data = as(data, "dgCMatrix"))
Idents(obj) <- "celltype"

# the dummy model, the same as in the test
genes <- rownames(data)
ligands <- paste0("gene", 30:34)
receptors <- c("gene8", "gene28", "gene38", "gene5", "gene20")
ligand_target_matrix <- outer(seq_along(genes) - 1, seq_along(ligands) - 1, function(i, j) ((i * 7 + j * 3) %% 11 + 1) / 12)
dimnames(ligand_target_matrix) <- list(genes, ligands)
lr_network <- tibble(from = ligands, to = receptors)
weighted_networks <- list(
  lr_sig = lr_network %>% mutate(weight = 1),
  gr = tibble(from = receptors, to = genes[1:5], weight = 1)
)

cat("NICHENETR_GENESET_OI = {\n")
for (pair in list(c("A", "B"), c("A", "C"), c("B", "C"), c("C", "A"))) {
  for (s in list(c(pct = 0.1, lfc = 0.25), c(pct = 0.3, lfc = 0.25), c(pct = 0.1, lfc = 0.5))) {
    out <- nichenet_seuratobj_aggregate(
      receiver = "R", seurat_obj = obj, condition_colname = "group", condition_oi = pair[1],
      condition_reference = pair[2], sender = "undefined", ligand_target_matrix = ligand_target_matrix,
      lr_network = lr_network, weighted_networks = weighted_networks, expression_pct = s[["pct"]],
      lfc_cutoff = s[["lfc"]], verbose = FALSE
    )
    genes_oi <- paste0('"', sort(out$geneset_oi), '"', collapse = ", ")
    cat(sprintf('    ("%s", "%s", %s, %s): {%s},\n', pair[1], pair[2], s[["pct"]], s[["lfc"]], genes_oi))
  }
}
cat("}\n")
cat(sprintf("# R %s, nichenetr %s, Seurat %s\n", getRversion(), packageVersion("nichenetr"), packageVersion("Seurat")))

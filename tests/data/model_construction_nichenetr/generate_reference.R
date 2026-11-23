# Reference values of nichenetr for the personalized PageRank (PPR) cases of
# tests/full_nichenet_test.py::test_model_construction: the first 10 rows of the ligand-target matrix.
# Run from this directory: Rscript generate_reference.R [folder with the network files]
# (default: ../tutorial_files/model_construction, where the test suite downloads them)
suppressMessages({
  library(nichenetr)
  library(dplyr)
  library(readr)
})
args <- commandArgs(trailingOnly = TRUE)
d <- if (length(args) > 0) args[1] else file.path("..", "tutorial_files", "model_construction")
gr <- read_csv(file.path(d, "gr_human.csv"), show_col_types = FALSE)
lr <- read_csv(file.path(d, "lr_network_human.csv"), show_col_types = FALSE)
sig <- read_csv(file.path(d, "lr_sig_human.csv"), show_col_types = FALSE)
annotation <- read_csv(file.path(d, "annotation_data_sources.csv"), show_col_types = FALSE)
optimized <- read_csv(file.path(d, "optimized_source_weights.csv"), show_col_types = FALSE)

# prints the first 10 rows of the ligand-target matrix as the Python list in the test
print_reference <- function(name, lr, sig, gr, source_weights, ligands) {
  wn <- construct_weighted_networks(lr_network = lr, sig_network = sig, gr_network = gr, source_weights_df = source_weights)
  wn <- apply_hub_corrections(weighted_networks = wn, lr_sig_hub = 0.115, gr_hub = 0.0803)
  m <- construct_ligand_target_matrix(
    weighted_networks = wn, lr_network = lr, ligands = ligands, ltf_cutoff = 0.926, algorithm = "PPR",
    damping_factor = 0.789, ligands_as_cols = TRUE
  )
  top <- as.matrix(m[1:10, , drop = FALSE])
  cat(sprintf("# %s targets x %s ligands (%s), first targets: %s\n", nrow(m), ncol(m),
              paste(colnames(m), collapse = ", "), paste(rownames(top), collapse = ", ")))
  cat(name, "= [\n")
  for (i in seq_len(nrow(top))) {
    values <- sprintf("%.10g", top[i, ])
    cat("    ", if (ncol(top) > 1) paste0("(", paste(values, collapse = ", "), ")") else values, ",\n", sep = "")
  }
  cat("]\n")
}

all_sources <- tibble(source = unique(c(lr$source, sig$source, gr$source)), weight = 1)
# 0: all data sources with weight 1, TNF and the combination TNF-IL6
print_reference("MC_top_10_ligand_target_matrix_PPR_0", lr, sig, gr, all_sources, list("TNF", c("TNF", "IL6")))
# 1: the optimized source weights (avg_weight), TNF
print_reference(
  "MC_top_10_ligand_target_matrix_PPR_1", lr, sig, gr,
  optimized %>% transmute(source, weight = avg_weight), list("TNF")
)
# 2: only the literature, comprehensive_db and ChIP data sources with weight 1, TNF
keep <- annotation %>% filter(type_db %in% c("literature", "comprehensive_db", "ChIP")) %>% pull(source)
print_reference(
  "MC_top_10_ligand_target_matrix_PPR_2", lr %>% filter(source %in% keep), sig %>% filter(source %in% keep),
  gr %>% filter(source %in% keep), all_sources %>% filter(source %in% keep), list("TNF")
)
cat(sprintf("# R %s, nichenetr %s, igraph %s\n", getRversion(), packageVersion("nichenetr"), packageVersion("igraph")))

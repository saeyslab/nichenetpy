# Reference values of nichenetr for test_steps, test_wrapper, test_steps_prioritization and
# test_target_prediction_evaluation_geneset in tests/full_nichenet_test.py: the steps of these tests (the notebooks
# steps, wrapper, steps_prioritization and target_prediction_evaluation_geneset) computed with nichenetr on the same
# data, after converting gene aliases to gene symbols as the notebooks do.
# Usage, from this directory: Rscript generate_reference.R <data directory> [convert]
#   <data directory> contains seuratObj.rds (Zenodo record 3531889) and the nichenetr v2 mouse model
#   ligand_target_matrix_nsga2r_final_mouse.rds, lr_network_mouse_21122021.rds and
#   weighted_networks_nsga2r_final_mouse.rds (Zenodo record 7074291)
#   convert: TRUE (default) converts gene aliases to symbols, FALSE keeps the original gene names
# It prints the reference values as the Python code in the test.
suppressPackageStartupMessages({
  library(nichenetr)
  library(Seurat)
  library(dplyr)
  library(tibble)
})
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
convert <- if (length(args) > 1) as.logical(args[2]) else TRUE

# ---- data and model ----
original <- UpdateSeuratObject(readRDS(file.path(data_dir, "seuratObj.rds")))
counts <- GetAssayData(original, assay = "RNA", layer = "counts")
data <- GetAssayData(original, assay = "RNA", layer = "data")
genes <- rownames(data)
if (convert) {
  # alias_to_symbol_seurat doesn't support Seurat v5 objects, so the expression matrices are renamed as it does (gene
  # names that would become the same symbol keep their original name) and a new Seurat object is created
  newnames <- convert_alias_to_symbols(genes, organism = "mouse", verbose = FALSE)
  doubles <- newnames %>% table() %>% .[. > 1] %>% names()
  genes_remove <- (names(newnames[newnames %in% doubles]) != (newnames[newnames %in% doubles])) %>%
    .[. == TRUE] %>% names()
  newnames[genes_remove] <- genes_remove
  genes <- unname(newnames)
}
rownames(counts) <- genes
rownames(data) <- genes
seurat_obj <- CreateSeuratObject(counts = counts, meta.data = original@meta.data, min.cells = 0, min.features = 0)
seurat_obj <- SetAssayData(seurat_obj, layer = "data", new.data = data)
Idents(seurat_obj) <- "celltype"

ligand_target_matrix <- readRDS(file.path(data_dir, "ligand_target_matrix_nsga2r_final_mouse.rds"))
lr_network <- readRDS(file.path(data_dir, "lr_network_mouse_21122021.rds")) %>% distinct(from, to)
weighted_networks <- readRDS(file.path(data_dir, "weighted_networks_nsga2r_final_mouse.rds"))

# ---- printing as Python ----
num <- function(x) if (is.logical(x)) as.character(as.integer(x)) else trimws(formatC(x, digits = 8, format = "g"))
py_tuples <- function(name, df) {
  cat(name, " = [\n", sep = "")
  for (i in seq_len(nrow(df))) {
    values <- vapply(df[i, ], function(v) if (is.character(v)) paste0('"', v, '"') else num(v), "")
    cat("    (", paste(values, collapse = ", "), "),\n", sep = "")
  }
  cat("]\n")
}
py_value <- function(name, x) cat(name, " = ", if (is.character(x)) paste0('"', x, '"') else num(x), "\n", sep = "")

receiver <- "CD8 T"
sender_celltypes <- c("CD4 T", "Treg", "Mono", "NK", "B", "DC")

# ---- test_steps / test_wrapper (notebooks steps and wrapper) ----
expressed_genes_receiver <- get_expressed_genes(receiver, seurat_obj, pct = 0.05)
expressed_receptors <- intersect(unique(lr_network$to), expressed_genes_receiver)
potential_ligands <- lr_network %>% filter(to %in% expressed_receptors) %>% pull(from) %>% unique()
expressed_genes_sender <- sender_celltypes %>% lapply(get_expressed_genes, seurat_obj, 0.05) %>% unlist() %>% unique()
potential_ligands_focused <- intersect(potential_ligands, expressed_genes_sender)
seurat_obj_receiver <- subset(seurat_obj, idents = receiver)
DE_table_receiver <- FindMarkers(
  object = seurat_obj_receiver, ident.1 = "LCMV", ident.2 = "SS", group.by = "aggregate", min.pct = 0.05
) %>% rownames_to_column("gene")
geneset_oi <- DE_table_receiver %>% filter(p_val_adj <= 0.05 & abs(avg_log2FC) >= 0.25) %>% pull(gene)
geneset_oi <- geneset_oi %>% .[. %in% rownames(ligand_target_matrix)]
background_expressed_genes <- expressed_genes_receiver %>% .[. %in% rownames(ligand_target_matrix)]

cat("# test_steps / test_wrapper\n")
py_value("n_expressed_genes_sender", length(expressed_genes_sender))
py_value("n_potential_ligands", length(potential_ligands))
py_value("n_potential_ligands_focused", length(potential_ligands_focused))
py_value("n_geneset", length(geneset_oi))
py_value("n_expressed_genes_receiver", length(expressed_genes_receiver))
py_value("n_background_expressed_genes", length(background_expressed_genes))

ligand_activities <- predict_ligand_activities(
  geneset = geneset_oi, background_expressed_genes = background_expressed_genes,
  ligand_target_matrix = ligand_target_matrix, potential_ligands = potential_ligands
)
links <- function(ligand_activities, suffix) {
  # the activities sorted (ties by ligand name, as the test), the ligand-target links of the top 30 ligands (n = 100)
  # and their ligand-receptor links
  ligand_activities <- ligand_activities %>% arrange(-aupr_corrected, test_ligand)
  py_value(paste0("n_ligand_activities", suffix), nrow(ligand_activities))
  py_tuples(paste0("BASIC_top_10_ligand_activities", suffix), ligand_activities[1:10, c("test_ligand", "aupr_corrected")])
  best_upstream_ligands <- ligand_activities$test_ligand[1:30]
  active_ligand_target_links <- best_upstream_ligands %>%
    lapply(get_weighted_ligand_target_links, geneset = geneset_oi, ligand_target_matrix = ligand_target_matrix, n = 100) %>%
    bind_rows()
  py_value(paste0("n_active_ligand_target_links", suffix), nrow(active_ligand_target_links))
  py_tuples(
    paste0("BASIC_top_10_active_ligand_target_links", suffix),
    active_ligand_target_links %>% arrange(desc(weight)) %>% head(10)
  )
  ligand_receptor_links <- get_weighted_ligand_receptor_links(
    best_upstream_ligands, expressed_receptors, lr_network, weighted_networks$lr_sig
  )
  py_value(paste0("n_ligand_receptor_links", suffix), nrow(ligand_receptor_links))
  py_tuples(
    paste0("BASIC_top_10_ligand_receptor_links", suffix),
    ligand_receptor_links %>% arrange(desc(weight), from, to) %>% head(10)
  )
  best_upstream_ligands
}
invisible(links(ligand_activities, ""))
best_upstream_ligands_focused <- links(ligand_activities %>% filter(test_ligand %in% potential_ligands_focused), "_focused")

# log fold changes of the top 30 focused ligands in the senders (LCMV vs SS), for the (ligand, sender) pairs of the test
lfcs <- sender_celltypes %>%
  lapply(get_lfc_celltype, seurat_obj, "aggregate", "LCMV", "SS", celltype_col = "celltype",
         features = best_upstream_ligands_focused, min.pct = 0, logfc.threshold = 0) %>%
  Reduce(function(x, y) full_join(x, y, by = "gene"), .)
pairs <- list(c("Il27", "CD4 T"), c("Ebi3", "Treg"), c("Tnf", "Mono"), c("H2-Eb1", "NK"), c("Ptprc", "B"), c("H2-M2", "DC"))
cat("# log fold changes: (ligand, sender, lfc)\n")
for (p in pairs) {
  value <- lfcs %>% filter(gene == p[1]) %>% pull(p[2])
  cat(sprintf('    ("%s", "%s", %s),\n', p[1], p[2], if (length(value) == 1) num(value) else "None"))
}

# ---- test_steps_prioritization (notebook steps_prioritization) ----
cat("# test_steps_prioritization\n")
ligand_activities_focused <- ligand_activities %>% filter(test_ligand %in% potential_ligands_focused) %>%
  arrange(-aupr_corrected) %>% mutate(rank = rank(desc(aupr_corrected)))
expressed_ligands <- intersect(unique(lr_network$from), expressed_genes_sender)
lr_network_filtered <- lr_network %>% filter(from %in% expressed_ligands & to %in% expressed_receptors)
info_tables <- generate_info_tables(
  seurat_obj, celltype_colname = "celltype", senders_oi = sender_celltypes, receivers_oi = receiver,
  lr_network_filtered = lr_network_filtered, condition_colname = "aggregate", condition_oi = "LCMV",
  condition_reference = "SS", scenario = "case_control"
)
print_row <- function(label, df) {
  cat("# ", label, "\n", sep = "")
  if (nrow(df) != 1) {
    cat("#   rows: ", nrow(df), "\n", sep = "")
    return(invisible())
  }
  for (col in colnames(df)) {
    v <- df[[col]]
    cat("#   ", col, ": ", if (is.character(v) || is.factor(v)) as.character(v) else num(v), "\n", sep = "")
  }
}
print_row("sender_receiver_de DC CD8 T H2-M2 Cd8a", info_tables$sender_receiver_de %>%
  filter(sender == "DC", receiver == "CD8 T", ligand == "H2-M2", receptor == "Cd8a"))
print_row("sender_receiver_info DC Mono B2m Tap1", info_tables$sender_receiver_info %>%
  filter(sender == "DC", receiver == "Mono", ligand == "B2m", receptor == "Tap1"))
print_row("lr_condition_de Cxcl11 Dpp4", info_tables$lr_condition_de %>% filter(ligand == "Cxcl11", receptor == "Dpp4"))
prior_table <- generate_prioritization_tables(
  info_tables$sender_receiver_info, info_tables$sender_receiver_de,
  ligand_activities_focused, info_tables$lr_condition_de, scenario = "case_control"
)
py_value("n_prior_table", nrow(prior_table))
print_row("prior_table NK CD8 T Ptprc Dpp4", prior_table %>%
  filter(sender == "NK", receiver == "CD8 T", ligand == "Ptprc", receptor == "Dpp4"))

# prioritization across multiple receivers
nichenet_output <- lapply(c("CD8 T", "CD4 T", "Treg"), function(receiver_ct) {
  output <- nichenet_seuratobj_aggregate(
    receiver = receiver_ct, seurat_obj = seurat_obj, condition_colname = "aggregate", condition_oi = "LCMV",
    condition_reference = "SS", sender = sender_celltypes, ligand_target_matrix = ligand_target_matrix,
    lr_network = lr_network, weighted_networks = weighted_networks, expression_pct = 0.05, verbose = FALSE
  )
  output$ligand_activities <- output$ligand_activities %>% mutate(receiver = receiver_ct)
  lr_network_receiver <- lr_network %>%
    filter(from %in% output$ligand_activities$test_ligand, to %in% output$background_expressed_genes)
  output$info_tables <- generate_info_tables(
    seurat_obj, celltype_colname = "celltype", senders_oi = sender_celltypes, receivers_oi = receiver_ct,
    lr_network_filtered = lr_network_receiver, condition_colname = "aggregate", condition_oi = "LCMV",
    condition_reference = "SS", scenario = "case_control"
  )
  output
})
info_tables_combined <- lapply(
  c(sender_receiver_de = "sender_receiver_de", sender_receiver_info = "sender_receiver_info",
    lr_condition_de = "lr_condition_de"),
  function(key) bind_rows(lapply(nichenet_output, function(o) o$info_tables[[key]])) %>% distinct()
)
ligand_activities_combined <- bind_rows(lapply(nichenet_output, function(o) o$ligand_activities))
prior_table_combined <- generate_prioritization_tables(
  info_tables_combined$sender_receiver_info, info_tables_combined$sender_receiver_de, ligand_activities_combined,
  info_tables_combined$lr_condition_de, scenario = "case_control"
)
py_tuples(
  "PRIOR_top_10_prior_table",
  prior_table_combined %>% filter(receiver == "CD8 T") %>% head(10) %>%
    select(sender, ligand, receptor, scaled_p_val_adapted_ligand, scaled_avg_exprs_ligand,
           scaled_p_val_adapted_receptor, scaled_avg_exprs_receptor, prioritization_score)
)

# ---- test_target_prediction_evaluation_geneset (random forests, n = 2 rounds of k = 3 folds) ----
cat("# test_target_prediction_evaluation_geneset\n")
set.seed(1)
output <- nichenet_output[[1]]
best_upstream_ligands <- output$ligand_activities %>% arrange(-aupr_corrected) %>% pull(test_ligand) %>% head(30)
gene_predictions_list <- seq(2) %>% lapply(
  assess_rf_class_probabilities, folds = 3, geneset = output$geneset_oi,
  background_expressed_genes = output$background_expressed_genes, ligands_oi = best_upstream_ligands,
  ligand_target_matrix = ligand_target_matrix
)
performances <- gene_predictions_list %>% lapply(classification_evaluation_continuous_pred_wrapper) %>% bind_rows()
py_value("tpp_mean_auroc", mean(performances$auroc))
py_value("tpp_mean_aupr", mean(performances$aupr))
py_value("tpp_mean_pearson", mean(performances$pearson))
discrete <- gene_predictions_list %>% lapply(calculate_fraction_top_predicted) %>% bind_rows() %>% ungroup()
py_tuples("TPEG_target_prediction_performances_discrete", as.data.frame(discrete))
py_value("fraction_true_target", mean(discrete$fraction_positive_predicted[discrete$true_target]))
py_value("fraction_no_target", mean(discrete$fraction_positive_predicted[!discrete$true_target]))
cat(sprintf(
  "# convert = %s; R %s, nichenetr %s, Seurat %s\n", convert, getRversion(), packageVersion("nichenetr"),
  packageVersion("Seurat")
))

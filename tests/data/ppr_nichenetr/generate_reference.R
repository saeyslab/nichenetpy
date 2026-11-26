# Reference ligand-tf matrices of nichenetr (personalized PageRank, igraph PRPACK) for tests/ppr_nichenetr_test.py.
# Run from this directory: Rscript generate_reference.R
suppressMessages({
  library(nichenetr)
  library(dplyr)
})
weighted_networks <- list(lr_sig = as_tibble(read.csv("lr_sig.csv")), gr = as_tibble(read.csv("gr.csv")))
# construct_ligand_tf_matrix only uses the ligand-receptor network to know the ligands
lr_network <- tibble(from = c("L1", "L2", "L3", "L4"), to = c("R1", "R2", "R3", "R1"))
ligands <- list("L1", "L2", "L3", "L4", c("L1", "L2"), c("L1", "L4"))
settings <- list(
  c(damping = 0, cutoff = 0), c(damping = 0.2, cutoff = 0), c(damping = 0.5, cutoff = 0),
  c(damping = 0.85, cutoff = 0), c(damping = 0.99, cutoff = 0), c(damping = 0.5, cutoff = 0.5)
)
for (s in settings) {
  m <- construct_ligand_tf_matrix(
    weighted_networks, lr_network, ligands,
    ltf_cutoff = s[["cutoff"]], algorithm = "PPR", damping_factor = s[["damping"]], ligands_as_cols = FALSE
  )
  write.csv(as.matrix(m), sprintf("ltf_damping%s_cutoff%s.csv", s[["damping"]], s[["cutoff"]]))
}

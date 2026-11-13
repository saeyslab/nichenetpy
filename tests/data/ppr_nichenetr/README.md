# Personalized PageRank vs nichenetr

Reference data for `tests/ppr_nichenetr_test.py`, which checks that `model_construction.construct_ligand_tf_matrix`
with `algorithm="PPR"` gives the same ligand-tf matrix as nichenetr's `construct_ligand_tf_matrix`, which computes
personalized PageRank with igraph (PRPACK).

> **Note:** `generate_reference.R` is not run by the test suite. The tests only read the `ltf_*.csv` files, so R and
> nichenetr are not needed to run them. The script documents where the reference values come from and is only run by
> hand, to regenerate the `ltf_*.csv` files after changing the data.

| File | Content |
|---|---|
| `lr_sig.csv` | dummy weighted ligand-signaling network (49 links) |
| `gr.csv` | dummy weighted gene regulatory network (56 links) |
| `generate_reference.R` | computes the reference ligand-tf matrices with nichenetr |
| `ltf_damping{d}_cutoff{c}.csv` | the reference ligand-tf matrix for `damping_factor` d and `ltf_cutoff` c (output of `generate_reference.R`), ligands x genes |

The ligand-signaling network covers the cases where PageRank implementations differ:

- genes without outgoing links (G9, G14 and the ligand L4, which only has an incoming link): in igraph their score
  goes back to the seed ligand;
- self-loops (L1, R2, G4, G13), which igraph follows like any other link;
- ligand combinations (L1-L2, L1-L4), the mean of the members' scores.

`generate_reference.R` computes the matrix for the ligands L1, L2, L3, L4, L1-L2 and L1-L4 with `damping_factor`
0, 0.2, 0.5, 0.85 and 0.99 and `ltf_cutoff` 0 (no clipping, the raw PageRank scores), and once with
`damping_factor` 0.5 and `ltf_cutoff` 0.5.

## Regenerating the reference

Run from this directory:

    Rscript generate_reference.R

The current files were generated with R 4.5.1, nichenetr 2.2.1.1 and igraph 2.1.4.

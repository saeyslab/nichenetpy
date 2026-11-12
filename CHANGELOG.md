# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [1.0.0] - 2026-05-29

## [1.0.1] - 2026-06-29

### Changed

- Changed some code in `ann_utils.subset_ann` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `extraction._get_expressed_features` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `model_construction.construct_ligand_tf_matrix` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `model_construction.construct_tf_target_matrix` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `network.Network.__init__` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `prioritization.process_table_to_ic` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `utils.decomplexify` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `visualization.visualize_ligand_signaling_graph` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `visualization.create_mushroom_plot` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `wilcoxon._rank_cells` so that it uses pandas built-in functions. By [@vmedaert]
- Changed some code in `wrappers.create_ligand_links_circos_plot` so that it uses pandas built-in functions. By [@vmedaert]
- Updated docstring in `model_construction.construct_ligand_tf_matrix`. By [@vmedaert]

## [1.0.2] - 2026-08-11

### Added

- New version of the optimization procedure. By [@vmedaert]
- Added extra comments in `evaluation.get_single_ligand_importances` and `evaluation.evaluate_single_importances_ligand_prediction`. By [@vmedaert]

### Fixed

- Fixed bug where `celltype` was hardcoded in `prioritization.get_avg_exp`. By [@vmedaert]
- Fixed bug where `celltype` was hardcoded in `visualization.assign_ligands_to_celltype`. By [@vmedaert]

## [1.0.3] - 2026-10-05

### Added

- Support for latest python versions.

### Changed

- `io_test.test_io` now uses randomly generated test objects in stead of objects saved in a file. By [@vmedaert]

## [1.0.4] - 2026-10-15

### Fixed

- Fixed bug when reading GeneAliasInfo or GeneInfo from csv files where commas inside gene symbols where used for splitting.

## [2.0.0] - TODO

### Added

- Support for scipy sparse arrays.

- Added function `metrics.precision_at_k`

- Added function `metrics.average_precision_at_k`

- Added function `metrics.dcg_at_k`

- Added function `metrics.ndcg_at_k`

- Added `ligand_evaluation_metrics` and/or `target_evaluation_metrics` arguments to choose which evaluation metrics are computed, in `parameter_optimization.evaluate_model` and `parameter_optimization.construct_and_evaluate` (both), `evaluation.evaluate_single_importances_ligand_prediction` (both), `evaluation.get_single_ligand_importances` (target) and `prediction.LigandActivityPredictor.evaluate_target_prediction` and `prediction.LigandActivityPredictor.predict_ligand_activities` (target).

- Added a `target_evaluation_metric` argument to `wrappers.run_nichenet`, `wrappers.run_nichenet_cluster_de`, `prioritization.generate_prioritization_table`, `visualization.create_barcode_plot` and the activity plotting functions in `wrappers` (`create_ligand_activity_hist`, `create_ligand_activity_heatmap`, `create_lfc_heatmap`), which previously always used `aupr_corrected`.

- Added `split_direct` and `direct_coef` arguments to `model_construction.construct_ligand_target_matrix`, `model_construction.construct_model_from_source_weights` and `parameter_optimization.construct_and_evaluate`. They control how much weight direct ligand-target links get relative to indirect ones. `split_direct` chooses which matrix is split into a direct and an indirect part: `"ltf"` (ligand-tf), `"tft"` (tf-target), `"ltf-tft"` (both) or `"no"` (default, unchanged behaviour). The ligand-target matrix is then `direct_coef * direct + (1 - direct_coef) * indirect`. `"ltf"` and `"ltf-tft"` require `algorithm="PPR"`.

- Added `split_ppr` argument to `model_construction.construct_ligand_tf_matrix`, which separates the PPR scores into two matrices.

- `visualization.heatmap_1d` and `visualization.heatmap_2d` now also accept a `matplotlib.colors.Colormap` for `cmap`, in addition to a colormap name.

### Changes

- Type signatures have been altered and static type checking has been enabled for the package. This is not considered backwards compatible!

- `parameter_optimization.construct_and_evaluate` now returns the empty solution (all scores 0) when all ligand-receptor and signaling sources have weight 0, instead of constructing and evaluating a model without a ligand-signaling network.

- Faster quantile clipping and conversion of shortest path lengths to scores (SPL algorithm) during ligand-tf matrix construction; shortest paths are now computed with `scipy.sparse.csgraph.dijkstra` (~20x faster SPL ligand-tf matrices).

- `model_construction.construct_ligand_tf_matrix` (and therefore `construct_ligand_target_matrix` and `construct_model_from_source_weights`) now raises a `ValueError` when none of the requested ligands are present in the weighted networks, instead of failing later during matrix multiplication.

- Removed the `graph` module (`graph.dijkstra_spl` and `graph.get_reachable_nodes`), use `scipy.sparse.csgraph` instead. As a result, `visualization.visualize_ligand_signaling_graph` may show a different path when several shortest paths have the same length.

### Fixed

- Corrected docstrings.

- Fixed bug in `prediction.LigandActivityPredictor.replace_zero_col_by_noisy_scores` where sparse matrices were not supported (all-zero columns of a sparse matrix were never filled). The method is also faster (~15x for a dense and ~3x for a sparse 25,000 x 1,300 matrix); the noise is now drawn with `numpy.random`.

- Fixed bug in `io.write_ligand_target_matrix` where sparse matrices were not supported.

- Fixed bug in `visualization.create_barcode_plot` where the last ranked element was omitted.

- Fixed bugs in type checks at runtime.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where the down cutoff was positive.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where the geneset argument was passed to `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` instead of `geneset_oi`.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where `geneset="up"` and `geneset="down"` both returned up- and downregulated genes. Fold changes are now computed as `receiver_affected` versus `receiver_reference`.

- Fixed bug in `model_construction.construct_model_from_source_weights` where tft and ltf matrices from `model_construction.construct_ligand_target_matrix` were swapped.

- Fixed bug in `model_construction.construct_ligand_target_matrix` where remove_direct_links argument did nothing.

- Fixed `model_construction.construct_ligand_target_matrix` with `secondary_targets=True`, which previously always raised a `ValueError`. The secondary-target scores are now computed as in NicheNetR: the quantile cutoff (`ltf_cutoff`) is applied only to the matrix used to infer secondary targets, and the primary ligand-target scores are left unclipped.

- Fixed `KeyError` in `wrappers.run_nichenet` with `get_prioritization_table=True` and `case_control=False`.

- Fixed bug in `wrappers.combine_weighted_ligand_target_links` that occurs when a ligand has no targets.

- Fixed bug in `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` that occurs when the ligand-target matrix is sparse.

- Added shape to `lr_sig_mat` construction in `visualization.get_ligand_signaling_path` which fixes an edge case bug.

- Fixed swapped edge colors in `visualization.visualize_ligand_signaling_graph` (the legend now matches the edges); the default values of `sig_color` and `gr_color` were swapped accordingly.

- Fixed bug in `ann_utils.subset_ann` where subsetting without specifying `layers` failed with newer anndata versions (≥ 0.12), which expose `X` as the layer `None`. This also fixes `metrics.group_metrics` when both `group_oi` and `group_ref` are given, and thereby `wrappers.run_nichenet` with `get_prioritization_table=True` and `case_control=True`.

- Fixed support for dense (`numpy.ndarray`) layers: `metrics.group_metrics` crashed and `extraction.get_lfc_celltype` returned a single value instead of a list of log fold changes. `utils.subset_matrix` is now faster for sparse inputs, and `metrics.gene_expression_pct` no longer modifies its input matrix or counts explicitly stored zeros as expressed.

- Fixed bug in `prioritization.get_avg_exp` where passing `features` subsetted the genes before normalization, so library sizes were computed from the selected genes only (inflating the average expression and changing its relative values across cell types). Results now match Seurat's `AverageExpression`. Also fixed a `KeyError` when `ann.var_names` is not named "gene", and `features` now accepts any iterable (including tuples) or a single gene name.

- Fixed bugs in `visualization.create_radar_plot`: omitting `max_val` raised an error and `matplotlib.colors.Colormap` objects were rejected for `cmap` despite its type hint. Also corrected the type of `data` to a sequence of sequences of floats or a 2-D `numpy.ndarray` (one row per plotted series).

- Fixed bug in `model_construction.construct_model_from_source_weights` where an empty ligand-signaling network (all ligand-receptor and signaling sources weighted 0) produced a transposed gene regulatory matrix with all genes as columns instead of a targets × ligands matrix. Such models now go through the regular construction path, where each ligand only reaches itself.

- Fixed ligand prediction evaluation for evaluation datasets with a ligand combination: `utils.is_ligand_active` never marked any ligand as active for a combination (it now marks each member, as in NicheNetR, and a combined "A-B" ligand), so these datasets were always dropped from the ligand prediction evaluation and from the corresponding optimization objectives. `evaluation.evaluate_single_importances_ligand_prediction` now assigns the full combination as true ligand to every output row, and no longer fails when the model contains only one ligand.

- Fixed bug in `evaluation.get_single_ligand_importances` and `evaluation.evaluate_single_importances_ligand_prediction` where the metric columns were mislabeled when the evaluation metrics were not passed in alphabetical order (e.g. `("auroc", "aupr_corrected")` swapped the two columns).

- Fixed bug in `parameter_optimization` (per-ligand averaging of the evaluation results) where single-ligand settings were matched by substring, so the scores of a ligand such as "TNF" also included settings of "TNFSF10" or "IL2" included "IL21", distorting the optimization objectives.

- Fixed bug in `parameter_optimization.compute_evaluation_scores` where a ValueError was raised when ligand prediction could not be evaluated on any data set, instead of falling back to scoring target prediction only (with ligand prediction objectives set to 0).

- Fixed bug in `parameter_optimization.choose_pareto_optimal_solution_normalized` where an objective without variance (the same value for every solution) led to a division by zero, so the first solution was always chosen. Such objectives are now not taken into account.

- Fixed bug in `network.Network.subset_sep` where calling it on a `network.WeightedNetwork` without `to` raised a `ValueError`.

- Fixed `network.Network.__contains__` (`(from, to) in network`): it raised a `KeyError` for an unknown "from" gene and never matched links of a `network.WeightedNetwork` without the exact weight. Membership is now based on (from, to) only and uses a set of (from, to) pairs that is built on the first membership check (much faster for repeated checks) and reset by `network.Network._build_index` when the links change.

- Fixed `metrics.group_metrics` with `wilcoxon_limma=True`: it failed when `features` was given or the layer was a sparse array, and it ignored `min_pct` and `min_abs_lfc`. It now tests the same genes as the default Wilcoxon test; p-values are verified against limma's `rankSumTestWithCorrelation` (as used by Seurat's `FindMarkers(test.use = "wilcox_limma")`).

- Fixed the `min_pct` / `min_abs_lfc` prefilter of `metrics.group_metrics` (and therefore `prioritization.calculate_de`, `wrappers.get_geneset_oi`, `wrappers.run_nichenet` and `wrappers.run_nichenet_cluster_de`): it only used the fraction of expressing cells in the group of interest, so genes lost in that group (e.g. strongly downregulated genes for `geneset="down"`) were never tested, and in one-vs-rest comparisons a gene passing the filter for one group was also reported for all other groups. The filter is now applied per comparison as in Seurat's `FindMarkers` (expressed in at least `min_pct` of the cells of either group and an absolute log fold change of at least `min_abs_lfc`); results are verified against Seurat. The output has a new column `pct_ref` (fraction of expressing cells in the reference cells).

- Fixed `metrics.group_metrics` and `extraction.get_lfc_celltype` when `features` is a generator (the gene names came back empty).

- Fixed `io.write_sparse_matrix` and `io.write_ligand_target_matrix` for matrices with a float type other than float64 (e.g. float32): the readers assume float64, so reading such a file raised a `ValueError` or returned wrong values. The data is now converted to float64 when it is written.

- Fixed `metrics.group_metrics` with `group_ref` but without `group_oi`: the log fold changes compared each group with `group_ref`, but the p-values compared it with all other cells, and `group_ref` itself was reported (compared with itself). P-values are now computed against `group_ref` and the reference group is no longer in the output. Also fixed an `IndexError` when no gene passes the `min_pct` / `min_abs_lfc` filters (an empty table is returned), and `group_oi` equal to `group_ref` now raises a `ValueError`.

- Fixed `metrics.calculate_evaluation_metrics` changing the global warning filters: after each call all warnings were reset to the "default" action (overriding filters set by the user, e.g. warnings as errors), and when an error was raised all warnings stayed silenced. Warnings are now only silenced during the call.

- Fixed `io.read_csc_matrix` and `io.read_csr_matrix` returning a matrix that was too small when its last rows (csc) or last columns (csr) contained only zeros; the shape is now taken from the stored row and column names. Fixed `io.write_sparse_matrix` for matrices with int64 indices (used by scipy for large matrices and after some operations), which could not be read back; the indices are now written as int32. `io.write_sparse_matrix` now raises a `ValueError` if the number of names doesn't match the shape of the matrix or if the matrix is too large for the file format.

- Fixed `io.write_network` and `io.write_weighted_network`, which failed for networks with fewer than 3 genes and used one byte per bit for the gene ids (e.g. 15 instead of 2 bytes for 25,000 genes). Files are now 2-3 times smaller and remain readable by older versions; older files can still be read.



[1.0.4]: https://github.com/saeyslab/nichenetpy/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/saeyslab/nichenetpy/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/saeyslab/nichenetpy/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/saeyslab/nichenetpy/compare/v1.0.0...v1.0.1
[@vmedaert]: https://github.com/vmedaert

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

## [2.0.0] - 2026-11-25

### Added

- Support for scipy sparse arrays.

- Combinations of int gene ids are supported: a combination is named by the tuple of its ids (`typing.GeneCombination`), a combination of gene symbols keeps its name (e.g. "TNF-LTB"). The ligand names of a `prediction.LigandActivityPredictor` can therefore be genes or tuples of genes. A combination can also be passed to the predictor as a tuple of genes, e.g. `("TNF", "LTB")` for "TNF-LTB".

- Added `exception.NotInModelError`, raised by `prediction.LigandActivityPredictor` when a gene or ligand isn't in the ligand-target matrix. It's a `KeyError` and, as before, a `ValueError`.

- Added `metrics.calculate_evaluation_metrics`, which computes a chosen subset of the metrics `aupr`, `aupr_corrected`, `auroc`, `pearson`, `map` and `ndcg` (`map` and `ndcg` only for ligand prediction).

- Added `exception.NoTrueSamplesError` (a `ValueError`), raised by `metrics.calculate_evaluation_metrics` and `metrics.calculate_prediction_evaluation_metrics` when there are no true samples (was a plain `ValueError`).

- Added function `metrics.precision_at_k`

- Added function `metrics.average_precision_at_k`

- Added function `metrics.dcg_at_k`

- Added function `metrics.ndcg_at_k`

- Added `ligand_evaluation_metrics` and/or `target_evaluation_metrics` arguments to choose which evaluation metrics are computed, in `parameter_optimization.evaluate_model` and `parameter_optimization.construct_and_evaluate` (both), `evaluation.evaluate_single_importances_ligand_prediction` (both), `evaluation.get_single_ligand_importances` (target) and `prediction.LigandActivityPredictor.evaluate_target_prediction` and `prediction.LigandActivityPredictor.predict_ligand_activities` (target). They accept any Iterable of metric names (a set is used in sorted order); a single str raises a `TypeError`.

- Added `metric_score_f` and `objective_fs` arguments to `parameter_optimization.compute_evaluation_scores` and `parameter_optimization.construct_and_evaluate`, to choose the metric score that selects the best ligand ranking metric and the optimization objectives. Presets are available in `parameter_optimization.metric_score_f_dct` (`"NNv2"`, `"all_geometric_mean"`) and `parameter_optimization.objective_fs_dct` (`"NNv2"`, `"map&ndcg"`); the default (`"NNv2"`) computes the same scores as before.

- Added a `target_evaluation_metric` argument to `wrappers.run_nichenet`, `wrappers.run_nichenet_cluster_de`, `prioritization.generate_prioritization_table`, `visualization.create_barcode_plot` and the activity plotting functions in `wrappers` (`create_ligand_activity_hist`, `create_ligand_activity_heatmap`, `create_lfc_heatmap`), which previously always used `aupr_corrected`.

- Added `split_direct` and `direct_coef` arguments to `model_construction.construct_ligand_target_matrix`, `model_construction.construct_model_from_source_weights` and `parameter_optimization.construct_and_evaluate`. They control how much weight direct ligand-target links get relative to indirect ones. `split_direct` chooses which matrix is split into a direct and an indirect part: `"ltf"` (ligand-tf), `"tft"` (tf-target), `"ltf-tft"` (both) or `"no"` (default, unchanged behaviour). The ligand-target matrix is then `direct_coef * direct + (1 - direct_coef) * indirect`. `"ltf"` and `"ltf-tft"` require `algorithm="PPR"`. With `"tft"` and `"ltf-tft"`, the direct links are the regulatory links that start at any ligand of the ligand-receptor network (including the members of ligand combinations), so a ligand's row doesn't depend on which other ligands are requested (e.g. the evaluation ligands during parameter optimization vs all ligands in the final model).

- Added `split_ppr` argument to `model_construction.construct_ligand_tf_matrix`, which separates the PPR scores into two matrices.

- `visualization.heatmap_1d`, `visualization.heatmap_2d` and `visualization.create_radar_plot` now also accept a `matplotlib.colors.Colormap` for `cmap`, in addition to a colormap name.

- `prediction.LigandActivityPredictor` accepts the row and column names as a `pandas.Index` (e.g. `ann.var_names`) or a `numpy.ndarray`, in addition to a list or tuple.

- Added a `random_state` argument (int, numpy.random.Generator or None) to `prediction.LigandActivityPredictor.replace_zero_col_by_noisy_scores` and `parameter_optimization.construct_and_evaluate`, to make the noise that replaces all-zero columns of the ligand-target matrix (and therefore the evaluation scores) reproducible. The default (None) uses numpy's global random state (nichenetpy 1.x used Python's `random` module).

- `utils.rank_genes_groups_to_dataframe` has a new column `pct_ref`: the fraction of expressing cells in the reference cells (the other cells, or the reference group), as the `pct_ref` column of `metrics.group_metrics`.

### Changed

- The `inplace` argument of pandas methods (deprecated in pandas 3.1) is no longer used.

- Type signatures have been altered and static type checking has been enabled for the package.

- `parameter_optimization.compute_evaluation_scores` returns a nested dict (`"target_prediction"` / `"ligand_prediction"` → metric → score) instead of a tuple of 4 floats, and `parameter_optimization.construct_and_evaluate` returns `(model, scores)` instead of `(model, target_auroc, target_aupr, ligand_auroc, ligand_aupr)`.

- `prediction.LigandActivityPredictor` stores a copy of the row and column names, so changing the names that were passed no longer makes the predictor inconsistent. `row_names` and `col_names` are read-only properties that return a tuple, also available as `gene_names` and `ligand_names` (predictors pickled with older versions are converted when they're loaded).

- `parameter_optimization.evaluate_model` and `construct_and_evaluate` no longer depend on the iteration order of sets: the rows of `performances_ligand_prediction` follow the order of the evaluation data, and the ligands are sorted, so results (including the noise of `replace_zero_col_by_noisy_scores` with a fixed `random_state`) are identical across Python processes.

- `parameter_optimization.construct_and_evaluate` now returns the empty solution (all scores 0, for the metrics of `objective_fs`) when all ligand-receptor and signaling sources have weight 0, instead of constructing and evaluating a model without a ligand-signaling network.

- Faster quantile clipping and conversion of shortest path lengths to scores (SPL algorithm) during ligand-tf matrix construction; shortest paths are now computed with `scipy.sparse.csgraph.dijkstra` (~20x faster SPL ligand-tf matrices).

- `model_construction.construct_ligand_tf_matrix` (and therefore `construct_ligand_target_matrix` and `construct_model_from_source_weights`) now raises a `ValueError` when none of the requested ligands are present in the weighted networks, instead of failing later during matrix multiplication.

- Removed the `graph` module (`graph.dijkstra_spl` and `graph.get_reachable_nodes`), use `scipy.sparse.csgraph` instead. As a result, `visualization.get_ligand_signaling_path` (and so the graph drawn by `visualization.visualize_ligand_signaling_graph`) may pick a different path when several shortest paths have the same length.

- Sparse matrix files written by `io.write_sparse_matrix` start with a header that stores whether the matrix is CSC or CSR, and `io.read_csc_matrix` / `read_csr_matrix` raise a `ValueError` for a file in the other format (a square matrix read with the wrong reader was silently transposed). Files without a header (written by nichenetpy 1.x) can still be read, but nichenetpy 1.x can't read the new files.

- `io.read_csc_matrix` and `io.read_csr_matrix` now return a `scipy.sparse.csc_array` / `csr_array` by default, as scipy expects to deprecate the sparse matrix interface. Use `sparse_array=False` to get a `csc_matrix` / `csr_matrix` as before.

- `io.write_ligand_target_matrix` and `io.write_sparse_matrix` are 2-2.5 times faster and no longer use twice the size of the matrix in extra memory while writing; apart from the header of sparse matrix files (see above), the files are unchanged.

- The matrix writers in `io` accept any ordered sequence of names (e.g. a tuple, numpy array or `pandas.Index`) instead of only a list, and `io.write_network` and `io.write_weighted_network` accept any iterable of links (e.g. a generator).

- `network.Network` and `network.WeightedNetwork` store a sorted copy of the mapping list, so the list no longer needs to be sorted by "from" and later changes to it don't affect the network.

- Faster `extraction.average_expression` (and therefore `prioritization.get_avg_exp`), e.g. ~25-60 times for 20,000 cells, and faster ligand activity prediction: `prediction.LigandActivityPredictor.predict_ligand_activities` ~2 times, `predict_single_cell_ligand_activities` ~2.5 times, and `get_col` on sparse matrices ~9 times. The results are unchanged.

- `ltf_cutoff` must be smaller than 1 in `model_construction.construct_ligand_tf_matrix`, `construct_ligand_target_matrix` and `construct_model_from_source_weights` (1 set every ligand-tf score to 0); NicheNetR allows 1.

- Personalized PageRank (`algorithm="PPR"`, the default) in `model_construction.construct_ligand_tf_matrix` (and therefore `construct_ligand_target_matrix`, `construct_model_from_source_weights` and the parameter optimization) now uses the model of NicheNetR (igraph's PRPACK solver) instead of scikit-network: the score that reaches a gene without outgoing links in the ligand-signaling network returns to the ligand, and self-loops are followed like other links. It's computed by power iteration, which stops for each ligand when its scores change by less than 1e-10 (PRPACK uses Gauss-Seidel); a ligand's scores don't depend on the other ligands. The scores match NicheNetR up to about 1e-9, so all PPR ligand-tf and ligand-target matrices change compared to before, more so for a higher `damping_factor` (up to 0.017 at 0.5 and 0.33 at 0.99 on a small test network). Each ligand is computed once, also when it is part of several combinations. `damping_factor` must now be smaller than 1 for every algorithm (1 was refused by scikit-network for PPR, and accepted for SPL and direct, which don't use it). The previous implementation is still available with the new argument `ppr_implementation="scikit-network"`; scikit-network is no longer a required dependency, install it with the optional extra `pip install nichenetpy[sknetwork]`.

- `normalization.scaling_zscore` (and therefore `activity_zscore` in `prioritization.generate_prioritization_table`) now uses the sample standard deviation (ddof=1), as NicheNetR's `scaling_zscore` (R's `sd`). The z-scores are smaller by a factor sqrt((n-1)/n) for n values; the order of the prioritization table and the prioritization scores don't change.

- `prioritization.generate_prioritization_table` now uses the sign of the log fold change as NicheNetR (`numpy.sign`: 0 for an lfc of 0 and NaN for a NaN lfc, both were treated as positive) to compute `pval_adapted_*`; this only matters for DE tables with an lfc of exactly 0 and a p-value below 1, or a NaN lfc.

- `wrappers.get_geneset_oi` has a new required argument `condition_ref`, right after `condition_oi` (as in `wrappers.run_nichenet`): the condition of interest is compared with this reference condition, as NicheNetR does, or with all other conditions (the previous behaviour) if it's None. Calls that pass `layer` or a later argument by position need to be adapted. It raises a `ValueError` when `condition_oi` or `condition_ref` isn't a condition of the receiver cells or both are the same.

- The scanpy path of `prioritization.calculate_de` (`use_scanpy=True`) applies `features`, `min_pct`, `min_abs_lfc` and `pval_thresh` as the default path (`metrics.group_metrics`) does; they were ignored. scanpy still tests all genes (so the adjusted p-values are computed over all genes, as with `group_metrics`), and only the rows of the genes in `features`, expressed in at least `min_pct` of the cells of the cell type or of the other cells, with an absolute log fold change (scanpy's) of at least `min_abs_lfc` and a p-value smaller than `pval_thresh` are kept. With the default values, only the rows with a p-value of 1 are left out, as in the default path. A gene of `features` that isn't in the AnnData object raises a `KeyError`.

- `wrappers.get_geneset_oi` with `use_scanpy=True` keeps a gene if it's expressed in at least `min_pct` of the cells of the condition of interest or of the reference cells, as the default path and Seurat's `min.pct`; it only used the condition of interest.

- Arguments that are an Iterable of str (gene symbols, metric names, cell types, labels) raise a `TypeError` for a single str, which was split into its characters (or, for `prioritization.get_avg_exp`, used as one gene): `ann_utils.subset_ann` (`genes`, `layers`), `metrics.group_metrics`, `prioritization.calculate_de` and `prioritization.get_avg_exp` (`features`), `extraction.average_expression` (`keys`), `extraction.get_weighted_ligand_receptor_links`, `metrics.calculate_evaluation_metrics` (`metrics`), `model_construction.construct_ligand_tf_matrix` and `construct_ligand_target_matrix` (`ligands`), `parameter_optimization.compute_evaluation_scores` (`ligands`) and `construct_and_evaluate` (the metrics), `evaluation.get_single_ligand_importances` and `evaluation.evaluate_single_importances_ligand_prediction`, the `prediction.LigandActivityPredictor` methods `predict_ligand_activities`, `get_weighted_ligand_target_links` and `evaluate_target_prediction`, `prediction.assess_rf_class_probabilities` (`ligands_oi`), `gene_symbol.GeneAliasInfo.alias_to_symbol`, `gene_symbol.GeneInfo.convert_human_to_mouse_symbols` and `convert_mouse_to_human_symbols` (which now check their argument when they're called), `visualization.create_radar_plot` (`value_labels`) and `wrappers.create_ligand_receptor_links_prioritization_circos_plot`. Arguments that also accept a single str say so in their annotation (e.g. `sender_celltypes` of `wrappers.run_nichenet`).

- Iterable arguments are only copied when that's needed: an argument that's used once isn't copied, and an argument that's used more than once is only stored in a list when it can be used only once (e.g. a generator).

### Fixed

- Corrected docstrings.

- Corrected the documentation of `sender_celltypes` in `wrappers.run_nichenet_cluster_de`: None doesn't filter ligands on sender expression (NicheNetR's sender = "undefined"); for NicheNetR's default sender = "all", pass all cell types.

- `visualization.reorder_labels` and `wilcoxon.wilcoxon_rank_sum_test` (`genes`) accept a numpy array and a `pandas.Index` (e.g. `ann.var_names`), and so does `utils.subset_matrix` (`rows`, `cols`), which rejected a numpy array for `cols` unless `rows` was one too. `reorder_labels` raises a `ValueError` when the number of labels doesn't match the shape of the matrix (too many labels were silently cut off) and always returns lists.

- The writers in `io` (`write_ligand_target_matrix`, `write_sparse_matrix`, `write_network`, `write_weighted_network`) raise a `TypeError` with a clear message when a name isn't a gene symbol (e.g. an int gene id).

- Fixed the validation of ligands in `evaluation.EvaluationData`. Created from a dict or `pandas.DataFrame` (or with `data[key] = ...`), it rejected int gene ids, numpy integers and combinations given as a tuple, and added the key to the caller's dict; `add` rejected numpy integers and tuples. Both accepted invalid members of a combination (e.g. booleans). All paths now check and convert a ligand the same way, and an invalid ligand raises a `TypeError` (was a `ValueError`).

- `wrappers.create_ligand_activity_heatmap` raises a clear `ValueError` when there are no ligands, as `create_ligand_activity_hist` does.

- Fixed `ann_utils.subset_ann`, `extraction.get_expressed_genes`, `extraction.get_lfc_celltype` and `metrics.group_metrics` selecting the wrong cells when a cell name occurs more than once (e.g. after concatenating datasets); cells are selected by position.

- NaN is no longer accepted for `damping_factor` (`model_construction.construct_ligand_tf_matrix` / `construct_ligand_target_matrix`), `hub` (`model_construction.apply_hub_correction`) and `pct` (`extraction.get_expressed_genes`); it gave NaN scores or no expressed genes.

- `parameter_optimization.evaluate_model` only leaves out (with a warning) the settings whose ligand prediction can't be evaluated because there is no active ligand; every other error was also turned into a warning, which hid bugs.

- Fixed bug in `prediction.LigandActivityPredictor.replace_zero_col_by_noisy_scores` where a sparse ligand-target matrix raised a `ValueError`. The method is also faster (~15x for a dense and ~3x for a sparse 25,000 x 1,300 matrix); the noise is now drawn with `numpy.random`.

- Fixed `io.write_ligand_target_matrix` failing for a predictor with a sparse ligand-target matrix; a sparse `mat` is accepted too.

- Fixed bugs in `visualization.create_barcode_plot`: the last ranked element was omitted, the barcode dropped its last segment and took the color of its first segment from the alphabetically first ligand instead of the top-ranked one.

- Fixed bugs in type checks at runtime.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where the down cutoff was positive.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where the geneset argument was passed to `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` instead of `geneset_oi`.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where `geneset="up"` and `geneset="down"` both returned up- and downregulated genes. Fold changes are now computed as `receiver_affected` versus `receiver_reference`.

- Fixed bug in `model_construction.construct_model_from_source_weights` where tft and ltf matrices from `model_construction.construct_ligand_target_matrix` were swapped.

- Fixed bug in `model_construction.construct_ligand_target_matrix` where remove_direct_links argument did nothing.

- Fixed `model_construction.construct_ligand_target_matrix` with `secondary_targets=True`, which previously always raised a `ValueError`. The secondary-target scores are now computed as in NicheNetR: the quantile cutoff (`ltf_cutoff`) is applied only to the matrix used to infer secondary targets, and the primary ligand-target scores are left unclipped.

- Fixed `KeyError` in `wrappers.run_nichenet` with `get_prioritization_table=True` and `case_control=False`.

- Fixed `wrappers.run_nichenet` returning a `ValueError` object instead of raising it (after running the whole analysis) when `get_prioritization_table=True` and `sender_celltypes` is None; it's now raised before the analysis starts.

- Fixed bug in `wrappers.combine_weighted_ligand_target_links` that occurs when a ligand has no targets.

- Fixed `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` returning sparse matrix rows instead of scores as the weights when the ligand-target matrix is sparse.

- Fixed an `IndexError` in `visualization.get_ligand_signaling_path` when a gene sorts after every gene with outgoing ligand-signaling links.

- Fixed swapped edge and node colors in `visualization.visualize_ligand_signaling_graph` (the legend now matches the edges; as documented and as in NicheNetR, ligand nodes use `sig_color` and target nodes `gr_color`); the default values of `sig_color` and `gr_color` were swapped accordingly, so with the default colors the graph looks the same and only the legend changes.

- Fixed bug in `ann_utils.subset_ann` where subsetting without specifying `layers` failed with newer anndata versions (≥ 0.12), which expose `X` as the layer `None`. This also fixes `metrics.group_metrics` when both `group_oi` and `group_ref` are given, and thereby `wrappers.generate_info_tables` with `case_control=True` and `wrappers.run_nichenet` with `get_prioritization_table=True` and `case_control=True`.

- Fixed support for dense (`numpy.ndarray`) layers: `metrics.group_metrics` crashed and `extraction.get_lfc_celltype` returned a single value instead of a list of log fold changes. `utils.subset_matrix` is now faster for sparse inputs, and `metrics.gene_expression_pct` no longer modifies its input matrix or counts explicitly stored zeros as expressed.

- Fixed `prioritization.get_avg_exp` raising a `TypeError` when `condition_oi` / `condition_col` weren't given (both default to None); giving only one of them raises a `ValueError`, as in NicheNetR. Passing `features` subset the genes before normalization, so library sizes were computed from the selected genes only (inflating the average expression and changing its relative values across cell types). Results now match Seurat's `AverageExpression`. `features` can now also be a generator.

- Fixed bugs in `visualization.create_radar_plot`: without `max_val` the plot was scaled to the largest second value of the series instead of the largest value (and failed for 1-D data), and a generator for `value_labels` failed. Also corrected the type of `data` to a sequence of sequences of floats or a 2-D `numpy.ndarray` (one row per plotted series).

- Fixed bug in `model_construction.construct_model_from_source_weights` where an empty ligand-signaling network (all ligand-receptor and signaling sources weighted 0) produced a transposed gene regulatory matrix with all genes as columns instead of a targets × ligands matrix. Such models now go through the regular construction path, where each ligand only reaches itself. With all source weights 0 it returns the dictionary of None values instead of a tuple containing it.

- Fixed ligand prediction evaluation for evaluation datasets with a ligand combination: `utils.is_ligand_active` never marked any ligand as active for a combination (it now marks each member, as in NicheNetR, and a combined "A-B" ligand), so these datasets were always dropped from the ligand prediction evaluation and from the corresponding optimization objectives. `evaluation.evaluate_single_importances_ligand_prediction` no longer raises an `IndexError` for a setting with a single test ligand (with `allow_nan=True`).

- Fixed bug in `parameter_optimization.choose_pareto_optimal_solution_normalized` where an objective without variance (the same value for every solution) led to a division by zero, so the first solution was always chosen. Such objectives are now not taken into account.

- Fixed bug in `network.Network.subset_sep` where calling it on a `network.WeightedNetwork` without `to` raised a `ValueError`.

- Fixed `network.Network.__contains__` (`(from, to) in network`): it raised a `KeyError` for an unknown "from" gene and never matched links of a `network.WeightedNetwork` without the exact weight. Membership is now based on (from, to) only and uses a set of (from, to) pairs that is built on the first membership check (much faster for repeated checks) and reset when the links change.

- Fixed `metrics.group_metrics` with `wilcoxon_limma=True`: it failed when `features` was given, and it ignored `min_pct` and `min_abs_lfc`. It now tests the same genes as the default Wilcoxon test; p-values are verified against limma's `rankSumTestWithCorrelation` (as used by Seurat's `FindMarkers(test.use = "wilcox_limma")`).

- Fixed the `min_pct` / `min_abs_lfc` prefilter of `metrics.group_metrics` (and therefore `prioritization.calculate_de`, `wrappers.get_geneset_oi`, `wrappers.run_nichenet` and `wrappers.run_nichenet_cluster_de`): it only used the fraction of expressing cells in the group of interest, so genes lost in that group (e.g. strongly downregulated genes for `geneset="down"`) were never tested, and in one-vs-rest comparisons a gene passing the filter for one group was also reported for all other groups. The filter is now applied per comparison as in Seurat's `FindMarkers` (expressed in at least `min_pct` of the cells of either group and an absolute log fold change of at least `min_abs_lfc`); results are verified against Seurat. The output has a new column `pct_ref` (fraction of expressing cells in the reference cells).

- Fixed `metrics.group_metrics` and `extraction.get_lfc_celltype` failing when `features` is a generator; `get_lfc_celltype` raises a `TypeError` for a single str (it was split into its characters).

- Fixed `io.write_sparse_matrix` and `io.write_ligand_target_matrix` for matrices with a float type other than float64 (e.g. float32): the readers assume float64, so reading such a file raised a `ValueError` or returned wrong values. The data is now converted to float64 when it is written.

- Fixed `metrics.group_metrics` with `group_ref` but without `group_oi`: the log fold changes compared each group with `group_ref`, but the p-values compared it with all other cells, and `group_ref` itself was reported (compared with itself). P-values are now computed against `group_ref` and the reference group is no longer in the output. Also fixed an `IndexError` when no gene passes the `min_pct` / `min_abs_lfc` filters (an empty table is returned), and `group_oi` equal to `group_ref` now raises a `ValueError`.

- Fixed `metrics.calculate_prediction_evaluation_metrics` changing the global warning filters: after each call all warnings were reset to the "default" action (overriding filters set by the user, e.g. warnings as errors), and when an error was raised all warnings stayed silenced. Warnings are now only silenced during the call.

- Fixed `io.read_csc_matrix` and `io.read_csr_matrix` returning a matrix that was too small when its last rows (csc) or last columns (csr) contained only zeros; the shape is now taken from the stored row and column names. Fixed `io.write_sparse_matrix` for matrices with int64 indices (used by scipy for large matrices and after some operations), which could not be read back; the indices are now written as int32. `io.write_sparse_matrix` now raises a `ValueError` if the number of names doesn't match the shape of the matrix or if the matrix is too large for the file format.

- Fixed `io.write_network` and `io.write_weighted_network`, which failed for networks with fewer than 3 genes and used one byte per bit for the gene ids (e.g. 15 instead of 2 bytes for 25,000 genes). Files are now about 2-2.5 times smaller and remain readable by older versions; older files can still be read.

- Fixed the readers in `io` splitting names on any whitespace, which shifted the names of matrices or silently changed the links of networks when a name contained a space or tab. Names are now only separated by newlines, can contain any (UTF-8) character instead of only ASCII, and the writers raise a `ValueError` if a name contains a newline. Existing files can still be read.

- `io.write_ligand_target_matrix` now raises a `ValueError` if the number of names doesn't match the shape of the matrix, instead of writing a file that reads back with a different shape.

- Fixed `prediction.LigandActivityPredictor` warning about a row-major `numpy.ndarray` for matrices with a single row or column (e.g. a single ligand), which are also column-major.

- Fixed `utils.read_matrix_from_csv` with the default arguments, which only read csv files where all names are quoted (as written by R): unquoted files (e.g. written by pandas) silently gave wrong row names, column names and values. Without custom `extract_row_name` / `extract_col_name` the file is now read with `pandas.read_csv`, and "NA" or empty values become NaN instead of raising an error.

- Fixed `visualization.assign_ligands_to_celltype` for dense layers and sparse arrays (it only worked for sparse matrices). The threshold now uses the sample standard deviation (as `sd()` in NicheNetR), so fewer ligands may be assigned to a cell type than before.

- Fixed `visualization.marker_plot`: the size legend showed wrong values (it now shows the value represented by each marker size), all-zero sizes gave markers of size NaN (a division by zero), a large `max_marker_size` raised a `ZeroDivisionError` when `clabel` was given, an int `max_marker_size` was rejected, and the default colors changed between Python sessions. Marker sizes are now relative to the largest absolute value, so negative values no longer give markers larger than `max_marker_size`.

- Fixed `visualization.create_mushroom_plot` sorting the caller's dataframe and adding a "show" column to it, and failing when `top_n` exceeds the number of rows.

- Fixed `visualization.get_ligand_signaling_path` failing when there are no ligands of interest or a ligand has no regulators, and `minmax_scaling=True` giving NaN weights when all weights are equal (they become 0.75; the weights range from 0.75 to 1.75 as in NicheNetR, not to 1 as documented). It no longer turns off pandas' chained assignment warnings globally.

- Fixed `visualization.reorder_labels` failing on matrices with a constant row or column.

- Fixed `wrappers.create_ligand_activity_hist` failing with fewer than 30 ligands; the dashed line now marks the `top_n`-th ligand (new argument, default 30) and spans the full height of the plot.

- Fixed the circos plots in `wrappers` failing for cell types or genes containing "_" and for `opacity` given as a generator; the genes within a sector are now always in the same order. Corrected the documentation of `wrappers.create_ligand_links_circos_plot` (the receiving genes are read from the column named by `dest_name`).

- Fixed `prediction.assess_rf_class_probabilities` removing the genes that aren't in the model from the caller's `geneset` and `background_expressed_genes` sets, and failing when `ligands_oi` is a generator. Added a `random_state` argument (for the fold splits and the random forests) to make the results reproducible.

- Fixed `prioritization.generate_prioritization_table` adding a "rank" column to the caller's `ligand_activities`.

- Fixed `utils.rank_genes_groups_to_dataframe` replacing scanpy's results in `ann.uns["rank_genes_groups"]` by the returned data frame (after which scanpy's functions on these results and a second call failed); `ann.uns` is no longer changed.

- `prioritization.generate_prioritization_table`, `utils.df_grouped_apply` and `ann_utils.subset_ann` no longer turn off pandas' chained assignment warnings globally. `ann_utils.subset_ann` keeps the categories of a categorical column in their original order (they were in a random order).

- Fixed `wrappers.run_nichenet` and `wrappers.run_nichenet_cluster_de` failing when `sender_celltypes` is a generator or a single cell type name (str).

- Fixed `wrappers.run_nichenet` and `wrappers.run_nichenet_cluster_de` failing when the ligand-receptor network contains ligands that aren't in the ligand-target matrix; these ligands are now left out with a warning.

- Fixed the prioritization table of `wrappers.run_nichenet` being based on the ligand activities of all potential ligands instead of the sender-focused ones (the ligands expressed by the sender cell types, as in NicheNetR), which changed the scaled activities and prioritization scores.

- Fixed `wrappers.generate_info_tables` (and therefore `wrappers.run_nichenet` with `get_prioritization_table=True`) always using the "data" and "counts" layers and overwriting `ann.uns["group_metrics"]`. It now has a `layer` argument (passed on by `wrappers.run_nichenet`) with the log-normalized data, from which the average expression is also computed (after expm1, as Seurat's `AverageExpression`; `prioritization.get_avg_exp` has a new `log_normalized` argument for this), so no counts layer is needed; `ann.uns` is no longer changed.

- Fixed `utils.subset_matrix` failing on an empty list of rows or columns. It raises an `IndexError` for a boolean mask that doesn't have the length of its axis (a shorter mask silently selected only the given rows) and a `TypeError` for indices that aren't integers or booleans.

- Fixed `evaluation.EvaluationData` with a custom `ligand_name` (`get_ligands` and `get_applicable_evaluation_datasets` always read the ligand from "from"), and `parameter_optimization.evaluate_model` and `evaluation.get_single_ligand_importances` with evaluation data that uses custom field names (they used the default names "name", "from" and "response"). `EvaluationData` has new read-only properties `key_name`, `ligand_name` and `de_genes_name`.

- Fixed `evaluation.get_single_ligand_importances` failing when `evaluation_data` or `all_ligands` is a generator (they were iterated twice). Corrected its documentation of the expected keys.

- Fixed `wilcoxon.wilcoxon_rank_sum_test_with_correlation` returning the lower and upper tail p-values in swapped order; they are now returned as documented and as limma's `rankSumTestWithCorrelation` ("less", "greater"). The upper tail is computed more precisely, so `metrics.group_metrics` with `wilcoxon_limma=True` matches limma more closely (it doesn't depend on the order).

- Fixed `wilcoxon.wilcoxon_rank_sum_test` (and `metrics.group_metrics`) ranking explicitly stored zeros of a sparse matrix above the implicit zeros, which gave wrong p-values. Explicit zeros are now removed from a copy of the matrix, with a warning.

- Fixed `gene_symbol.GeneInfo` giving no mouse-to-human conversions when created from a generator, and converting mouse symbols to "NA" (and "NA" to a human symbol) instead of None; missing symbols are now left out in both directions. Reading a gene info file with NA Entrez ids no longer fails, and looking up a mouse symbol (`info[symbol]`) or testing membership (`symbol in info`) no longer raises an `AttributeError`.

- Fixed `prediction.LigandActivityPredictor` silently accepting row and column names whose number doesn't match the shape of the ligand-target matrix or that contain duplicates (the names then pointed to wrong or missing rows/columns); it now raises a `ValueError`.

- Single genes (ligands, the ligand of evaluation data) and `val` of `ann_utils.subset_ann` now accept numpy integers (e.g. from a numpy array or DataFrame column), which are converted to `int`, and reject booleans (which were silently used as the gene ids 1 and 0).

- Fixed `evaluation.EvaluationData.get_applicable_evaluation_datasets` (and `parameter_optimization.evaluate_model`) failing with a `TypeError` on a combination of int gene ids; such a combination is matched to the model's column named by the tuple of its ids.

- `model_construction.construct_ligand_tf_matrix` now raises a `TypeError` when a ligand isn't an Iterable of genes; a string was split into its characters.

- `utils.ncycle` rejects an `n` that isn't an int with a clear `TypeError` (a bool was used as 0 or 1). It cycles through an iterator (e.g. a generator) `n` times instead of once, and checks its arguments when it's called instead of when it's iterated.

- Fixed `parameter_optimization.compute_evaluation_scores` with `ligands` as a generator: it's used twice (target and ligand prediction), the second time it was empty, which raised a `ValueError`. Fixed `ann_utils.subset_ann` with `layers` as a generator, which raised a `StopIteration`.

- Fixed `network.WeightedNetwork.get_all` including the link weights, and `to_dataframe` of `network.Network` and `network.WeightedNetwork` failing for an empty network. `to_dataframe` is ~4 times faster and its index now runs from 0 to the number of links.

- Fixed `ann_utils.subset_ann` failing with `subset_X=False` (the output now has no X, which saves subsetting it) and when `genes` is a generator.

- Fixed ligand combinations in `model_construction.construct_ligand_tf_matrix` (and `construct_ligand_target_matrix`) getting the same name as a single ligand: a combination with a member that isn't in the networks was named after the remaining member, and a combination of int gene ids after its first member. Combinations are now named after the requested members ("A-B" for gene symbols, the tuple of ids for int gene ids), two ligands with the same name raise a `ValueError`.

- Fixed `model_construction.construct_weighted_networks` accepting NaN source weights (they were dropped like a weight of 0); they now raise a `ValueError`.

- Fixed `normalization.relative_counts` (and therefore `log_normalize` and `prioritization.get_avg_exp`) turning cells without counts into NaN for dense data (with a divide-by-zero warning for all data); they now stay 0, as in Seurat and scanpy. The result now also has the same type as the input (it was always a `csr_matrix` for sparse input).

- Fixed `normalization.scale_quantile` (and `scale_quantile_adapted`) rejecting lists of numpy numbers (e.g. `list(arr)`) and lists of 1-dimensional arrays. Non-numeric data (bool, str, complex) now always raises a `ValueError`.

- Fixed `prioritization.generate_prioritization_table` failing with an `UnboundLocalError` when `lr_condition_de` is None but the `ligand_condition_specificity` or `receptor_condition_specificity` weight is nonzero; it now raises a `ValueError` that explains this, as NicheNetR.

- Fixed `parameter_optimization.compute_evaluation_scores` choosing a ligand importance metric with a NaN score (e.g. a negative mean `aupr_corrected`) as the best metric when it came first; NaN scores are now never chosen unless all scores are NaN. (NicheNetR selects no metric at all in this case, which makes the ligand prediction objectives NA.)

- Fixed `parameter_optimization.choose_pareto_optimal_solution` normalizing a numpy array of weights in place (changing the caller's array) and failing for an int array or a generator of weights.

- Fixed `normalization.scaling_zscore` with NaN values: a list became all 0 and a numpy array all NaN. NaN values are now ignored when computing the mean and standard deviation and stay NaN, as in NicheNetR (a pandas Series already behaved this way).

- Fixed `normalization.scale_quantile` (and `scale_quantile_adapted`) returning all NaN when the data contains a NaN value; the quantiles are now computed without the NaN values, which stay NaN, as in NicheNetR. In `prioritization.generate_prioritization_table`, a ligand with a NaN activity (e.g. a NaN pearson correlation) made the scaled activity and prioritization score NaN for every interaction; now only that ligand's interactions are NaN.

- Fixed `wrappers.run_nichenet` ignoring `condition_ref` for the gene set of interest: the receiver's condition of interest was compared with all other conditions instead of with the reference condition (as NicheNetR's `FindMarkers(ident.1 = condition_oi, ident.2 = condition_reference)`). This changes the results only with more than two conditions.

- Fixed `prediction.LigandActivityPredictor.predict_single_cell_ligand_activities` silently giving wrong activities when the number of row or column names doesn't match the shape of `expression_scaled` or a name occurs more than once (a name pointed to a wrong or missing row or column); it now raises a `ValueError`, and the error for a cell that isn't in `expression_scaled_rows` names that argument (it said the ligand-target matrix).

- Fixed unclear errors for edge cases: `evaluation.EvaluationData.to_dataframe` of empty evaluation data (a bare `StopIteration`; it now returns an empty data frame with the key, ligand and DE genes columns), `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` with `n=0` (a numpy error; it now gives no targets, and a negative `n` raises a `ValueError`), `model_construction.construct_weighted_networks` with an empty dict of source weights (a zip error; it now gives empty networks, as weights of 0 do) and `extraction.average_expression` when none of the `keys` are values of the `groupby` column ("not enough values to unpack"; it now raises a `ValueError`, as for a missing `groupby` column when `keys` are given).

- Fixed `utils.df_grouped_apply` (used by `prioritization.generate_prioritization_table`) dropping the rows with a missing value (NaN) in the `groupby` column, or failing when the other values are strings; these rows now form a group of their own.

- `normalization.scaling_zscore` accepts any sequence of floats (e.g. a tuple), as `scaling_modified_zscore` does; only a list was accepted.

- Fixed `utils.ligand_activities_df` (and therefore `prioritization.generate_prioritization_table` with ligand activities as a dict or list) putting a ligand's values in the wrong columns when its metrics were in another order than those of the first ligand; the values are now matched by metric name, ligands with different metrics raise a `ValueError`, and no ligands give an empty data frame (it raised an unclear error).

- Dict subclasses (e.g. `collections.OrderedDict`) are accepted wherever a dict is accepted; they were rejected (or iterated as an Iterable): `evaluation.EvaluationData`, `utils.ligand_activities_df`, `utils.combine_dicts`, `prioritization.generate_prioritization_table` (`ligand_activities`, `prioritizing_weights`), `model_construction.construct_weighted_networks`, `construct_model_from_source_weights` and `parameter_optimization.construct_and_evaluate` (`source_weights`), `model_construction.construct_ligand_tf_matrix`, `construct_tf_target_matrix` and `construct_ligand_target_matrix` (`weighted_networks`), `parameter_optimization.compute_evaluation_scores` (`eval_res`, `objective_fs`), `prediction.LigandActivityPredictor.evaluate_target_prediction` (`response`), `gene_symbol.GeneAliasInfo`, `visualization.create_barcode_plot` and the circos plots in `wrappers` (`colors`).

## [2.0.1] - TODO

### Fixed

- Fixed `visualization.get_ligand_signaling_path` choosing the path with a link that occurs more than once in `lr_sig` by the sum of the costs (1 / weight) of its copies, so a duplicated link was less likely to be on the path; the weights of a duplicated link are now summed before the path search, as in the returned signaling network. Networks made by `model_construction.construct_weighted_networks` have no duplicated links.



[2.0.1]: https://github.com/saeyslab/nichenetpy/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/saeyslab/nichenetpy/compare/v1.0.4...v2.0.0
[1.0.4]: https://github.com/saeyslab/nichenetpy/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/saeyslab/nichenetpy/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/saeyslab/nichenetpy/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/saeyslab/nichenetpy/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/saeyslab/nichenetpy/releases/tag/v1.0.0
[@vmedaert]: https://github.com/vmedaert

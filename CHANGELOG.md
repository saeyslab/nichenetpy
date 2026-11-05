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

- metrics.precision_at_k

- metrics.average_precision_at_k

- metrics.dcg_at_k

- metrics.ndcg_at_k

- You can now pass `ligand_evaluation_metrics` and/or `target_evaluation_metrics` to evaluation / objective functions.

- You can now pass `ligand_evaluation_metric` and/or `target_evaluation_metric` to functions that always used `aupr_corrected` prior to this version.

- Added optional arguments to split matrices during model construction.

### Changes

- Type signatures have been altered and static type checking has been enabled for the package. This is not considered backwards compatible!

### Fixed

- Corrected type signatures.

- Corrected docstrings.

- Fixed bug in `prediction.replace_zero_col_by_noisy_scores` where sparse matrices were not supported.

- Fixed bug in `io.write_ligand_target_matrix` where sparse matrices were not supported.

- Fixed bug in `visualization.create_barcode_plot` where the last ranked element was omitted.

- Fixed bugs in type checks at runtime.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where the down cutoff was positive.

- Fixed bug in `wrappers.run_nichenet_cluster_de` where the geneset argument was passed to `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` in stead of geneset_oi.

- Fixed bug in `model_construction.construct_model_from_source_weights` where tft and ltf matrices from `model_construction.construct_ligand_target_matrix` were swapped.

- Fixed bug in `model_construction.construct_ligand_target_matrix` where remove_direct_links argument did nothing.

- Fixed `model_construction.construct_ligand_target_matrix` with `secondary_targets=True`, which previously always raised a `ValueError`. The secondary-target scores are now computed as in NicheNetR: the quantile cutoff (`ltf_cutoff`) is applied only to the matrix used to infer secondary targets, and the primary ligand-target scores are left unclipped.

- Fixed bug in `wrappers.run_nichenet` which occurs when `case_control` is False.

- Fixed bug in `wrappers.combine_weighted_ligand_target_links` that occurs when a ligand has no targets.

- Fixed bug in `prediction.LigandActivityPredictor.get_weighted_ligand_target_links` that occurs when the ligand-target matrix is sparse.

- Added shape to `lr_sig_mat` construction in `visualization.get_ligand_signaling_path` which fixes an edge case bug.

- Fixed swapped colors in `visualization/visualize_ligand_signaling_graph`.

[1.0.4]: https://github.com/saeyslab/nichenetpy/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/saeyslab/nichenetpy/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/saeyslab/nichenetpy/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/saeyslab/nichenetpy/compare/v1.0.0...v1.0.1
[@vmedaert]: https://github.com/vmedaert

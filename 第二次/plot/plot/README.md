# Plot Folder

This folder collects the plotting code, plotting data, source image asset, and current rendered figure outputs used by the manuscript.

## Layout

- `code/`: source-preserving copies of the current plotting scripts.
- `data/main_results/`: input tables for the main benchmark, DML, and transfer figures.
- `data/figure_experiments/`: diagnostic figure input tables.
- `data/bookdash_stress/`: BookDash stress-test input tables and run metadata.
- `assets/`: source asset for the current framework overview image.
- `figures/`: exact copied PDF/PNG figure outputs currently used by `paper/main.tex`.
- `FILE_MANIFEST.csv`: file inventory with size and SHA-256 hashes.

## Figure Mapping

| Manuscript figure file | Code source | Data source |
| --- | --- | --- |
| `framework_overview.pdf/png` | external source image asset, not the old Matplotlib schematic | `assets/framework_overview_source_candidate_1.png` |
| `main_results.pdf/png` | `code/make_figures.py` | `data/main_results/server_v2_cv_metrics.csv` |
| `dml_visual_gain.pdf/png` | `code/make_figures.py` | `data/main_results/server_v2_cv_metrics.csv` |
| `screening_transfer.pdf/png` | `code/make_figures.py` | `data/main_results/server_v2_tier_metrics.csv`, `data/main_results/server_v2_transfer_metrics.csv` |
| `oof_regression_marginal.pdf/png` | `code/make_diagnostic_figures.py` | `data/figure_experiments/oof_regression_predictions.csv` |
| `feature_corr_pie_heatmap.pdf/png` | `code/make_diagnostic_figures.py` | `data/figure_experiments/correlation_panel_data.csv` |
| `tier_fan_confusion.pdf/png` | `code/make_diagnostic_figures.py` | `data/figure_experiments/tier_diagnostic_predictions.csv`, `data/figure_experiments/tier_diagnostic_metrics.csv` |
| `pca_tier_violin.pdf/png` | `code/make_diagnostic_figures.py` | `data/figure_experiments/pca_tier_coordinates.csv`, `data/figure_experiments/diagnostic_summary.json` |
| `bookdash_stress_test.pdf/png` | `code/server_bookdash_stress_test.py` | `data/bookdash_stress/` |
| `bookdash_degradation_examples.pdf/png` | `code/server_bookdash_stress_test.py` | `data/bookdash_stress/` |

## Notes

- The copied scripts preserve their original path assumptions. The canonical runnable locations remain `paper/` for `make_figures.py` and `make_diagnostic_figures.py`, and `experiment/` for `server_bookdash_stress_test.py`.
- The current `framework_overview.pdf` was manually replaced from `candidate_1.png`. Rerunning the old `framework_figure()` function in `make_figures.py` would regenerate the earlier schematic rather than this current image-based overview.
- `run_experiment_legacy_figures.py` is retained because it contains older plotting functions, but its output figures are not part of the current manuscript figure set.

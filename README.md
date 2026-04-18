# SEHS5052 Topic E Project (CICIDS-2017)

This project implements **Topic E: Explainable AI (XAI) for SOC Decision Support** for SEHS5052.
It trains a baseline and a black-box model on CICIDS-2017, generates SHAP/LIME explanations, builds SOC alert cases, and exports report-ready artifacts.

## 1. Scope and Requirement Mapping

This codebase is aligned to Topic E requirements:

- Train at least two models:
  - Baseline: Logistic Regression
  - Black-box: Random Forest
- Explainability:
  - SHAP global importance
  - SHAP summary plot
  - SHAP waterfall plots (5 selected cases)
  - LIME local explanations for the same cases
  - SHAP vs LIME comparison table
- SOC casebook:
  - 5 selected cases with actionable text
  - TP/FP/FN coverage strategy
- Evaluation:
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrices
  - Overfitting diagnostics (train/val/test comparison)
- EDA:
  - Class distribution
  - Correlation heatmap

## 2. Repository Structure

- `src/run_experiment.py`: end-to-end pipeline entry
- `src/data_pipeline.py`: dataset loading, cleaning, sampling, split
- `src/train_models.py`: model training and evaluation
- `src/explainability.py`: SHAP/LIME generation and plot export
- `src/soc_simulation.py`: SOC case selection, casebook, analyst metrics
- `src/logging_utils.py`: centralized logging setup
- `tests/`: unit tests
- `artifacts/`: generated tables, figures, logs
- `report/SEHS5052_Topic_E_Report.md`: coursework report draft (Markdown; figures use `artifacts/figures/…` paths relative to the **repo root** for IDE preview)

## 3. Data Preparation

This project expects data to be **already extracted** (ZIP auto-extraction is disabled).

Place CSV files under:

- `data/cicids/`

Recommended source file family for this project:

- `MachineLearningCSV` (preferred)

## 4. Environment Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple pytest pandas numpy scikit-learn matplotlib seaborn shap lime
```

## 5. Run the Full Pipeline

Recommended run (balanced speed + full Topic E outputs):

```bash
.venv/bin/python src/run_experiment.py \
  --data-dir data \
  --max-rows 80000 \
  --sample-size 30000 \
  --rf-n-estimators 80 \
  --rf-max-depth 16 \
  --output-dir artifacts \
  --log-level INFO \
  --log-file artifacts/full_pipeline.log
```

Notes:

- Do **not** pass `--disable-lime` for final submission runs, because Topic E requires SHAP/LIME comparison.
- Sampling is performed once before split; all models use the same sampled data.

## 6. Key CLI Arguments

- `--data-dir`: data root (default `data`)
- `--cicids-path`: explicit CICIDS CSV path (optional)
- `--cicids-label-col`: label column name (default `Label`)
- `--max-rows`: bounded initial load (`-1` means full)
- `--sample-size`: random stratified sample size before split
- `--rf-n-estimators`: RandomForest tree count
- `--rf-max-depth`: RandomForest max depth (`-1` means None)
- `--rf-min-samples-leaf`: RandomForest min leaf samples
- `--disable-lime`: skip LIME for quick debug only
- `--log-level`: `DEBUG|INFO|WARNING|ERROR`
- `--log-file`: log file path

## 7. Output Artifacts

### Tables (`artifacts/tables/`)

- `model_metrics.csv`
- `overfitting_diagnostics.csv`
- `eda_label_distribution.csv`
- `shap_global_importance.csv`
- `soc_alert_report.csv`
- `shap_lime_case_comparison.csv`

### Figures (`artifacts/figures/`)

- `cm_baseline.png`
- `cm_blackbox.png`
- `eda_class_distribution.png`
- `eda_correlation_heatmap.png`
- `overfitting_f1_by_split.png`
- `shap_summary.png`
- `shap_waterfall_case_*.png` (5 cases)

### Summary

- `artifacts/run_summary.json`

Important fields:

- `sample_size`
- `effective_rows`
- `case_presence` (should include `TP`, `FP`, `FN` as `true`)
- `elapsed_seconds`

## 8. Validate Requirement Coverage

After running, verify:

1. `run_summary.json` contains:
   - `case_presence.TP == true`
   - `case_presence.FP == true`
   - `case_presence.FN == true`
2. SHAP and LIME outputs both exist:
   - `shap_summary.png`
   - `shap_waterfall_case_*.png`
   - `shap_lime_case_comparison.csv`
3. Evaluation and overfitting artifacts exist:
   - `model_metrics.csv`
   - `overfitting_diagnostics.csv`
   - confusion matrix images
4. EDA artifacts exist:
   - class distribution + correlation heatmap

## 9. Testing

```bash
.venv/bin/python -m pytest -q
```

## 10. Reproducibility

- Random seed is fixed in code (`RANDOM_STATE = 42`).
- Sampling is deterministic under the same seed and parameters.

## 11. Practical Tips

- If runtime is too long, reduce:
  - `--sample-size`
  - `--rf-n-estimators`
  - `--rf-max-depth`
- For final report figures, keep SHAP/LIME enabled and use stable parameters.

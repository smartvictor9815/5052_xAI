#!/usr/bin/env python3
"""Generate Colab-ready topic_e_xai_soc.ipynb from src/*.py sources."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "topic_e_xai_soc.ipynb"


def read_src(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def lines_to_nb_source(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            line += "\n"
        out.append(line)
    return out


RUN_PIPELINE = r'''# End-to-end run (mirrors src/run_experiment.py)

def _load_main_dataset(
    path: str | None,
    label_col: str,
    max_rows: int | None = None,
    dataset_name: str | None = None,
    data_dir: str | None = None,
    class_balance_retry_cap: int | None = 300000,
) -> DatasetBundle:
    csv_candidates: list = []
    if path is not None:
        raw = None
        if dataset_name and data_dir:
            dataset_dir = Path(data_dir) / dataset_name
            csv_candidates = sorted(dataset_dir.rglob("*.csv")) if dataset_dir.exists() else []
            if len(csv_candidates) > 1:
                frames = []
                loaded = 0
                for csv_path in csv_candidates:
                    remaining = None if max_rows is None else max_rows - loaded
                    if remaining is not None and remaining <= 0:
                        break
                    chunk = load_csv(str(csv_path), nrows=remaining)
                    if chunk.empty:
                        continue
                    frames.append(chunk)
                    loaded += len(chunk)
                if frames:
                    raw = pd.concat(frames, ignore_index=True)

        if raw is None:
            raw = load_csv(path, nrows=max_rows)
        bundle = unify_binary_labels(raw, label_col=label_col)
        if max_rows is not None and bundle.y.nunique() < 2:
            if csv_candidates:
                frames = [load_csv(str(p), nrows=None) for p in csv_candidates]
                raw = pd.concat(frames, ignore_index=True)
                if class_balance_retry_cap is not None and len(raw) > class_balance_retry_cap:
                    raw = raw.sample(n=class_balance_retry_cap, random_state=42)
            else:
                retry_rows = class_balance_retry_cap if class_balance_retry_cap is not None else None
                raw = load_csv(path, nrows=retry_rows)
            bundle = unify_binary_labels(raw, label_col=label_col)
        cleaned_X = basic_cleaning(bundle.X)
        return DatasetBundle(
            X=cleaned_X,
            y=bundle.y.loc[cleaned_X.index].reset_index(drop=True),
            attack_type=bundle.attack_type.loc[cleaned_X.index].reset_index(drop=True),
        )
    return build_synthetic_dataset()


def _save_confusion_matrix(cm, out_path: Path, title: str) -> None:
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _generate_eda_outputs(bundle: DatasetBundle, figs_dir: Path, tables_dir: Path) -> None:
    label_counts = bundle.y.value_counts().rename(index={0: "Benign", 1: "Attack"}).sort_index()
    label_counts.to_csv(tables_dir / "eda_label_distribution.csv", header=["count"])

    plt.figure(figsize=(5, 4))
    sns.barplot(x=label_counts.index.tolist(), y=label_counts.values.tolist(), hue=label_counts.index.tolist(), palette="Blues_d", legend=False)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figs_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    numeric_cols = bundle.X.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_cols:
        top_cols = numeric_cols[:20]
        corr_sample = bundle.X[top_cols]
        if len(corr_sample) > 10000:
            corr_sample = corr_sample.sample(n=10000, random_state=42)
        corr = corr_sample.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0.0)
        plt.title("Feature Correlation Heatmap (Top Numeric Features)")
        plt.tight_layout()
        plt.savefig(figs_dir / "eda_correlation_heatmap.png", dpi=150)
        plt.close()


def run_topic_e() -> dict:
    start = time.perf_counter()
    max_rows = None if MAX_ROWS < 0 else MAX_ROWS
    rf_max_depth = None if RF_MAX_DEPTH < 0 else RF_MAX_DEPTH
    cb_cap = None if CLASS_BALANCE_RETRY_CAP < 0 else CLASS_BALANCE_RETRY_CAP

    out_dir = Path(OUTPUT_DIR)
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if LOG_FILE:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(fh)

    cicids_path = ensure_dataset_csv(
        dataset_name="cicids",
        data_root=DATA_DIR,
        preferred_path=CICIDS_PATH,
    )
    if cicids_path is None:
        raise FileNotFoundError(
            f"No CICIDS CSV found under data directory '{DATA_DIR}'. "
            "Place extracted CSV files in data/cicids/ (or set CICIDS_PATH)."
        )

    bundle = _load_main_dataset(
        cicids_path,
        CICIDS_LABEL_COL,
        max_rows=max_rows,
        dataset_name="cicids",
        data_dir=DATA_DIR,
        class_balance_retry_cap=cb_cap,
    )
    bundle = sample_dataset_bundle(bundle, sample_size=SAMPLE_SIZE)
    _generate_eda_outputs(bundle, figs_dir=figs_dir, tables_dir=tables_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(bundle.X, bundle.y)

    baseline = train_baseline(X_train, y_train, X_val, y_val)
    blackbox = train_blackbox(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=RF_N_ESTIMATORS,
        max_depth=rf_max_depth,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    )

    baseline_test_metrics, baseline_cm, _ = evaluate_on_split(
        baseline.pipeline, baseline.threshold, X_test, y_test
    )
    blackbox_test_metrics, blackbox_cm, blackbox_probs = evaluate_on_split(
        blackbox.pipeline, blackbox.threshold, X_test, y_test
    )

    overfit_rows = []
    for model_name, model_artifacts in (("baseline", baseline), ("blackbox", blackbox)):
        for split_name, X_split, y_split in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            split_metrics, _, _ = evaluate_on_split(
                model_artifacts.pipeline,
                model_artifacts.threshold,
                X_split,
                y_split,
            )
            overfit_rows.append({"model": model_name, "split": split_name, **split_metrics})
    overfit_df = pd.DataFrame(overfit_rows)
    overfit_df.to_csv(tables_dir / "overfitting_diagnostics.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=overfit_df, x="split", y="f1", hue="model")
    plt.title("F1 by Split (Overfitting Check)")
    plt.tight_layout()
    plt.savefig(figs_dir / "overfitting_f1_by_split.png", dpi=150)
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {"model": baseline.name, **baseline_test_metrics},
            {"model": blackbox.name, **blackbox_test_metrics},
        ]
    )
    metrics_df.to_csv(tables_dir / "model_metrics.csv", index=False)

    _save_confusion_matrix(baseline_cm, figs_dir / "cm_baseline.png", "Baseline Confusion Matrix")
    _save_confusion_matrix(blackbox_cm, figs_dir / "cm_blackbox.png", "Black-box Confusion Matrix")

    case_ids, forced_case_types, case_thresholds, case_presence = choose_required_soc_cases(
        y_true=y_test.reset_index(drop=True),
        y_prob=blackbox_probs,
        default_threshold=blackbox.threshold,
        max_cases=5,
    )
    case_threshold = float(np.mean(list(case_thresholds.values()))) if case_thresholds else float(blackbox.threshold)

    xai = run_explainability(
        blackbox.pipeline,
        X_train,
        X_test,
        sample_ids=case_ids,
        top_k=8,
        enable_lime=not DISABLE_LIME,
        plot_output_dir=str(figs_dir),
    )
    if xai.shap_global is not None:
        xai.shap_global.to_csv(tables_dir / "shap_global_importance.csv", index=False)
    if xai.shap_summary_plot:
        print("SHAP summary:", xai.shap_summary_plot)
    if xai.shap_waterfall_plots:
        print("SHAP waterfalls:", len(xai.shap_waterfall_plots), "files")

    case_report = build_soc_alert_report(
        y_true=y_test.reset_index(drop=True),
        y_prob=blackbox_probs,
        threshold=blackbox.threshold,
        shap_local=xai.shap_local_top_features,
        lime_local=xai.lime_local_top_features,
        case_ids=case_ids,
        forced_case_types=forced_case_types,
        case_thresholds=case_thresholds,
    )
    case_report.to_csv(tables_dir / "soc_alert_report.csv", index=False)

    comparison_df = build_shap_lime_comparison_table(
        case_report=case_report,
        shap_local=xai.shap_local_top_features,
        lime_local=xai.lime_local_top_features,
        top_k=5,
    )
    comparison_df.to_csv(tables_dir / "shap_lime_case_comparison.csv", index=False)

    analyst_metrics = compute_analyst_metrics(
        case_report,
        shap_local=xai.shap_local_top_features,
        lime_local=xai.lime_local_top_features,
    )

    summary = {
        "baseline_test_metrics": baseline_test_metrics,
        "blackbox_test_metrics": blackbox_test_metrics,
        "analyst_metrics": analyst_metrics,
        "cicids_path": cicids_path,
        "sample_size": SAMPLE_SIZE,
        "effective_rows": len(bundle.X),
        "case_threshold": case_threshold,
        "case_presence": case_presence,
        "output_dir": str(out_dir.resolve()),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


summary = run_topic_e()
summary
'''


def main() -> None:
    cells: list[dict] = []

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SEHS5052 Topic E — Explainable AI for SOC (CICIDS-2017)\n",
                "\n",
                "This notebook mirrors `src/run_experiment.py`: **CICIDS-2017 only**, pre-extracted CSVs under `data/cicids/` (no ZIP extraction).\n",
                "\n",
                "**Colab**:\n",
                "1. Upload this notebook and your `data/cicids/` folder (or mount Google Drive).\n",
                "2. Adjust `DATA_DIR` / `CICIDS_PATH` in the configuration cell.\n",
                "3. Run all cells. Outputs go to `OUTPUT_DIR` (default `artifacts/`): tables, figures, `run_summary.json`.\n",
                "\n",
                "Section headers **Report Part 2–4** align with the written report (`report/SEHS5052_Topic_E_Report.md`).\n",
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines_to_nb_source(
                "%pip install -q numpy pandas scikit-learn matplotlib seaborn shap lime\n"
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines_to_nb_source(
                """from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# --- Colab / local paths (edit for your environment) ---
DATA_DIR = "data"
CICIDS_PATH: str | None = None  # e.g. "/content/drive/MyDrive/.../some_cicids.csv"
CICIDS_LABEL_COL = "Label"
MAX_ROWS = 300_000
SAMPLE_SIZE = 120_000
CLASS_BALANCE_RETRY_CAP = 300_000
RF_N_ESTIMATORS = 120
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2
DISABLE_LIME = False
OUTPUT_DIR = "artifacts"
LOG_FILE: str | None = None  # e.g. str(Path(OUTPUT_DIR) / "run.log")
"""
            ),
        }
    )

    # Markdown headers mirror Written Report Parts 2–4 (brief PDF requirement).
    sections: list[tuple[str, str]] = [
        (
            "## Report Part 2 — Dataset description and preprocessing\n\n"
            "Matches **Written Report Part 2**: CICIDS load, cleaning, stratified sampling, train/val/test split. "
            "Source module: `src/data_pipeline.py`.\n",
            "data_pipeline.py",
        ),
        (
            "## Report Part 3 — AI model design and implementation\n\n"
            "### Baseline and black-box models (`train_models.py`)\n",
            "train_models.py",
        ),
        (
            "### Explainability — SHAP and LIME (`explainability.py`)\n",
            "explainability.py",
        ),
        (
            "### SOC alert casebook (`soc_simulation.py`)\n",
            "soc_simulation.py",
        ),
    ]
    for md_text, fname in sections:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines_to_nb_source(md_text)})
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines_to_nb_source(read_src(fname)),
            }
        )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines_to_nb_source(
                "## Report Part 4 — Evaluation and XAI-focused analysis\n\n"
                "Matches **Written Report Part 4**: test metrics, confusion matrices, overfitting diagnostics, "
                "SHAP summary/waterfalls, LIME comparison, SOC report. "
                "Run the cell below to write all tables and figures under `OUTPUT_DIR`.\n"
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines_to_nb_source(RUN_PIPELINE),
        }
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

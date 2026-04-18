"""End-to-end Topic E experiment runner (py-first phase)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_pipeline import (
    DatasetBundle,
    basic_cleaning,
    build_synthetic_dataset,
    ensure_dataset_csv,
    load_csv,
    sample_dataset_bundle,
    split_train_val_test,
    unify_binary_labels,
)
from explainability import run_explainability
from logging_utils import configure_logging
from soc_simulation import (
    build_shap_lime_comparison_table,
    build_soc_alert_report,
    choose_required_soc_cases,
    compute_analyst_metrics,
)
from train_models import evaluate_on_split, train_baseline, train_blackbox

logger = logging.getLogger(__name__)


def _load_main_dataset(
    path: str | None,
    label_col: str,
    max_rows: int | None = None,
    dataset_name: str | None = None,
    data_dir: str | None = None,
    class_balance_retry_cap: int | None = 300000,
) -> DatasetBundle:
    csv_candidates = []
    if path is not None:
        logger.info("Loading primary dataset from %s", path)
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
                    logger.info("Merged %d CSV files from %s", len(frames), dataset_dir)

        if raw is None:
            raw = load_csv(path, nrows=max_rows)
        bundle = unify_binary_labels(raw, label_col=label_col)
        if max_rows is not None and bundle.y.nunique() < 2:
            # Small head-only slices of CICIDS often contain just benign traffic.
            # Retry with a bounded upper limit instead of loading the entire dataset.
            if csv_candidates:
                frames = [load_csv(str(p), nrows=None) for p in csv_candidates]
                raw = pd.concat(frames, ignore_index=True)
                if class_balance_retry_cap is not None and len(raw) > class_balance_retry_cap:
                    raw = raw.sample(n=class_balance_retry_cap, random_state=42)
                logger.warning(
                    "max_rows sample had one class; retried with expanded bounded sample from %d CSV files.",
                    len(csv_candidates),
                )
            else:
                retry_rows = class_balance_retry_cap if class_balance_retry_cap is not None else None
                raw = load_csv(path, nrows=retry_rows)
                logger.warning("max_rows sample had one class; retried with bounded load from %s.", path)
            bundle = unify_binary_labels(raw, label_col=label_col)
        cleaned_X = basic_cleaning(bundle.X)
        logger.info("Main dataset ready. features=%d rows=%d", cleaned_X.shape[1], cleaned_X.shape[0])
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
    """Generate required EDA visuals for report Part 2."""
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
        # Keep heatmap readable and computationally bounded.
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
    logger.info("Saved EDA outputs to %s and %s", figs_dir, tables_dir)


def run(args: argparse.Namespace) -> Dict[str, object]:
    start = time.perf_counter()
    logger.info("Starting experiment run with args=%s", vars(args))
    out_dir = Path(args.output_dir)
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts directory prepared at %s", out_dir.resolve())

    cicids_path = ensure_dataset_csv(
        dataset_name="cicids",
        data_root=args.data_dir,
        preferred_path=args.cicids_path,
    )
    if cicids_path is None:
        raise FileNotFoundError(
            f"No CICIDS CSV found under data directory '{args.data_dir}'. "
            "Please place extracted CSV files in data/cicids."
        )
    logger.info("Resolved CICIDS path: %s", cicids_path)
    bundle = _load_main_dataset(
        cicids_path,
        args.cicids_label_col,
        max_rows=args.max_rows,
        dataset_name="cicids",
        data_dir=args.data_dir,
        class_balance_retry_cap=args.class_balance_retry_cap,
    )
    bundle = sample_dataset_bundle(bundle, sample_size=args.sample_size)
    _generate_eda_outputs(bundle, figs_dir=figs_dir, tables_dir=tables_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(bundle.X, bundle.y)
    logger.info("Data split completed. Train labels distribution: %s", y_train.value_counts().to_dict())

    baseline = train_baseline(X_train, y_train, X_val, y_val)
    blackbox = train_blackbox(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
    )

    baseline_test_metrics, baseline_cm, _ = evaluate_on_split(
        baseline.pipeline, baseline.threshold, X_test, y_test
    )
    blackbox_test_metrics, blackbox_cm, blackbox_probs = evaluate_on_split(
        blackbox.pipeline, blackbox.threshold, X_test, y_test
    )

    # Overfitting diagnostics: train/val/test metrics for both models.
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
    logger.info("Saved overfitting diagnostics to %s", (tables_dir / "overfitting_diagnostics.csv").resolve())

    metrics_df = pd.DataFrame(
        [
            {"model": baseline.name, **baseline_test_metrics},
            {"model": blackbox.name, **blackbox_test_metrics},
        ]
    )
    metrics_df.to_csv(tables_dir / "model_metrics.csv", index=False)
    logger.info("Saved metrics table to %s", (tables_dir / "model_metrics.csv").resolve())

    _save_confusion_matrix(baseline_cm, figs_dir / "cm_baseline.png", "Baseline Confusion Matrix")
    _save_confusion_matrix(blackbox_cm, figs_dir / "cm_blackbox.png", "Black-box Confusion Matrix")
    logger.info("Saved confusion matrices under %s", figs_dir.resolve())

    case_ids, forced_case_types, case_thresholds, case_presence = choose_required_soc_cases(
        y_true=y_test.reset_index(drop=True),
        y_prob=blackbox_probs,
        default_threshold=blackbox.threshold,
        max_cases=5,
    )
    case_threshold = float(np.mean(list(case_thresholds.values()))) if case_thresholds else float(blackbox.threshold)
    logger.info("Selected case set TP/FP/FN presence=%s", case_presence)

    xai = run_explainability(
        blackbox.pipeline,
        X_train,
        X_test,
        sample_ids=case_ids,
        top_k=8,
        enable_lime=not args.disable_lime,
        plot_output_dir=str(figs_dir),
    )
    if xai.shap_global is not None:
        xai.shap_global.to_csv(tables_dir / "shap_global_importance.csv", index=False)
        logger.info("Saved SHAP global importance to %s", (tables_dir / "shap_global_importance.csv").resolve())
    if xai.shap_summary_plot:
        logger.info("Saved SHAP summary plot to %s", xai.shap_summary_plot)
    if xai.shap_waterfall_plots:
        logger.info("Saved %d SHAP waterfall plots.", len(xai.shap_waterfall_plots))

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
    logger.info("Saved SOC alert report to %s", (tables_dir / "soc_alert_report.csv").resolve())

    comparison_df = build_shap_lime_comparison_table(
        case_report=case_report,
        shap_local=xai.shap_local_top_features,
        lime_local=xai.lime_local_top_features,
        top_k=5,
    )
    comparison_df.to_csv(tables_dir / "shap_lime_case_comparison.csv", index=False)
    logger.info(
        "Saved SHAP/LIME comparison table to %s",
        (tables_dir / "shap_lime_case_comparison.csv").resolve(),
    )

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
        "sample_size": args.sample_size,
        "effective_rows": len(bundle.X),
        "case_threshold": case_threshold,
        "case_presence": case_presence,
        "output_dir": str(out_dir.resolve()),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Run completed successfully in %.2fs", time.perf_counter() - start)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Topic E experiment."""
    parser = argparse.ArgumentParser(description="Topic E XAI SOC experiment runner (CICIDS-2017 only).")
    parser.add_argument("--cicids-path", type=str, default=None, help="Local path to CICIDS CSV/ZIP.")
    parser.add_argument("--cicids-label-col", type=str, default="Label", help="Label column for CICIDS.")
    parser.add_argument("--data-dir", type=str, default="data", help="Local data root directory.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300000,
        help="Maximum rows loaded before sampling (set -1 for full dataset).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120000,
        help="Randomly sample this many rows before train/val/test split (all models share same sample).",
    )
    parser.add_argument(
        "--class-balance-retry-cap",
        type=int,
        default=300000,
        help="Upper bound rows for class-balance retry when max_rows sample has only one class (-1 disables cap).",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=120, help="RandomForest n_estimators.")
    parser.add_argument("--rf-max-depth", type=int, default=20, help="RandomForest max_depth (-1 means None).")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2, help="RandomForest min_samples_leaf.")
    parser.add_argument("--disable-lime", action="store_true", help="Disable LIME to reduce runtime.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    return parser


def get_default_run_config() -> Dict[str, object]:
    """
    Return defaults used by CLI flags.

    Notebook should read these defaults to avoid drifting from this file.
    """
    defaults = vars(build_parser().parse_args([]))
    return {
        "data_dir": defaults["data_dir"],
        "cicids_path": defaults["cicids_path"],
        "cicids_label_col": defaults["cicids_label_col"],
        "max_rows": defaults["max_rows"],
        "sample_size": defaults["sample_size"],
        "class_balance_retry_cap": defaults["class_balance_retry_cap"],
        "rf_n_estimators": defaults["rf_n_estimators"],
        "rf_max_depth": defaults["rf_max_depth"],
        "rf_min_samples_leaf": defaults["rf_min_samples_leaf"],
        "disable_lime": defaults["disable_lime"],
        "output_dir": defaults["output_dir"],
        "log_level": defaults["log_level"],
        "log_file": defaults["log_file"],
    }


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 0:
        args.max_rows = None
    if args.class_balance_retry_cap is not None and args.class_balance_retry_cap < 0:
        args.class_balance_retry_cap = None
    if args.rf_max_depth is not None and args.rf_max_depth < 0:
        args.rf_max_depth = None
    return args


if __name__ == "__main__":
    args = parse_args()
    default_log_path = Path(args.output_dir) / "run.log" if args.log_file is None else args.log_file
    configure_logging(level=args.log_level, log_file=str(default_log_path))
    result = run(args)
    print(json.dumps(result, indent=2))

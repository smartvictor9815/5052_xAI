"""SHAP/LIME utilities for SOC explainability workflow."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityArtifacts:
    shap_global: Optional[pd.DataFrame]
    shap_local_top_features: Dict[int, List[Tuple[str, float]]]
    lime_local_top_features: Dict[int, List[Tuple[str, float]]]
    feature_names: List[str]
    shap_summary_plot: Optional[str]
    shap_waterfall_plots: Dict[int, str]


def _extract_transformed_space(
    model_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_samples: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    preprocessor = model_pipe.named_steps["preprocessor"]
    model = model_pipe.named_steps["model"]
    X_train_t = preprocessor.transform(X_train)
    X_samples_t = preprocessor.transform(X_samples)
    feature_names = preprocessor.get_feature_names_out().tolist()
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    else:
        X_train_t = np.asarray(X_train_t)
    if hasattr(X_samples_t, "toarray"):
        X_samples_t = X_samples_t.toarray()
    else:
        X_samples_t = np.asarray(X_samples_t)
    X_train_t = X_train_t.astype(np.float64, copy=False)
    X_samples_t = X_samples_t.astype(np.float64, copy=False)
    return X_train_t, X_samples_t, feature_names


def _compute_shap(
    model_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_samples: pd.DataFrame,
    sample_ids: List[int],
    top_k: int = 8,
    max_global_samples: int = 4000,
) -> Tuple[Optional[pd.DataFrame], Dict[int, List[Tuple[str, float]]], List[str]]:
    try:
        import shap
    except Exception as exc:
        logger.warning("SHAP unavailable: %s", exc)
        return None, {}, []

    X_train_t, X_samples_t, feature_names = _extract_transformed_space(model_pipe, X_train, X_samples)
    model = model_pipe.named_steps["model"]
    logger.info("Running SHAP explainability for %d samples (top_k=%d).", len(sample_ids), top_k)

    explainer = shap.TreeExplainer(model)
    rng = np.random.default_rng(42)
    if X_samples_t.shape[0] > max_global_samples:
        chosen = rng.choice(X_samples_t.shape[0], size=max_global_samples, replace=False)
        X_global = X_samples_t[chosen]
    else:
        X_global = X_samples_t

    shap_values_global = explainer.shap_values(X_global)
    if isinstance(shap_values_global, list):
        shap_class_global = np.asarray(shap_values_global[-1])
    else:
        shap_class_global = np.asarray(shap_values_global)
        if shap_class_global.ndim == 3:
            shap_class_global = shap_class_global[:, :, -1]

    mean_abs = np.mean(np.abs(shap_class_global), axis=0)
    shap_global = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    local = {}
    for sid in sample_ids:
        single = explainer.shap_values(X_samples_t[sid : sid + 1])
        if isinstance(single, list):
            vals = np.asarray(single[-1])[0]
        else:
            arr = np.asarray(single)
            if arr.ndim == 3:
                vals = arr[0, :, -1]
            else:
                vals = arr[0]
        top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
        local[sid] = [(feature_names[i], float(vals[i])) for i in top_idx]
    return shap_global, local, feature_names


def _compute_lime(
    model_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_samples: pd.DataFrame,
    sample_ids: List[int],
    top_k: int = 8,
) -> Dict[int, List[Tuple[str, float]]]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        logger.warning("LIME unavailable: %s", exc)
        return {}

    preprocessor = model_pipe.named_steps["preprocessor"]
    model = model_pipe.named_steps["model"]
    X_train_t = preprocessor.transform(X_train)
    X_samples_t = preprocessor.transform(X_samples)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_dense = X_train_t.toarray() if hasattr(X_train_t, "toarray") else np.asarray(X_train_t)
    X_samples_dense = X_samples_t.toarray() if hasattr(X_samples_t, "toarray") else np.asarray(X_samples_t)

    explainer = LimeTabularExplainer(
        training_data=X_train_dense,
        feature_names=feature_names,
        class_names=["Benign", "Attack"],
        mode="classification",
        discretize_continuous=True,
    )

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        return model.predict_proba(arr)

    local = {}
    for sid in sample_ids:
        exp = explainer.explain_instance(X_samples_dense[sid], predict_fn, num_features=top_k)
        entries = []
        for feat_desc, weight in exp.as_list():
            # LIME returns formatted rule text; keep raw text for analyst readability.
            entries.append((feat_desc, float(weight)))
        local[sid] = entries
    logger.info("Generated LIME local explanations for %d samples.", len(local))
    return local


def run_explainability(
    model_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_ids: List[int],
    top_k: int = 8,
    enable_lime: bool = True,
    plot_output_dir: Optional[str] = None,
) -> ExplainabilityArtifacts:
    """Compute SHAP global/local and LIME local explanations."""
    logger.info("Starting explainability stage.")
    shap_global, shap_local, feature_names = _compute_shap(model_pipe, X_train, X_test, sample_ids, top_k=top_k)
    if enable_lime:
        lime_local = _compute_lime(model_pipe, X_train, X_test, sample_ids, top_k=top_k)
    else:
        logger.info("LIME disabled by configuration; skipping LIME explanations.")
        lime_local = {}
    logger.info(
        "Explainability complete: shap_global=%s shap_local=%d lime_local=%d",
        "yes" if shap_global is not None else "no",
        len(shap_local),
        len(lime_local),
    )

    shap_summary_plot = None
    shap_waterfall_plots: Dict[int, str] = {}
    if plot_output_dir is not None:
        shap_summary_plot, shap_waterfall_plots = export_shap_plots(
            model_pipe=model_pipe,
            X_train=X_train,
            X_test=X_test,
            sample_ids=sample_ids,
            output_dir=plot_output_dir,
        )

    return ExplainabilityArtifacts(
        shap_global=shap_global,
        shap_local_top_features=shap_local,
        lime_local_top_features=lime_local,
        feature_names=feature_names,
        shap_summary_plot=shap_summary_plot,
        shap_waterfall_plots=shap_waterfall_plots,
    )


def export_shap_plots(
    model_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_ids: List[int],
    output_dir: str,
    max_global_samples: int = 4000,
) -> Tuple[Optional[str], Dict[int, str]]:
    """
    Export SHAP summary plot and waterfall plots for selected cases.
    Returns (summary_plot_path, {sample_id: waterfall_path}).
    """
    try:
        import shap
    except Exception as exc:
        logger.warning("Skip SHAP plot export because shap import failed: %s", exc)
        return None, {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train_t, X_test_t, feature_names = _extract_transformed_space(model_pipe, X_train, X_test)
    model = model_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)

    rng = np.random.default_rng(42)
    if X_test_t.shape[0] > max_global_samples:
        chosen = rng.choice(X_test_t.shape[0], size=max_global_samples, replace=False)
        X_global = X_test_t[chosen]
    else:
        X_global = X_test_t
    X_global_df = pd.DataFrame(X_global, columns=feature_names)

    shap_values_global = explainer.shap_values(X_global)
    if isinstance(shap_values_global, list):
        shap_global_vals = np.asarray(shap_values_global[-1])
    else:
        shap_global_vals = np.asarray(shap_values_global)
        if shap_global_vals.ndim == 3:
            shap_global_vals = shap_global_vals[:, :, -1]

    summary_path = str(out_dir / "shap_summary.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_global_vals, X_global_df, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()
    logger.info("Saved SHAP summary plot to %s", summary_path)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_scalar = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        expected_scalar = float(expected_value)

    waterfalls: Dict[int, str] = {}
    for sid in sample_ids:
        if sid < 0 or sid >= X_test_t.shape[0]:
            continue
        single_vals_raw = explainer.shap_values(X_test_t[sid : sid + 1])
        if isinstance(single_vals_raw, list):
            single_vals = np.asarray(single_vals_raw[-1])[0]
        else:
            arr = np.asarray(single_vals_raw)
            if arr.ndim == 3:
                single_vals = arr[0, :, -1]
            else:
                single_vals = arr[0]

        explanation = shap.Explanation(
            values=single_vals,
            base_values=expected_scalar,
            data=X_test_t[sid],
            feature_names=feature_names,
        )
        out_path = str(out_dir / f"shap_waterfall_case_{sid}.png")
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        waterfalls[sid] = out_path

    logger.info("Saved %d SHAP waterfall plots to %s", len(waterfalls), out_dir)
    return summary_path, waterfalls

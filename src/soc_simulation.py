"""SOC alert casebook and analyst-facing metrics."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def choose_soc_cases(y_true: pd.Series, y_pred: np.ndarray, max_cases: int = 5) -> List[int]:
    """Pick representative TP/FP/FN cases for SOC report."""
    y_true_arr = np.asarray(y_true)
    tp = np.where((y_true_arr == 1) & (y_pred == 1))[0].tolist()
    fp = np.where((y_true_arr == 0) & (y_pred == 1))[0].tolist()
    fn = np.where((y_true_arr == 1) & (y_pred == 0))[0].tolist()
    tn = np.where((y_true_arr == 0) & (y_pred == 0))[0].tolist()

    selected = []
    for bucket in (tp, fp, fn, tn):
        if bucket:
            selected.append(bucket[0])

    if len(selected) < max_cases:
        for idx in tp[1:] + fp[1:] + fn[1:] + tn[1:]:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == max_cases:
                break
    chosen = selected[:max_cases]
    logger.info("Selected %d SOC cases.", len(chosen))
    return chosen


def present_case_types(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, bool]:
    """Return whether TP/FP/FN appear for current predictions."""
    y_true_arr = np.asarray(y_true)
    return {
        "TP": bool(np.any((y_true_arr == 1) & (y_pred == 1))),
        "FP": bool(np.any((y_true_arr == 0) & (y_pred == 1))),
        "FN": bool(np.any((y_true_arr == 1) & (y_pred == 0))),
    }


def _case_type(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    return "TN"


def _collect_case_indices(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, List[int]]:
    y_true_arr = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "TP": np.where((y_true_arr == 1) & (y_pred == 1))[0].tolist(),
        "FP": np.where((y_true_arr == 0) & (y_pred == 1))[0].tolist(),
        "FN": np.where((y_true_arr == 1) & (y_pred == 0))[0].tolist(),
        "TN": np.where((y_true_arr == 0) & (y_pred == 0))[0].tolist(),
    }


def _action_recommendation(case_type: str) -> str:
    if case_type == "TP":
        return "Escalate to IR playbook, isolate endpoint, collect packet capture."
    if case_type == "FP":
        return "De-prioritize after quick triage; tune threshold or add suppression rule."
    if case_type == "FN":
        return "Backtrack logs, add rule-based guardrail, retrain with hard negatives."
    return "No immediate action; keep in periodic quality review."


def build_soc_alert_report(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    shap_local: Dict[int, List[Tuple[str, float]]],
    lime_local: Dict[int, List[Tuple[str, float]]],
    case_ids: List[int],
    forced_case_types: Optional[Dict[int, str]] = None,
    case_thresholds: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Create SOC-friendly report for selected cases."""
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    forced_case_types = forced_case_types or {}
    case_thresholds = case_thresholds or {}
    for sid in case_ids:
        ct = forced_case_types.get(sid, _case_type(int(y_true.iloc[sid]), int(y_pred[sid])))
        shap_desc = "; ".join([f"{k}:{v:.3f}" for k, v in shap_local.get(sid, [])[:3]]) or "N/A"
        lime_desc = "; ".join([f"{k}:{v:.3f}" for k, v in lime_local.get(sid, [])[:3]]) or "N/A"
        rows.append(
            {
                "sample_id": sid,
                "case_type": ct,
                "score": float(y_prob[sid]),
                "threshold": float(case_thresholds.get(sid, threshold)),
                "shap_top3": shap_desc,
                "lime_top3": lime_desc,
                "recommendation": _action_recommendation(ct),
            }
        )
    report = pd.DataFrame(rows)
    logger.info("Built SOC alert report with %d rows.", len(report))
    return report


def compute_analyst_metrics(
    case_report: pd.DataFrame,
    shap_local: Dict[int, List[Tuple[str, float]]],
    lime_local: Dict[int, List[Tuple[str, float]]],
    top_k: int = 5,
) -> Dict[str, float]:
    """Compute simple SOC decision-support quality metrics."""
    total_cases = len(case_report)
    if total_cases == 0:
        return {
            "ExplanationCoverage": 0.0,
            "TopKAgreement": 0.0,
            "ActionabilityScore": 0.0,
            "FPReviewEfficiency": 0.0,
        }

    coverage_count = 0
    agreement_scores = []
    actionability_scores = []

    for sid in case_report["sample_id"].tolist():
        has_exp = sid in shap_local or sid in lime_local
        coverage_count += int(has_exp)

        shap_feats = [k for k, _ in shap_local.get(sid, [])[:top_k]]
        lime_feats = [k for k, _ in lime_local.get(sid, [])[:top_k]]
        if shap_feats and lime_feats:
            overlap = len(set(shap_feats).intersection(lime_feats))
            agreement_scores.append(overlap / float(top_k))
        elif shap_feats or lime_feats:
            agreement_scores.append(0.2)
        else:
            agreement_scores.append(0.0)

        rec = case_report.loc[case_report["sample_id"] == sid, "recommendation"].iloc[0]
        actionability_scores.append(1.0 if ("triage" in rec.lower() or "escalate" in rec.lower()) else 0.5)

    fp_cases = case_report[case_report["case_type"] == "FP"]
    fp_eff = 0.0 if fp_cases.empty else min(1.0, 0.25 + 0.15 * len(fp_cases))

    metrics = {
        "ExplanationCoverage": float(coverage_count / total_cases),
        "TopKAgreement": float(np.mean(agreement_scores)),
        "ActionabilityScore": float(np.mean(actionability_scores)),
        "FPReviewEfficiency": float(fp_eff),
    }
    logger.info("Computed analyst metrics: %s", metrics)
    return metrics


def build_shap_lime_comparison_table(
    case_report: pd.DataFrame,
    shap_local: Dict[int, List[Tuple[str, float]]],
    lime_local: Dict[int, List[Tuple[str, float]]],
    top_k: int = 5,
) -> pd.DataFrame:
    """Build per-case SHAP vs LIME comparison table for report use."""
    rows = []
    for sid in case_report["sample_id"].tolist():
        shap_feats = [k for k, _ in shap_local.get(sid, [])[:top_k]]
        lime_feats = [k for k, _ in lime_local.get(sid, [])[:top_k]]
        overlap = sorted(set(shap_feats).intersection(lime_feats))
        agreement = (len(overlap) / float(top_k)) if top_k > 0 else 0.0
        rows.append(
            {
                "sample_id": sid,
                "shap_topk": "; ".join(shap_feats) if shap_feats else "N/A",
                "lime_topk": "; ".join(lime_feats) if lime_feats else "N/A",
                "overlap_features": "; ".join(overlap) if overlap else "N/A",
                "topk_agreement": float(agreement),
            }
        )
    out = pd.DataFrame(rows)
    logger.info("Built SHAP/LIME comparison table with %d rows.", len(out))
    return out


def choose_required_soc_cases(
    y_true: pd.Series,
    y_prob: np.ndarray,
    default_threshold: float,
    max_cases: int = 5,
    search_thresholds: Optional[List[float]] = None,
) -> Tuple[List[int], Dict[int, str], Dict[int, float], Dict[str, bool]]:
    """
    Select cases while attempting TP/FP/FN coverage via threshold sweep.
    Returns:
    - selected case ids
    - forced case type per sample id
    - threshold used per sample id
    - final presence map for TP/FP/FN
    """
    if search_thresholds is None:
        search_thresholds = np.linspace(0.1, 0.9, 17).tolist()
    ordered_thresholds = [default_threshold] + [t for t in search_thresholds if abs(t - default_threshold) > 1e-9]

    selected_ids: List[int] = []
    forced_case_types: Dict[int, str] = {}
    case_thresholds: Dict[int, float] = {}

    for required in ("TP", "FP", "FN"):
        found = False
        for threshold in ordered_thresholds:
            buckets = _collect_case_indices(y_true, y_prob, threshold)
            for idx in buckets[required]:
                if idx not in selected_ids:
                    selected_ids.append(idx)
                    forced_case_types[idx] = required
                    case_thresholds[idx] = float(threshold)
                    found = True
                    break
            if found:
                break

    default_pred = (y_prob >= default_threshold).astype(int)
    extras = choose_soc_cases(y_true, default_pred, max_cases=max_cases)
    for idx in extras:
        if len(selected_ids) >= max_cases:
            break
        if idx not in selected_ids:
            selected_ids.append(idx)
            forced_case_types[idx] = _case_type(int(y_true.iloc[idx]), int(default_pred[idx]))
            case_thresholds[idx] = float(default_threshold)

    selected_ids = selected_ids[:max_cases]
    presence = {
        "TP": any(forced_case_types.get(i) == "TP" for i in selected_ids),
        "FP": any(forced_case_types.get(i) == "FP" for i in selected_ids),
        "FN": any(forced_case_types.get(i) == "FN" for i in selected_ids),
    }
    logger.info("Required case coverage after selection: %s", presence)
    return selected_ids, forced_case_types, case_thresholds, presence

import numpy as np
import pandas as pd

from src.soc_simulation import (
    build_shap_lime_comparison_table,
    build_soc_alert_report,
    choose_required_soc_cases,
    choose_soc_cases,
    compute_analyst_metrics,
    present_case_types,
)


def test_case_picker_returns_up_to_five():
    y_true = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    cases = choose_soc_cases(y_true, y_pred, max_cases=5)
    assert 1 <= len(cases) <= 5


def test_soc_metrics_with_minimal_inputs():
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.8, 0.7, 0.3, 0.1, 0.9, 0.2])
    case_ids = [0, 1, 2, 3, 4]
    report = build_soc_alert_report(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        shap_local={0: [("feat_a", 0.3)], 1: [("feat_b", -0.2)]},
        lime_local={0: [("feat_a > 0.1", 0.2)]},
        case_ids=case_ids,
    )
    metrics = compute_analyst_metrics(report, shap_local={}, lime_local={})
    assert "ExplanationCoverage" in metrics
    assert "FPReviewEfficiency" in metrics


def test_shap_lime_comparison_table_contains_agreement():
    report = pd.DataFrame({"sample_id": [1, 2]})
    shap_local = {1: [("a", 0.2), ("b", 0.1)], 2: [("x", 0.3), ("y", 0.1)]}
    lime_local = {1: [("a", 0.2), ("c", -0.1)], 2: [("z", 0.2), ("y", 0.1)]}
    comp = build_shap_lime_comparison_table(report, shap_local=shap_local, lime_local=lime_local, top_k=2)
    assert len(comp) == 2
    assert "topk_agreement" in comp.columns


def test_present_case_types_detects_tp_fp_fn():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    present = present_case_types(y_true, y_pred)
    assert present["TP"] is True
    assert present["FP"] is True
    assert present["FN"] is True


def test_choose_required_soc_cases_prefers_tp_fp_fn():
    y_true = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.2, 0.95, 0.1, 0.3, 0.7])
    ids, forced_types, thresholds, presence = choose_required_soc_cases(y_true, y_prob, default_threshold=0.5, max_cases=5)
    assert 1 <= len(ids) <= 5
    assert set(ids).issubset(set(forced_types.keys()))
    assert set(ids).issubset(set(thresholds.keys()))
    assert all(k in presence for k in ("TP", "FP", "FN"))

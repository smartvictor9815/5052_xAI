"""Model training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    name: str
    pipeline: Pipeline
    threshold: float
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for mixed tabular features."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def select_threshold_by_f1(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Find probability threshold that maximizes F1 on validation set."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute required classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> ModelArtifacts:
    """Train baseline logistic regression model."""
    logger.info("Training baseline LogisticRegression model.")
    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    threshold = select_threshold_by_f1(y_val, val_prob)
    metrics, cm = compute_metrics(y_val, val_prob, threshold)
    logger.info("Baseline trained. threshold=%.3f metrics=%s", threshold, metrics)
    return ModelArtifacts("logistic_regression", pipe, threshold, metrics, cm)


def train_blackbox(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 120,
    max_depth: int | None = 20,
    min_samples_leaf: int = 2,
) -> ModelArtifacts:
    """Train black-box model (RF default)."""
    logger.info(
        "Training black-box RandomForest model (n_estimators=%d, max_depth=%s, min_samples_leaf=%d).",
        n_estimators,
        str(max_depth),
        min_samples_leaf,
    )
    preprocessor = build_preprocessor(X_train)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    threshold = select_threshold_by_f1(y_val, val_prob)
    metrics, cm = compute_metrics(y_val, val_prob, threshold)
    logger.info("Black-box trained. threshold=%.3f metrics=%s", threshold, metrics)
    return ModelArtifacts("random_forest", pipe, threshold, metrics, cm)


def evaluate_on_split(
    model_pipe: Pipeline,
    threshold: float,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate trained model on a dataset split."""
    probs = model_pipe.predict_proba(X)[:, 1]
    metrics, cm = compute_metrics(y, probs, threshold)
    logger.info("Evaluation metrics=%s", metrics)
    return metrics, cm, probs

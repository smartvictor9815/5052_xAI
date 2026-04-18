"""Data loading and preprocessing helpers for Topic E experiments."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container that keeps features and labels together."""

    X: pd.DataFrame
    y: pd.Series
    attack_type: pd.Series


def load_csv(path_or_url: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a CSV file from local path or URL."""
    logger.debug("Loading CSV from %s (nrows=%s)", path_or_url, nrows)
    try:
        df = pd.read_csv(path_or_url, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying with latin1.", path_or_url)
        df = pd.read_csv(path_or_url, nrows=nrows, encoding="latin1", low_memory=False)
    logger.debug("Loaded dataframe shape from %s: %s", path_or_url, df.shape)
    return df


def unify_binary_labels(
    df: pd.DataFrame,
    label_col: str,
    benign_tokens: Tuple[str, ...] = ("benign", "normal"),
) -> DatasetBundle:
    """Map dataset labels to binary labels: benign=0, attack=1."""
    if label_col not in df.columns:
        # CIC/UNSW CSVs often contain extra spaces in header names.
        normalized_map = {str(c).strip().lower(): c for c in df.columns}
        resolved = normalized_map.get(label_col.strip().lower())
        if resolved is None:
            raise ValueError(f"Label column '{label_col}' not found.")
        label_col = resolved

    attack_type = df[label_col].astype(str)
    norm = attack_type.str.strip().str.lower()
    benign_mask = norm.isin(set(benign_tokens))
    y = (~benign_mask).astype(int)

    X = df.drop(columns=[label_col]).copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info(
        "Mapped labels using '%s': benign=%d attack=%d",
        label_col,
        int((y == 0).sum()),
        int((y == 1).sum()),
    )
    return DatasetBundle(X=X, y=y, attack_type=attack_type)


def basic_cleaning(X: pd.DataFrame, missing_ratio_threshold: float = 0.4) -> pd.DataFrame:
    """Drop highly missing columns and remove duplicate rows."""
    cleaned = X.copy()
    missing_ratio = cleaned.isna().mean()
    to_drop = missing_ratio[missing_ratio > missing_ratio_threshold].index.tolist()
    if to_drop:
        cleaned = cleaned.drop(columns=to_drop)
    cleaned = cleaned.drop_duplicates()
    logger.info(
        "Basic cleaning completed: input=%s output=%s dropped_cols=%d",
        X.shape,
        cleaned.shape,
        len(to_drop),
    )
    return cleaned


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train/val/test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio_in_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_in_train_val,
        stratify=y_train_val,
        random_state=random_state,
    )
    logger.info("Split sizes -> train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def sample_dataset_bundle(
    bundle: DatasetBundle,
    sample_size: Optional[int],
    random_state: int = RANDOM_STATE,
) -> DatasetBundle:
    """
    Randomly sample a subset of rows once, then reuse for all models.
    Stratified sampling is used when both classes exist.
    """
    if sample_size is None or sample_size <= 0 or sample_size >= len(bundle.X):
        return bundle

    indices = np.arange(len(bundle.X))
    y = bundle.y.reset_index(drop=True)
    stratify = y if y.nunique() > 1 else None
    sampled_idx, _ = train_test_split(
        indices,
        train_size=sample_size,
        random_state=random_state,
        stratify=stratify,
    )
    sampled_idx = np.sort(sampled_idx)
    logger.info(
        "Random sampled dataset: original=%d sampled=%d stratified=%s",
        len(bundle.X),
        sample_size,
        "yes" if stratify is not None else "no",
    )
    return DatasetBundle(
        X=bundle.X.iloc[sampled_idx].reset_index(drop=True),
        y=bundle.y.iloc[sampled_idx].reset_index(drop=True),
        attack_type=bundle.attack_type.iloc[sampled_idx].reset_index(drop=True),
    )


def align_common_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only common columns for cross-dataset evaluation."""
    common = sorted(set(df_a.columns).intersection(df_b.columns))
    if not common:
        raise ValueError("No common columns found between datasets.")
    return df_a[common].copy(), df_b[common].copy()


def build_synthetic_dataset(n_samples: int = 2000, random_state: int = RANDOM_STATE) -> DatasetBundle:
    """Build synthetic tabular data for smoke tests and quick demo runs."""
    rng = np.random.default_rng(seed=random_state)
    X = pd.DataFrame(
        {
            "FlowDuration": rng.normal(200, 40, n_samples),
            "TotFwdPkts": rng.normal(20, 6, n_samples),
            "TotBwdPkts": rng.normal(18, 7, n_samples),
            "PktLenMean": rng.normal(500, 120, n_samples),
            "Protocol": rng.choice(["TCP", "UDP", "ICMP"], n_samples, p=[0.7, 0.25, 0.05]),
            "Flag": rng.choice(["S0", "SF", "REJ"], n_samples, p=[0.15, 0.75, 0.10]),
        }
    )
    score = (
        0.007 * X["FlowDuration"]
        + 0.04 * X["TotFwdPkts"]
        + 0.03 * X["TotBwdPkts"]
        + 0.002 * X["PktLenMean"]
        + (X["Protocol"] == "ICMP").astype(int) * 1.6
        + (X["Flag"] == "S0").astype(int) * 1.2
        + rng.normal(0, 0.8, n_samples)
    )
    y = (score > np.quantile(score, 0.68)).astype(int)
    attack_type = pd.Series(np.where(y == 1, "Attack", "Benign"), name="Label")
    return DatasetBundle(X=X, y=pd.Series(y, name="y"), attack_type=attack_type)


def _dataset_dir(data_root: str | Path, dataset_name: str) -> Path:
    root = Path(data_root)
    return root / dataset_name.lower()


def _find_first_csv(target_dir: Path, dataset_name: Optional[str] = None) -> Optional[Path]:
    if not target_dir.exists():
        return None
    csvs = sorted(target_dir.rglob("*.csv"))
    if not csvs:
        return None

    if dataset_name and dataset_name.lower() == "cicids":
        # Prefer MachineLearningCSV-derived files over GeneratedLabelledFlows files.
        preferred = [p for p in csvs if "machinelearning" in str(p).lower()]
        if preferred:
            return preferred[0]
    return csvs[0]


def ensure_dataset_csv(
    dataset_name: str,
    data_root: str | Path = "data",
    preferred_path: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve dataset CSV path from:
    1) preferred_path if exists (CSV only)
    2) existing CSV files under data/<dataset_name>/

    Note:
    - ZIP extraction is intentionally disabled.
    - Datasets are expected to be pre-extracted by the user.
    """
    data_root_path = Path(data_root)
    data_root_path.mkdir(parents=True, exist_ok=True)

    if preferred_path:
        path_obj = Path(preferred_path)
        if path_obj.exists():
            logger.info("Using preferred dataset path: %s", path_obj)
            if path_obj.suffix.lower() == ".csv":
                return str(path_obj)
            logger.warning("Preferred path exists but is not a CSV: %s", path_obj)
            return None

    dataset_dir = _dataset_dir(data_root_path, dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    discovered = _find_first_csv(dataset_dir, dataset_name=dataset_name)
    if discovered:
        logger.info("Found existing dataset CSV: %s", discovered)
        return str(discovered)
    logger.info("No CSV found in %s. Please pre-extract dataset files first.", dataset_dir)
    return None

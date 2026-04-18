from pathlib import Path

import pandas as pd

from src.data_pipeline import (
    _find_first_csv,
    align_common_columns,
    build_synthetic_dataset,
    ensure_dataset_csv,
    sample_dataset_bundle,
    split_train_val_test,
    unify_binary_labels,
)


def test_unify_binary_labels_maps_benign_to_zero():
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "Label": ["Benign", "DDoS", "Normal", "PortScan"],
        }
    )
    bundle = unify_binary_labels(df, label_col="Label")
    assert bundle.y.tolist() == [0, 1, 0, 1]


def test_split_preserves_total_rows():
    bundle = build_synthetic_dataset(n_samples=500)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(bundle.X, bundle.y)
    assert len(X_train) + len(X_val) + len(X_test) == 500
    assert len(y_train) + len(y_val) + len(y_test) == 500


def test_align_common_columns_returns_intersection():
    a = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    b = pd.DataFrame({"y": [2], "z": [3], "k": [4]})
    aa, bb = align_common_columns(a, b)
    assert aa.columns.tolist() == ["y", "z"]
    assert bb.columns.tolist() == ["y", "z"]


def test_ensure_dataset_csv_uses_existing_preferred_path(tmp_path: Path):
    csv_path = tmp_path / "my.csv"
    pd.DataFrame({"a": [1], "Label": ["Benign"]}).to_csv(csv_path, index=False)
    resolved = ensure_dataset_csv("cicids", data_root=tmp_path, preferred_path=str(csv_path))
    assert resolved == str(csv_path)


def test_ensure_dataset_csv_finds_existing_dataset_folder_csv(tmp_path: Path):
    ds_dir = tmp_path / "unsw"
    ds_dir.mkdir(parents=True)
    csv_path = ds_dir / "part.csv"
    pd.DataFrame({"a": [1], "label": [0]}).to_csv(csv_path, index=False)
    resolved = ensure_dataset_csv("unsw", data_root=tmp_path, preferred_path=None)
    assert resolved == str(csv_path)


def test_ensure_dataset_csv_returns_none_when_only_zip_exists(tmp_path: Path):
    data_root = tmp_path / "data"
    ds_dir = data_root / "cicids"
    ds_dir.mkdir(parents=True)
    (ds_dir / "MachineLearningCSV.zip").write_text("dummy", encoding="utf-8")

    resolved = ensure_dataset_csv("cicids", data_root=data_root, preferred_path=None)
    assert resolved is None


def test_find_first_csv_prefers_machinelearning_for_cicids(tmp_path: Path):
    target_dir = tmp_path / "cicids"
    target_dir.mkdir(parents=True)
    glf_csv = target_dir / "TrafficLabelling" / "a.csv"
    ml_csv = target_dir / "MachineLearningCSV" / "b.csv"
    glf_csv.parent.mkdir(parents=True)
    ml_csv.parent.mkdir(parents=True)
    glf_csv.write_text("x,Label\n1,Benign\n", encoding="utf-8")
    ml_csv.write_text("x,Label\n2,Attack\n", encoding="utf-8")

    chosen = _find_first_csv(target_dir, dataset_name="cicids")
    assert chosen is not None
    assert "machinelearningcsv" in str(chosen).lower()


def test_sample_dataset_bundle_reduces_rows_and_keeps_alignment():
    bundle = build_synthetic_dataset(n_samples=1000)
    sampled = sample_dataset_bundle(bundle, sample_size=200)
    assert len(sampled.X) == 200
    assert len(sampled.y) == 200
    assert len(sampled.attack_type) == 200
    assert sampled.X.index.tolist() == list(range(200))


def test_ensure_dataset_csv_ignores_bad_zip_and_uses_valid_csv(tmp_path: Path):
    data_root = tmp_path / "data"
    cicids_dir = data_root / "cicids"
    cicids_dir.mkdir(parents=True)

    bad_zip = cicids_dir / "broken.zip"
    bad_zip.write_text("this is not a zip archive", encoding="utf-8")

    valid_csv = cicids_dir / "MachineLearningCSV_valid.csv"
    valid_csv.write_text("x,Label\n1,Benign\n2,Attack\n", encoding="utf-8")

    resolved = ensure_dataset_csv("cicids", data_root=data_root, preferred_path=None)
    assert resolved is not None
    assert "machinelearningcsv" in str(resolved).lower()

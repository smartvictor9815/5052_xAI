from src.data_pipeline import build_synthetic_dataset, split_train_val_test
from src.train_models import evaluate_on_split, train_baseline, train_blackbox


def test_models_return_required_metrics():
    bundle = build_synthetic_dataset(n_samples=800)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(bundle.X, bundle.y)

    baseline = train_baseline(X_train, y_train, X_val, y_val)
    blackbox = train_blackbox(X_train, y_train, X_val, y_val)

    for art in (baseline, blackbox):
        metrics, cm, probs = evaluate_on_split(art.pipeline, art.threshold, X_test, y_test)
        assert set(["accuracy", "precision", "recall", "f1", "roc_auc"]).issubset(metrics.keys())
        assert cm.shape == (2, 2)
        assert len(probs) == len(X_test)

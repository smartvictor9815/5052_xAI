# Contributing Guide

This project is a coursework repository for SEHS5052 Topic E.
Keep contributions reproducible, testable, and easy for teammates to review.

## 1. Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

## 2. Branch and Commit Style

- Create a feature/fix branch from the main working branch.
- Use focused commits for single concerns.
- Use clear messages (e.g., `add SHAP waterfall export`).

## 3. Run Before Commit

```bash
.venv/bin/python -m pytest -q
```

If you use pre-commit:

```bash
pre-commit install
pre-commit run --all-files
```

## 4. Coding Standards

- Keep modules under `src/` focused and reusable.
- Add/maintain unit tests under `tests/` when behavior changes.
- Prefer deterministic behavior (fixed random seed where appropriate).
- Do not commit large generated outputs or dataset files.

## 5. Data and Outputs

- Put extracted CICIDS CSV files in `data/cicids/`.
- Generated outputs should go to `artifacts/` and are ignored by git.

## 6. Pull Request Checklist

- [ ] Code runs end-to-end (`src/run_experiment.py`)
- [ ] Tests pass
- [ ] New behavior is documented in `README.md` if needed
- [ ] No secrets, credentials, or personal data committed

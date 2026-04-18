# Report (Markdown)

- **[SEHS5052_Topic_E_Report.md](SEHS5052_Topic_E_Report.md)** — full SEHS5052 Topic E written report draft (Parts 1–5, references, Colab checklist).

## Image paths

Figures in `SEHS5052_Topic_E_Report.md` use paths like `artifacts/figures/<name>.png`. The folder **`report/artifacts`** is a **symlink** to `../artifacts` so previews that resolve URLs **relative to the Markdown file** still find the PNGs.

Create or repair the link (Unix/macOS) from the repository root:

```bash
ln -sf ../artifacts report/artifacts
```

On Windows (Developer Mode or admin `cmd`): `mklink /D report\artifacts ..\artifacts`

| Asset | Path in Markdown |
|-------|------------------|
| EDA, metrics, SHAP, CM, overfitting PNGs | `artifacts/figures/<name>.png` |

After running `src/run_experiment.py`, regenerate or refresh `artifacts/` so previews in Cursor/VS Code resolve correctly.

## Regenerate numbers and plots

From the repository root (see main [README.md](../README.md)):

```bash
.venv/bin/python src/run_experiment.py \
  --data-dir data \
  --max-rows 80000 \
  --sample-size 30000 \
  --rf-n-estimators 80 \
  --rf-max-depth 16 \
  --output-dir artifacts \
  --log-level INFO
```

Then update any hard-coded numeric paragraphs in the report if you change CLI parameters.

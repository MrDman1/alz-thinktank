# alz-thinktank (Path B scaffold)

## Quickstart
1) Create synthetic data:
```bash
python scripts/synthetic_mci.py
```
2) Run the pipeline:
```bash
python orchestrator/run.py
```

Artifacts will appear in `artifacts/`:
- `datacard.json` (schema checks, split plan)
- `processed.parquet`
- `metrics.json`
- `pred_risk_hist.png`
- `referee_report.json`

## LLM Client
- Default is `ManualBackend` (no API required).
- Later, switch to `OpenAIBackend` in `orchestrator/llm_client.py` when you want API-driven agents.

## Dependencies (minimal)
- pandas, numpy, lifelines, scikit-learn, matplotlib, pyyaml, pyarrow

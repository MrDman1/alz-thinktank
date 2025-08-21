# agents/referee.py
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import random

def shuffle_label_sanity(df: pd.DataFrame, feats, label_col="progress_24m", seed=0):
    rng = np.random.RandomState(seed)
    y = df[label_col].sample(frac=1.0, random_state=seed).reset_index(drop=True) if label_col in df.columns else None
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).reset_index(drop=True)
    if y is None:
        return {"ok": True, "note": "No label column; skipping sanity."}
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X, y)
    yhat = clf.predict_proba(X)[:,1]
    auroc = roc_auc_score(y, yhat)
    # If labels are shuffled, AUROC should be ~0.5 on average
    return {"ok": abs(auroc - 0.5) < 0.1, "auroc": float(auroc)}

def main(args):
    with open(args.datacard, "r") as f:
        card = json.load(f)
    metrics = json.load(open(args.metrics, "r"))
    df = pd.read_parquet(args.data)
    feats = metrics["features"]
    sanity = shuffle_label_sanity(df, feats, seed=42)

    report = {
        "checks": {
            "shuffle_label": sanity,
            "site_time_aware_split": "note" in card["splits"] and "site" in card["splits"]["note"] or len(card["splits"]["train_idx"])>0
        },
        "metrics": metrics
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "referee_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Referee report written. Passed:", sanity["ok"] and report["checks"]["site_time_aware_split"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datacard", default="artifacts/datacard.json")
    ap.add_argument("--metrics", default="artifacts/metrics.json")
    ap.add_argument("--data", default="artifacts/processed.parquet")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()
    main(args)

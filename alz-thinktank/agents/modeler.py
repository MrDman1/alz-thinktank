# agents/modeler.py
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_splits(datacard):
    return datacard["splits"]["train_idx"], datacard["splits"]["val_idx"], datacard["splits"]["test_idx"]

def prepare_time_to_event(df: pd.DataFrame):
    # For synthetic demo: time horizon ~24 months (simulate)
    # We assume 'progress_24m' indicates event by 24m; treat time as 24 for events, 24 for censored.
    T = np.full(len(df), 24.0)
    E = df["progress_24m"].astype(int).values if "progress_24m" in df.columns else np.zeros(len(df))
    return T, E

def model_features(df: pd.DataFrame):
    cand = ["age","mmse","adas","mri_hippocampus","csf_abeta","csf_tau"]
    feats = [c for c in cand if c in df.columns]
    return feats

def main(args):
    df = pd.read_parquet(args.data)
    with open(args.datacard, "r") as f:
        card = json.load(f)
    tr, va, te = load_splits(card)

    # features/labels
    feats = model_features(df)
    if not feats:
        raise RuntimeError("No model features found; check columns.")
    T, E = prepare_time_to_event(df)

    # CoxPH on train
    train_df = df.iloc[tr].copy()
    val_df = df.iloc[va].copy()
    test_df = df.iloc[te].copy()

    # lifelines expects a dataframe with time & event columns
    for col in feats:
        for d in (train_df, val_df, test_df):
            d[col] = pd.to_numeric(d[col], errors="coerce")
    train_df["T"], train_df["E"] = T[tr], E[tr]
    val_df["T"], val_df["E"] = T[va], E[va]
    test_df["T"], test_df["E"] = T[te], E[te]

    cph = CoxPHFitter()
    cph.fit(train_df[feats+["T","E"]], duration_col="T", event_col="E")
    c_index_val = cph.concordance_index_(val_df[feats+["T","E"]], event_col="E", duration_col="T")
    c_index_test = cph.concordance_index_(test_df[feats+["T","E"]], event_col="E", duration_col="T")

    # Simple classification baseline for demo
    # Predict progress_24m using GBM
    if "progress_24m" in df.columns:
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = df["progress_24m"].astype(int)
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict_proba(X_te)[:,1]
        auroc = roc_auc_score(y_te, y_pred)
        aupr = average_precision_score(y_te, y_pred)
    else:
        auroc = None
        aupr = None

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "c_index_val": float(c_index_val),
        "c_index_test": float(c_index_test),
        "auroc_test": float(auroc) if auroc is not None else None,
        "aupr_test": float(aupr) if aupr is not None else None,
        "features": feats,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

    # Calibration-like plot (risk score histogram as a placeholder demo)
    plt.figure()
    if "progress_24m" in df.columns:
        plt.hist(y_pred, bins=20)
        plt.title("Predicted risk distribution (test)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "pred_risk_hist.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/processed.parquet")
    ap.add_argument("--datacard", default="artifacts/datacard.json")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()
    main(args)

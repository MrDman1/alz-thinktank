# agents/curator.py
import argparse, json, yaml, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

def normalize_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    colmap = {}
    canon = schema["columns"]
    for canon_name, candidates in canon.items():
        for c in candidates:
            if c in df.columns:
                colmap[c] = canon_name
                break
    df = df.rename(columns=colmap)
    return df

def derive_progression(df: pd.DataFrame) -> pd.DataFrame:
    if "progress_24m" in df.columns:
        return df
    # Derive from baseline/24m diagnosis if present
    if "diagnosis_baseline" in df.columns and "diagnosis_24m" in df.columns:
        # Mark progression if goes from MCI -> AD or CN -> MCI/AD, etc.
        baseline = df["diagnosis_baseline"].astype(str).str.upper()
        follow = df["diagnosis_24m"].astype(str).str.upper()
        progressed = (
            ((baseline == "MCI") & (follow == "AD")) |
            ((baseline == "CN") & (follow.isin(["MCI","AD"])))
        )
        df["progress_24m"] = progressed.astype(int)
    return df

def missingness_report(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False).to_dict()
    return miss

def leakage_analysis(df: pd.DataFrame):
    risks = []
    if "visit_date" in df.columns:
        # check date leakage if labels formed after date
        risks.append("Time leakage risk if features post-date label window; ensure baseline-only features.")
    if "site_id" in df.columns:
        risks.append("Site leakage risk; enforce site-aware splits.")
    if "mmse" in df.columns and "diagnosis_24m" in df.columns:
        risks.append("Outcome leakage if follow-up-derived features included at baseline.")
    return risks

def propose_splits(df: pd.DataFrame, cfg: dict):
    seed = cfg.get("seed", 42)
    rng = np.random.RandomState(seed)
    if "site_id" not in df.columns or "visit_date" not in df.columns:
        # fallback random split
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n = len(idx)
        tr = int(0.6*n); va = int(0.8*n)
        return {"train_idx": idx[:tr].tolist(), "val_idx": idx[tr:va].tolist(), "test_idx": idx[va:].tolist(), "note":"fallback random split (site/time cols missing)"}
    # site-aware: split sites into folds, then time blocks within sites (simple heuristic)
    sites = df["site_id"].astype(str).unique()
    rng.shuffle(sites)
    n_sites = len(sites)
    n_train = max(1, int(0.6*n_sites))
    n_val = max(1, int(0.2*n_sites))
    train_sites = set(sites[:n_train])
    val_sites = set(sites[n_train:n_train+n_val])
    test_sites = set(sites[n_train+n_val:])
    tr_idx = df.index[df["site_id"].astype(str).isin(train_sites)].tolist()
    va_idx = df.index[df["site_id"].astype(str).isin(val_sites)].tolist()
    te_idx = df.index[df["site_id"].astype(str).isin(test_sites)].tolist()
    return {"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx, "note":"site-aware split (simple site partition)"}

def main(args):
    df = pd.read_csv(args.input)
    with open(args.schema, "r") as f:
        schema = yaml.safe_load(f)
    with open(args.splits, "r") as f:
        split_cfg = yaml.safe_load(f)

    df = normalize_columns(df, schema)
    # coerce date
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df = derive_progression(df)

    # sanity: check required
    missing_required = [c for c in schema["required"] if c not in df.columns]
    datacard = {
        "source": str(args.input),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "required_missing": missing_required,
        "n_rows": int(len(df)),
        "missingness": missingness_report(df),
        "leakage_risks": leakage_analysis(df),
    }
    splits = propose_splits(df, split_cfg)
    datacard["splits"] = splits

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "processed.parquet", index=False)
    with open(out_dir / "datacard.json", "w") as f:
        json.dump(datacard, f, indent=2, default=str)
    print("Wrote:", out_dir / "datacard.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV")
    ap.add_argument("--schema", default="config/schema.yaml")
    ap.add_argument("--splits", default="config/splits.yaml")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()
    main(args)

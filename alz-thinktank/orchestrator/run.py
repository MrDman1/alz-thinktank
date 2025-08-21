# orchestrator/run.py
import subprocess, sys, os

def sh(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # Curate
    sh([sys.executable, "agents/curator.py", "--input", "data/synthetic_mci.csv",
        "--schema", "config/schema.yaml", "--splits", "config/splits.yaml", "--outdir", "artifacts"])
    # Model
    sh([sys.executable, "agents/modeler.py", "--data", "artifacts/processed.parquet",
        "--datacard", "artifacts/datacard.json", "--outdir", "artifacts"])
    # Referee
    sh([sys.executable, "agents/referee.py", "--datacard", "artifacts/datacard.json",
        "--metrics", "artifacts/metrics.json", "--data", "artifacts/processed.parquet",
        "--outdir", "artifacts"])

if __name__ == "__main__":
    main()

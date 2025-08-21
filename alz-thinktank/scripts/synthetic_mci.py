# scripts/synthetic_mci.py
import numpy as np, pandas as pd
from datetime import datetime, timedelta
np.random.seed(7)

N = 200
age = np.random.normal(72, 6, N).clip(55, 90).round(1)
sex = np.random.choice(["M","F"], size=N)
apoe = np.random.choice(["e3/e3","e3/e4","e4/e4","e2/e3"], p=[0.5,0.35,0.1,0.05], size=N)
mmse = np.random.normal(27, 2.5, N).clip(20, 30).round(1)
adas = np.random.normal(12, 6, N).clip(0, 40).round(1)
start = datetime(2010,1,1)
visit_date = [start + timedelta(days=int(np.random.uniform(0, 365*3))) for _ in range(N)]
site_id = np.random.choice([f"S{i:02d}" for i in range(1, 8)], size=N, p=[.2,.15,.15,.15,.15,.1,.1])
mri_hippocampus = np.random.normal(3.5, 0.6, N).clip(2.2, 5.5).round(3)  # arbitrary units
csf_abeta = np.random.normal(650, 150, N).clip(300, 1200).round(1)
csf_tau = np.random.normal(300, 90, N).clip(100, 600).round(1)

# baseline dx = MCI
diagnosis_baseline = np.array(["MCI"]*N)
# progression probability influenced by age, apoe e4, low mmse, low abeta, high tau
logit = (
    0.02*(age-70) +
    0.9*np.isin(apoe, ["e3/e4","e4/e4"]).astype(float) +
    -0.15*(mmse-27) +
    -0.002*(csf_abeta-650) +
    0.003*(csf_tau-300)
)
p = 1/(1+np.exp(-logit))
progress_24m = (np.random.rand(N) < p).astype(int)
diagnosis_24m = np.where(progress_24m==1, "AD", "MCI")

df = pd.DataFrame({
    "age": age, "sex": sex, "apoe": apoe, "mmse": mmse, "adas": adas,
    "visit_date": [d.strftime("%Y-%m-%d") for d in visit_date],
    "site_id": site_id, "mri_hippocampus": mri_hippocampus,
    "csf_abeta": csf_abeta, "csf_tau": csf_tau,
    "diagnosis_baseline": diagnosis_baseline, "diagnosis_24m": diagnosis_24m,
    "progress_24m": progress_24m
})
df.to_csv("/mnt/data/alz-thinktank/data/synthetic_mci.csv", index=False)
print("Wrote /mnt/data/alz-thinktank/data/synthetic_mci.csv")

"""Weekly PSI/KL of every feature against the training window.

usage: python drift_report.py <dataset.parquet> <repo_root>
Writes reports/drift_weekly.csv (long table) and reports/drift_summary.csv (per week).
"""
import sys, numpy as np, pandas as pd
root = sys.argv[2]; sys.path.insert(0, root)
from drift_lab import DAY, DEFAULT_SPLIT
from drift_lab.drift import drift_table
df = pd.read_parquet(sys.argv[1])
feat = [c for c in df.columns if c not in ("TransactionID", "isFraud", "TransactionDT")]
week = ((df.TransactionDT.to_numpy() - DAY) // (7 * DAY)).astype(int) + 1
ref = (df.TransactionDT >= DEFAULT_SPLIT.train[0]).to_numpy() & (df.TransactionDT < DEFAULT_SPLIT.train[1]).to_numpy()
t = drift_table(df, feat, ref, week)
t.to_csv(f"{root}/reports/drift_weekly.csv", index=False, float_format="%.5f")
s = t.groupby("group").agg(features=("feature", "size"), psi_gt_0_1=("psi", lambda x: int((x > .1).sum())),
                           psi_gt_0_25=("psi", lambda x: int((x > .25).sum())), median_psi=("psi", "median"),
                           max_psi=("psi", "max"), median_kl=("kl", "median")).round(4)
s.index.name = "week"; s.to_csv(f"{root}/reports/drift_summary.csv")
print(s.to_string())
top = t[t.group >= 14].groupby("feature").psi.mean().sort_values(ascending=False).head(12).round(3)
print("\nmean PSI over weeks 14-26, top features:\n" + top.to_string())

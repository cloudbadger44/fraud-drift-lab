"""For each delay policy: labels visible at test start and at checkpoints; real-data invariants.

usage: python label_availability.py <dataset.parquet> <repo_root>
"""
import sys, numpy as np, pandas as pd
root = sys.argv[2]; sys.path.insert(0, root)
from drift_lab import DAY, DEFAULT_SPLIT, LabelStream, policy_from_name
df = pd.read_parquet(sys.argv[1], columns=["TransactionID", "TransactionDT", "isFraud"])
cps = DEFAULT_SPLIT.checkpoints(); rows = []
for name in ["none", "fixed7", "fixed14", "fixed30", "random30"]:
    s = LabelStream(df, policy_from_name(name), seed=42)
    for k in (0, 6, 12):
        t = cps[k]; X, y = s.view(t).labelled()
        arrived = int((df.TransactionDT < t).sum()); fr_arrived = int(df.isFraud[df.TransactionDT < t].sum())
        assert (s._release[s._released_idx(t)] < t).all()
        assert len(X) == int((s._release < t).sum())
        Xw, yw = s.view(t).labelled(window_days=28)
        rows.append(dict(policy=name, checkpoint=k + 1, day=t // DAY, arrived=arrived, labelled=len(X),
                         labelled_pct=round(100 * len(X) / arrived, 1), fraud_labelled=int(y.sum()),
                         fraud_arrived=fr_arrived, fraud_labelled_pct=round(100 * y.sum() / fr_arrived, 1),
                         window28_labelled=len(Xw)))
out = pd.DataFrame(rows); out.to_csv(f"{root}/reports/label_availability.csv", index=False)
print(out.to_string(index=False))

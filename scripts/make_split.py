"""Write configs/split.json and reports/split_summary.csv from the built dataset.

usage: python make_split.py <dataset.parquet> <repo_root>
"""
import sys, hashlib, json
import pandas as pd
sys.path.insert(0, sys.argv[2])
from drift_lab import DEFAULT_SPLIT, describe_split

df = pd.read_parquet(sys.argv[1], columns=["TransactionID", "TransactionDT", "isFraud"])
root = sys.argv[2]
DEFAULT_SPLIT.to_json(f"{root}/configs/split.json")
s = describe_split(df)
s.to_csv(f"{root}/reports/split_summary.csv")
fp = hashlib.sha256(pd.util.hash_pandas_object(df[["TransactionID", "TransactionDT"]], index=False).values.tobytes()).hexdigest()
json.dump({"rows": len(df), "sha256_of_id_time": fp}, open(f"{root}/configs/data_fingerprint.json", "w"), indent=2)
print(s.to_string()); print("checkpoints:", len(DEFAULT_SPLIT.checkpoints()))

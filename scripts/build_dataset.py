"""Merge IEEE-CIS train transaction + identity files, sort by TransactionDT, save parquet.

usage: python build_dataset.py <csv_dir> <out.parquet>
"""
import sys
import pandas as pd

csv_dir, out = sys.argv[1], sys.argv[2]
t = pd.read_csv(f"{csv_dir}/train_transaction.csv")
i = pd.read_csv(f"{csv_dir}/train_identity.csv")
assert t.TransactionID.is_unique and i.TransactionID.is_unique
df = t.merge(i, on="TransactionID", how="left", validate="one_to_one")
df["has_identity"] = df["TransactionID"].isin(i["TransactionID"]).astype("int8")
df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype("category")
    elif str(df[c].dtype) == "float64":
        df[c] = df[c].astype("float32")
df["TransactionDT"] = df["TransactionDT"].astype("int64")
df.to_parquet(out, index=False)
print(df.shape, "sorted:", df.TransactionDT.is_monotonic_increasing)

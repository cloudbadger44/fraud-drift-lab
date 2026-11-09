"""Skeleton of the evaluation loop every strategy plugs into (no model yet).

usage: python example_loop.py <dataset.parquet> <repo_root> <policy> [seed]
"""
import sys, pandas as pd
root = sys.argv[2]; sys.path.insert(0, root)
from drift_lab import DAY, DEFAULT_SPLIT, LabelStream, policy_from_name
df = pd.read_parquet(sys.argv[1], columns=["TransactionID", "TransactionDT", "isFraud"])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
store = LabelStream(df, policy_from_name(sys.argv[3]), seed=seed)
evaluator, cfg, prev = store.evaluator(), DEFAULT_SPLIT, None
for t in cfg.checkpoints():
    view = store.view(t)
    X_batch, y_batch = view.labelled()                  # batch: all released labels
    X_win, y_win = view.labelled(window_days=60)        # sliding window of W = 60 days
    X_new, y_new = view.new_labels(since=prev if prev is not None else 0)  # online feed
    X_next, y_next = evaluator.window(t, t + cfg.checkpoint_days * DAY)    # scored on truth
    print(f"day {t // DAY:3d}: batch={len(y_batch):6d} window={len(y_win):6d} new={len(y_new):6d} scored={len(y_next):6d}")
    prev = t

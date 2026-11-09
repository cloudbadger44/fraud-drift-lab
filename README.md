# drift_lab-fraud-drift: delayed-label data layer

Shared data-access layer for comparing batch, sliding-window and online fraud models
under label delay (IEEE-CIS). See the pipeline document for design and results.

    pip install pandas numpy pyarrow pytest
    python scripts/build_dataset.py <csv_dir> data/ieee_train.parquet
    python scripts/make_split.py data/ieee_train.parquet .
    python scripts/label_availability.py data/ieee_train.parquet .
    python -m pytest -q tests

Strategies get data only via `LabelStream(df, policy, seed).view(t)`; scoring uses `.evaluator()`.

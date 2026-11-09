import numpy as np, pandas as pd, pytest
from drift_lab import (DAY, FixedDelay, BoundedRandomDelay, NoDelay, LabelStream,
                 SplitConfig, DEFAULT_SPLIT, assign_window)


def make_df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(DAY, 183 * DAY, n))
    return pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": t,
                         "f1": rng.normal(size=n), "isFraud": (rng.random(n) < 0.035).astype(int)})


POLICIES = [FixedDelay(7), FixedDelay(14), FixedDelay(30), BoundedRandomDelay(30), NoDelay()]


@pytest.mark.parametrize("policy", POLICIES)
def test_no_label_before_release(policy):
    df = make_df(); s = LabelStream(df, policy, seed=1)
    for t in np.linspace(DAY, 200 * DAY, 25).astype(int):
        v = s.view(t); X, y = v.labelled()
        assert len(X) == len(y) == v.n_labelled()
        assert (s._release[s._released_idx(t)] < t).all()
        # everything not returned is genuinely unreleased
        assert (s._release < t).sum() == len(X)


def test_fixed_release_time_exact():
    df = make_df(); s = LabelStream(df, FixedDelay(14), seed=1)
    assert (s._release - s._time == 14 * DAY).all()


def test_random_bounded():
    df = make_df(); s = LabelStream(df, BoundedRandomDelay(30), seed=1)
    d = s._release - s._time
    assert d.min() >= 1 and d.max() <= 30 * DAY


def test_seed_reproducible_and_different():
    df = make_df()
    a = LabelStream(df, BoundedRandomDelay(30), seed=3)._release
    b = LabelStream(df, BoundedRandomDelay(30), seed=3)._release
    c = LabelStream(df, BoundedRandomDelay(30), seed=4)._release
    assert (a == b).all() and not (a == c).all()


def test_all_strategies_see_identical_labelled_data():
    """Batch, sliding and online feeds draw from one release rule: the union of
    new_labels over consecutive intervals equals labelled() at the end."""
    df = make_df(); s = LabelStream(df, BoundedRandomDelay(30), seed=2)
    ts = list(range(60 * DAY, 160 * DAY, 7 * DAY))
    seen, prev = [], 60 * DAY
    seen.append(s.view(prev).labelled()[0].index.to_numpy())
    for t in ts[1:]:
        Xn, _ = s.view(t).new_labels(since=prev)
        seen.append(Xn.index.to_numpy()); prev = t
    online = np.sort(np.concatenate(seen))
    batch = np.sort(s.view(ts[-1]).labelled()[0].index.to_numpy())
    assert (online == batch).all() and len(set(online)) == len(online)


def test_sliding_window_is_subset_of_batch():
    df = make_df(); s = LabelStream(df, FixedDelay(7), seed=1)
    t = 120 * DAY; v = s.view(t)
    Xb, _ = v.labelled(); Xw, _ = v.labelled(window_days=30)
    assert set(Xw.index) <= set(Xb.index)
    assert (s._time[Xw.index] >= t - 30 * DAY).all()


def test_label_column_never_in_features():
    df = make_df(); s = LabelStream(df, FixedDelay(7), seed=1)
    X, _ = s.view(150 * DAY).labelled()
    assert "isFraud" not in X.columns
    assert "isFraud" not in s.view(150 * DAY).arrivals(100 * DAY, 120 * DAY).columns


def test_arrivals_cannot_look_ahead():
    df = make_df(); s = LabelStream(df, FixedDelay(7), seed=1)
    with pytest.raises(ValueError):
        s.view(100 * DAY).arrivals(90 * DAY, 101 * DAY)


def test_unsorted_input_rejected():
    df = make_df().iloc[::-1]
    with pytest.raises(ValueError):
        LabelStream(df, FixedDelay(7), seed=1)


def test_split_windows_disjoint_contiguous_and_cover():
    df = make_df(20000); w = assign_window(df.TransactionDT)
    assert set(w) == {"train", "val", "test"} and len(w) == len(df)
    mx = lambda k: df.TransactionDT[w == k].max(); mn = lambda k: df.TransactionDT[w == k].min()
    assert mx("train") < mn("val") <= mx("val") < mn("test")


def test_checkpoints_stay_inside_test_window_and_scoring_window_unseen_by_training():
    cfg = DEFAULT_SPLIT; cps = cfg.checkpoints()
    assert cps[0] == cfg.test[0] and cps[-1] + 7 * DAY <= cfg.test[1] and len(cps) == 13
    df = make_df(); s = LabelStream(df, NoDelay(), seed=1)
    for t in cps:
        Xtr, _ = s.view(t).labelled()
        Xte, _ = s.evaluator().window(t, t + 7 * DAY)
        assert not set(Xtr.index) & set(Xte.index)   # train-before-test even with zero delay


def test_transaction_arriving_exactly_at_checkpoint_is_not_in_training():
    t = 120 * DAY
    extra = pd.DataFrame({"TransactionID": [-1], "TransactionDT": [t], "f1": [0.0], "isFraud": [1]})
    df = pd.concat([make_df(), extra]).sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    s = LabelStream(df, NoDelay(), seed=1)
    assert -1 not in set(s.view(t).labelled()[0].TransactionID)
    assert -1 in set(s.evaluator().window(t, t + DAY)[0].TransactionID)

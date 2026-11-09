import numpy as np, pandas as pd
from drift_lab.drift import Binner, psi_kl, drift_table


def test_identical_distribution_has_near_zero_psi():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=50000)); b = pd.Series(rng.normal(size=50000))
    bn = Binner(a); psi, kl = psi_kl(bn.freq(a), bn.freq(b))
    assert psi < 0.01 and kl < 0.01


def test_shifted_distribution_is_large_psi():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=50000)); b = pd.Series(rng.normal(1.5, 1, size=50000))
    bn = Binner(a); psi, _ = psi_kl(bn.freq(a), bn.freq(b))
    assert psi > 0.25


def test_missingness_shift_is_detected():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=20000)); b = a.copy(); b[rng.random(20000) < 0.5] = np.nan
    bn = Binner(a); psi, _ = psi_kl(bn.freq(a), bn.freq(b))
    assert psi > 0.25


def test_unseen_category_goes_to_other_bin():
    a = pd.Series(["x"] * 900 + ["y"] * 100); b = pd.Series(["z"] * 1000)
    bn = Binner(a); psi, _ = psi_kl(bn.freq(a), bn.freq(b))
    assert psi > 1.0


def test_drift_table_shape_and_zero_for_reference_group():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"f": rng.normal(size=4000), "c": rng.choice(["a", "b"], 4000)})
    grp = np.repeat([1, 2], 2000); ref = grp == 1
    t = drift_table(df, ["f", "c"], ref, grp)
    assert len(t) == 4 and t[(t.group == 1)].psi.max() < 1e-9

"""
Gaussian HMM regime overlay: fit on train returns only; use at evaluation to scale SAC weights.

The HMM is not used during SAC training. Pass daily_risk_scale=g_t into trading_env;
each step uses portfolio_return = g_t * (w^T r) with SAC weights w on the simplex (cash earns 0).

Posteriors from score_samples use forward-backward on the segment passed (smoothing within
that segment only). For test evaluation, only test returns are passed; HMM params were fit on train.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

TICKERS = ["SPY", "QQQ", "TLT"]
N_COMPONENTS = 3
# Assigned to states sorted by average train return volatility (low -> high): calm -> stress
MULTIPLIER_BY_VOL_RANK = np.array([1.0, 0.6, 0.2])


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def daily_returns_from_split(split_df: pd.DataFrame) -> pd.DataFrame:
    df = split_df.reset_index()
    px = df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    r = px.pct_change().dropna()
    return r[TICKERS].dropna()


def _canonicalize_hmm_by_vol(hmm: GaussianHMM, order: np.ndarray) -> None:
    """
    Permute HMM states so new id 0 = calmest, 2 = most volatile (order = argsort vol ascending).
    order[k] = old state index placed at new index k.
    """
    o = np.asarray(order, dtype=int)
    hmm.means_ = hmm.means_[o]
    if hmm.covariance_type == "diag":
        hmm.covars_ = hmm.covars_[o]
    elif hmm.covariance_type == "full":
        hmm.covars_ = hmm.covars_[o]
    else:
        raise NotImplementedError(f"canonicalize for {hmm.covariance_type}")
    hmm.startprob_ = hmm.startprob_[o]
    hmm.startprob_ = np.clip(hmm.startprob_, 1e-16, 1.0)
    hmm.startprob_ /= hmm.startprob_.sum()
    hmm.transmat_ = hmm.transmat_[np.ix_(o, o)]
    hmm.transmat_ = np.clip(hmm.transmat_, 1e-16, 1.0)
    hmm.transmat_ /= hmm.transmat_.sum(axis=1, keepdims=True)


def fit_risk_model_from_pickle(
    pkl_path: Path | None = None,
    out_path: Path | None = None,
    random_state: int = 42,
) -> dict:
    pkl_path = pkl_path or _repo_root() / "data_files" / "engineered.pkl"
    out_path = out_path or Path(__file__).resolve().parent / "risk_model.pkl"

    data = pd.read_pickle(pkl_path)
    r_train = daily_returns_from_split(data["train"])
    X = r_train.values.astype(np.float64)
    if len(X) < N_COMPONENTS * 10:
        raise ValueError(f"Need more train days for HMM; got {len(X)}")

    # min_covar avoids near-singular covariances; more n_iter / looser tol reduces "not converging" noise
    hmm = GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        n_iter=2000,
        tol=5e-3,
        random_state=random_state,
        min_covar=1e-4,
        verbose=False,
    )
    # Suppress hmmlearn's "Model is not converging" print (EM still returns a usable fit).
    with contextlib.redirect_stdout(io.StringIO()):
        hmm.fit(X)

    states = hmm.predict(X)
    vol_by_state = np.zeros(N_COMPONENTS)
    for k in range(N_COMPONENTS):
        mask = states == k
        if not np.any(mask):
            vol_by_state[k] = np.inf
        else:
            vol_by_state[k] = float(np.mean(np.linalg.norm(X[mask], axis=1)))

    # Permute states so id 0 = lowest train vol … id 2 = highest (fixes label switching).
    order = np.argsort(vol_by_state)
    _canonicalize_hmm_by_vol(hmm, order)
    state_multipliers = MULTIPLIER_BY_VOL_RANK.astype(np.float64).copy()

    bundle = {
        "hmm": hmm,
        "state_multipliers": state_multipliers,
        "tickers": TICKERS,
        "n_components": N_COMPONENTS,
        "train_end": r_train.index.max(),
    }
    joblib.dump(bundle, out_path)
    print(f"Saved {out_path}")
    print("state_multipliers (state 0=calm … 2=stress):", state_multipliers)
    return bundle


def load_risk_model(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent / "risk_model.pkl"
    return joblib.load(path)


def expected_risk_scale_series(bundle: dict, returns_df: pd.DataFrame) -> pd.Series:
    """Expected risk scale g_t = sum_k P(state=k|seq) * m_k for each row of returns_df."""
    tickers = bundle["tickers"]
    X = returns_df[tickers].values.astype(np.float64)
    _, post = bundle["hmm"].score_samples(X)
    m = bundle["state_multipliers"]
    g = post @ m
    return pd.Series(g, index=returns_df.index, name="risk_scale")


if __name__ == "__main__":
    fit_risk_model_from_pickle()

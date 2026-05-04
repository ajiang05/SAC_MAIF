import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from env import trading_env
from regime_hmm import expected_risk_scale_series, load_risk_model
from stable_baselines3 import SAC

_RL_DIR = Path(__file__).resolve().parent
_ROOT = _RL_DIR.parent

# Load engineered data (run from repo root or rl/ — paths are absolute via _ROOT)
data = pd.read_pickle(_ROOT / "data_files" / "engineered.pkl")
test_data = data["val"]

feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "Past_close", "RSI", "BB_Mid", "BB_Upper", "BB_Lower",
    "MACD", "MACD_signal",
]

df_reset = test_data.reset_index()
print(df_reset.groupby(["Date", "Ticker"]).size().value_counts())

pivot_features = df_reset.pivot_table(
    index="Date",
    columns="Ticker",
    values=feature_cols,
    aggfunc="first",
)
pivot_features = pivot_features.ffill().bfill()
pivot_features.columns = [f"{col[1]}_{col[0]}" for col in pivot_features.columns]

price_df = df_reset.pivot(index="Date", columns="Ticker", values="Close")
returns = price_df.pct_change().dropna()
#returns table that shows the percentage change in the closing prices of the tickers for each date
returns = price_df.pct_change().dropna() 
returns = returns.shift(-1) # Shift returns backwards so day t features predict day t+1 returns

common_index = pivot_features.index.intersection(price_df.index)
pivot_features = pivot_features.loc[common_index]
price_df = price_df.loc[common_index]

features = pivot_features.iloc[1:]
ret_1 = price_df.pct_change().shift(1).iloc[1:]
ret_5 = price_df.pct_change(5).shift(1).iloc[1:]

features = pd.concat([features, ret_1, ret_5], axis=1)
features = features.dropna()
returns = returns.loc[features.index]

with open(_RL_DIR / "scaler.pkl", "rb") as f:
    mean, std = pickle.load(f)

features = (features - mean) / std
returns = returns.loc[features.index]
returns = returns.dropna()

# keep only assets
returns = returns[["SPY", "QQQ", "TLT"]].copy()
returns["Cash"] = 0.0
#realign features AFTER dropping NaNs
features = features.loc[returns.index]

print("Features shape:", features.shape, "Returns shape:", returns.shape)

model_path = _RL_DIR / "model" / "sac_model_20260427_Aidan.zip"
model = SAC.load(model_path)


def sharpe(rets):
    rets = np.array(rets)
    return (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)


def run_sac_backtest(env, sac_model, deterministic=True):
    obs, _ = env.reset()
    for _ in range(len(env.returns)):
        action, _ = sac_model.predict(obs, deterministic=deterministic)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    return env.portfolioReturns


# --- SAC only (baseline)
env_sac = trading_env(features, returns)
port_sac = run_sac_backtest(env_sac, model)

# --- SAC + HMM regime scaling
risk_path = _RL_DIR / "risk_model.pkl"
if not risk_path.is_file():
    print(
        "\nNo risk_model.pkl — run:  python regime_hmm.py\n"
        "from the rl folder (or: python -m rl.regime_hmm from repo root if packaged).\n"
        "Skipping HMM overlay.\n"
    )
    port_hmm = None
else:
    bundle = load_risk_model(risk_path)
    risk_scale = expected_risk_scale_series(bundle, returns)
    assert risk_scale.index.equals(returns.index), "Risk scale index must match returns"
    env_hmm = trading_env(features, returns, daily_risk_scale=risk_scale.values)
    port_hmm = run_sac_backtest(env_hmm, model)

# --- Random baseline
env_random = trading_env(features, returns)
obs_r, _ = env_random.reset()
for _ in range(len(env_random.returns)):
    action = env_random.action_space.sample()
    obs_r, _, done, _, _ = env_random.step(action)
    if done:
        break

model_sharpe = sharpe(port_sac)
random_sharpe = sharpe(env_random.portfolioReturns)
equal_returns = returns.mean(axis=1)
equal_sharpe = (equal_returns.mean() / (equal_returns.std() + 1e-8)) * np.sqrt(252)

# --- Equal-weight full metrics
cum_eq = equal_returns.cumsum()
max_dd_eq = (cum_eq - cum_eq.cummax()).min()
ann_return_eq = equal_returns.mean() * 252
calmar_eq = ann_return_eq / (abs(max_dd_eq) + 1e-8)

print(f"\nEqual-weight - Max Drawdown: {max_dd_eq:.2%}")
print(f"Equal-weight - Annualized Return: {ann_return_eq:.2%}")
print(f"Equal-weight - Calmar: {calmar_eq:.2f}")

# --- Random full metrics
random_returns = np.array(env_random.portfolioReturns)

cum_rand = pd.Series(random_returns).cumsum()
max_dd_rand = (cum_rand - cum_rand.cummax()).min()
ann_return_rand = random_returns.mean() * 252
calmar_rand = ann_return_rand / (abs(max_dd_rand) + 1e-8)

print(f"\nRandom - Max Drawdown: {max_dd_rand:.2%}")
print(f"Random - Annualized Return: {ann_return_rand:.2%}")
print(f"Random - Calmar: {calmar_rand:.2f}")

print(f"\nSAC Sharpe: {model_sharpe:.4f}")
if port_hmm is not None:
    print(f"SAC + HMM Sharpe: {sharpe(port_hmm):.4f}")
print(f"Random Sharpe: {random_sharpe:.4f}")
print(f"Equal-weight Sharpe: {equal_sharpe:.4f}")

cum_for_metrics = pd.Series(port_sac).cumsum()
max_dd = (cum_for_metrics - cum_for_metrics.cummax()).min()
annualized_return = float(np.mean(port_sac) * 252)
calmar = annualized_return / (abs(max_dd) + 1e-8)
print(f"\nSAC - Max Drawdown (cum sum space): {max_dd:.2%}")
print(f"SAC - Annualized Return (mean*252): {annualized_return:.2%}")
print(f"SAC - Calmar: {calmar:.2f}")

if port_hmm is not None:
    cum_h = pd.Series(port_hmm).cumsum()
    max_dd_h = (cum_h - cum_h.cummax()).min()
    ann_h = float(np.mean(port_hmm) * 252)
    print(f"\nSAC+HMM - Max Drawdown: {max_dd_h:.2%}")
    print(f"SAC+HMM - Annualized Return: {ann_h:.2%}")
    print(f"SAC+HMM - Calmar: {ann_h / (abs(max_dd_h) + 1e-8):.2f}")

idx = returns.index[: len(port_sac)]
model_series = pd.Series(port_sac, index=idx)
cum_model = model_series.cumsum()
cum_equal = returns.mean(axis=1).iloc[: len(model_series)].cumsum()
cum_random = pd.Series(env_random.portfolioReturns, index=returns.index[: len(env_random.portfolioReturns)]).cumsum()


# --- Combine ALL splits (fixed key)
full_data = pd.concat([data["train"], data["val"], data["test"]])

# --- Create price dataframe
df_reset = full_data.reset_index()
price_df = df_reset.pivot(index="Date", columns="Ticker", values="Close")

plt.figure()

plt.plot(price_df.index, price_df["SPY"], color="black", linewidth=1.5)

plt.axvspan(pd.to_datetime('2020-02-01'),
            pd.to_datetime('2020-06-01'),
            alpha=0.15)


plt.text(pd.to_datetime('2020-03-01'),
         price_df["SPY"].min() + 140,
         "High Volatility",
         fontsize=10)

plt.title("Market Regimes and Volatility (SPY)")
plt.xlabel("Date")
plt.ylabel("Price")

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cum_model, label="SAC")
if port_hmm is not None:
    plt.plot(pd.Series(port_hmm, index=idx).cumsum(), label="SAC + HMM")
plt.plot(cum_equal, label="Equal-weight")
plt.plot(cum_random, label="Random")
plt.legend()
plt.title("Test set — cumulative simple returns")
plt.xlabel("Date")
plt.ylabel("Cumulative return")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cum_model - cum_model.cummax(), label="SAC")
if port_hmm is not None:
    ch = pd.Series(port_hmm, index=idx).cumsum()
    plt.plot(ch - ch.cummax(), label="SAC + HMM")
plt.legend()
plt.title("Drawdown (cumulative return space)")
plt.tight_layout()
plt.show()
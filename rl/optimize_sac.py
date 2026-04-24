import pandas as pd
import numpy as np
from stable_baselines3 import SAC
import pickle
import optuna
import warnings
from env import trading_env

# Suppress warnings to keep the console clean during the 20 trials
warnings.filterwarnings("ignore")

# --- 1. PREPARE THE DATA (Same as train_sac.py) ---
data = pd.read_pickle("data_files/engineered.pkl")
train_data = data["train"]

feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "Past_close", "RSI", "BB_Mid", "BB_Upper", "BB_Lower",
    "MACD", "MACD_signal"
]

df_reset = train_data.reset_index() 
pivot_features = df_reset.pivot_table(index="Date", columns="Ticker", values=feature_cols, aggfunc="first")
pivot_features = pivot_features.ffill().bfill() 
pivot_features.columns = [f"{col[1]}_{col[0]}" for col in pivot_features.columns]

price_df = df_reset.pivot(index="Date", columns="Ticker", values="Close")
returns = price_df.pct_change().dropna() 

features = pivot_features.iloc[1:]
ret_1 = price_df.pct_change().iloc[1:]
ret_5 = price_df.pct_change(5).iloc[1:]
ret_1 = ret_1.loc[features.index]
ret_5 = ret_5.loc[features.index]

features = pd.concat([features, ret_1, ret_5], axis=1).dropna()
returns = returns.loc[features.index]

# Normalize
mean = features.iloc[:252].mean()
std = features.iloc[:252].std() + 1e-8
features = (features - mean) / std

returns = returns[["SPY", "QQQ", "TLT"]] 

# Create the training environment
env = trading_env(features, returns)

# --- 2. OPTUNA OBJECTIVE FUNCTION ---
def objective(trial):
    # 1. Optuna suggests hyperparameters to try
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1])

    # 2. Create the model with the guessed hyperparameters
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate, 
        batch_size=batch_size, 
        buffer_size=buffer_size,
        ent_coef=ent_coef,
        verbose=0 # Turn off printing so it doesn't spam the console
    )

    # 3. Do a mini training run (20,000 steps is enough to see if it is learning)
    model.learn(total_timesteps=20000)

    # 4. Evaluate the model on the environment
    obs, _ = env.reset()
    for i in range(len(env.returns)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if done:
            break

    # 5. Calculate the Sharpe Ratio
    port_returns = np.array(env.portfolioReturns)
    sharpe = port_returns.mean() / (port_returns.std() + 1e-8)
    
    # 6. Return the score to Optuna!
    return sharpe

# --- 3. RUN THE STUDY ---
if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization...")
    print("This will test 20 different combinations. Grab a coffee, this might take a few minutes!\n")
    
    # direction="maximize" because we want the highest Sharpe ratio possible
    study = optuna.create_study(direction="maximize")
    
    # n_trials=20 means it will test 20 different combinations
    study.optimize(objective, n_trials=20)

    print("\n==================================")
    print("OPTIMIZATION FINISHED!")
    print(f"Best Sharpe Ratio: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==================================")

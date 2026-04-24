import pandas as pd
from env import trading_env
import numpy as np
from stable_baselines3 import SAC
import pickle #loads saved data (scaler)
import matplotlib.pyplot as plt

#Load the training data (same as test_env.py)
data = pd.read_pickle("data_files/engineered.pkl")
test_data = data["test"]

feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "Past_close", "RSI", "BB_Mid", "BB_Upper", "BB_Lower",
    "MACD", "MACD_signal"
]

#turns the row labels to just 0 to whatever number of rows there are
df_reset = test_data.reset_index() 

#check to see if there are any duplicates
print(df_reset.groupby(["Date", "Ticker"]).size().value_counts())
 
#Makes all the tickers and indicators in one row with only dates as the row label, and the indicators as the columns
pivot_features = df_reset.pivot_table(
    index="Date",
    columns="Ticker",
    values=feature_cols,
    aggfunc="first"
)

pivot_features = pivot_features.ffill().bfill() 

#Makes the columns names more readable for example (Close, SPY) To: SPY_Close
pivot_features.columns = [
    f"{col[1]}_{col[0]}" for col in pivot_features.columns 
]

#price table that shows the closing prices of the tickers for each date
price_df = df_reset.pivot(
index="Date",
columns="Ticker",
values="Close" 
)

#returns table that shows the percentage change in the closing prices of the tickers for each date
returns = price_df.pct_change().dropna() 

#aligns the index of the pivot features and the price df by taking dates that are in both
common_index = pivot_features.index.intersection(price_df.index)

#only keeping dates that are in both the pivot features and the price df
pivot_features = pivot_features.loc[common_index] 
price_df = price_df.loc[common_index]


features = pivot_features.iloc[1:]
ret_1 = price_df.pct_change().iloc[1:] #returns for the next day
ret_5 = price_df.pct_change(5).iloc[1:] #returns for the next 5 days

print("Pivot features sample:")
print(pivot_features.head())

print("\nPrice DF sample:")
print(price_df.head())

print("\nCheck if price_df columns differ:")
print(price_df.iloc[0])

features = pd.concat([features, ret_1, ret_5], axis=1) #concatenates the features and the returns to make one big dataframe

# remove NaNs
features = features.dropna()

# aligns the returns to the features
returns = returns.loc[features.index]

#loads the scaler that was used to normalize the data
with open("scaler.pkl", "rb") as f:
    mean, std = pickle.load(f)


print("RAW FEATURES CHECK:")
print(features[["SPY_Close", "QQQ_Close", "TLT_Close"]].head())

features = (features - mean) / std #normalizes the features using the mean and std of the training data

# align returns
returns = returns.loc[features.index]
returns = returns[["SPY", "QQQ", "TLT"]] #only keeps the returns for the tickers


#Create the environment for the model
env = trading_env(features, returns)

#Load the model
model = SAC.load("rl/model/sac_training_model") #load the model
obs, _ = env.reset() #reset the environment
model_rewards = []
#Loop through the returns and step the environment forward by one time step
for i in range(len(env.returns)): #loop through the returns
    action, _ = model.predict(obs) #predict the action
    obs, reward, done, _, _ = env.step(action) #step the environment forward by one time step
    model_rewards.append(reward)
    if done: #if the episode is done
        break

#Create the environment for the random agent (same as env)
env_random = trading_env(features, returns)
obs_random, _ = env_random.reset() 
random_rewards = []
for i in range(len(env_random.returns)): 
    action = env_random.action_space.sample() #sample a random action from the action space
    obs_random, reward, done, _, _ = env_random.step(action)
    random_rewards.append(reward)
    if done: 
        break

#sharpe ratio of the model
def sharpe(returns):
    returns = np.array(returns)
    return returns.mean() / (returns.std() + 1e-8) #add a small epsilon to avoid division by zero

#calculate the sharpe ratio of the model and the random agent
model_sharpe = sharpe(env.portfolioReturns)
random_sharpe = sharpe(env_random.portfolioReturns)

print(f"Model Sharpe: {model_sharpe}")
print(f"Random Sharpe: {random_sharpe}")

#equal weight baseline (equal weight of the returns for each ticker)
equal_returns = returns.mean(axis=1) 
equal_sharpe = equal_returns.mean() / (equal_returns.std() + 1e-8)

print(f"Equal-weight Sharpe: {equal_sharpe}")

if model_sharpe > random_sharpe:
    print("The model is better than the random agent")
else:
    print("The random agent is better than the model")


print(action)

print("Mean return:", np.mean(env.portfolioReturns))
print("Volatility:", np.std(env.portfolioReturns))


# Align model returns with dates
model_series = pd.Series(
    env.portfolioReturns,
    index=returns.index[:len(env.portfolioReturns)]
)


#Graphs:

cum_model = model_series.cumsum() #total growth over time
cum_equal = returns.mean(axis=1).iloc[:len(model_series)].cumsum() #total growth over time for the equal weight baseline

#plots the cumulative returns for the model
plt.figure()
plt.plot(cum_model)
plt.title("Model Cumulative Returns (Test Set)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()

#plots the cumulative returns for the model and the equal weight baseline
plt.figure()
plt.plot(cum_model, label="Model")
plt.plot(cum_equal, label="Equal-weight")
plt.legend()
plt.title("Model vs Equal-weight")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()

cum = cum_model
drawdown = cum - cum.cummax()

#plots the drawdown for the model
plt.plot(drawdown)
plt.title("Drawdown")
plt.show()
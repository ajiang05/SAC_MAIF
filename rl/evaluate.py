import pandas as pd
from env import trading_env
import numpy as np
from stable_baselines3 import SAC

#Load the training data (same as test_env.py)
data = pd.read_pickle("data_files/engineered.pkl")
val_data = data["val"]

feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "Past_close", "RSI", "BB_Mid", "BB_Upper", "BB_Lower",
    "MACD", "MACD_signal"
]

#turns the row labels to just 0 to whatever number
df_reset = val_data.reset_index() 

#Makes all the tickers and indicators in one row with only dates as the row label
pivot_features = df_reset.pivot_table( 
index="Date", 
columns="Ticker",
values=feature_cols 
)

pivot_features = pivot_features.ffill().bfill() 

pivot_features.columns = [
    f"{col[0]}_{col[1]}" for col in pivot_features.columns 
]

price_df = df_reset.pivot(
index="Date",
columns="Ticker",
values="Close" 
)

returns = price_df.pct_change().dropna() 
features = pivot_features.iloc[1:] 
returns = returns[["SPY", "QQQ", "TLT"]] 


#Create the environment for the model
env = trading_env(features, returns)

#Load the model
model = SAC.load("sac_training_model") #load the model
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
model_sharpe = sharpe(model_rewards)
random_sharpe = sharpe(random_rewards)

print(f"Model Sharpe: {model_sharpe}")
print(f"Random Sharpe: {random_sharpe}")

if model_sharpe > random_sharpe:
    print("The model is better than the random agent")
else:
    print("The random agent is better than the model")




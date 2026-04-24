import pandas as pd
from env import trading_env
import numpy as np
from stable_baselines3 import SAC

#Load the training data (same as test_env.py)
data = pd.read_pickle("data_files/engineered.pkl")
train_data = data["train"]

feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "Past_close", "RSI", "BB_Mid", "BB_Upper", "BB_Lower",
    "MACD", "MACD_signal"
]

#turns row labels to numbers from 0 to len(row) and pushes the dates to the right, so column size increases by 1
df_reset = train_data.reset_index() 

#creates multi level index with the features first then the tickers
pivot_features = df_reset.pivot_table( 
index="Date", 
columns="Ticker",
values=feature_cols 
)

print(pivot_features)

pivot_features = pivot_features.ffill().bfill() 

pivot_features.columns = [
    f"{col[0]}_{col[1]}" for col in pivot_features.columns 
]

price_df = df_reset.pivot(
index="Date",
columns="Ticker",
values="Close" 
)

#creates a df for the percent change between current row and previous row. Basically yesterday to today percent change
returns = price_df.pct_change().dropna() 
#We get rid of row 0 to match with returns which got rid of first day(there is no yesterday yet)
features = pivot_features.iloc[1:] 
returns = returns[["SPY", "QQQ", "TLT"]] 

print("Features shape:", features.shape) 
print("Returns shape:", returns.shape)


#Create the environment
env = trading_env(features, returns)

model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=100000, batch_size=64) #create the model

model.learn(total_timesteps=100000) #train the model
model.save("sac_training_model") #save the model

print("Model trained and saved")
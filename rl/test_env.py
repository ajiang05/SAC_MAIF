import pandas as pd
from env import trading_env

#Load the training data
data = pd.read_pickle("data_files/engineered.pkl")
train_data = data["train"]

#Print the type and dtypes of the training data
print("Train data type:", type(train_data))
print(train_data.dtypes)


#Convert the training data from long format to wide format (time, asset-feature)
#Basically the data is that each row is a single observation of a single asset at a single time but we want to pivot it to a wide format so that each row is a single observation of all assets at a single time
    
# Define the columns to pivot (price info and indicators)
feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "Past_close", "RSI", "BB_Mid", "BB_Upper", "BB_Lower",
    "MACD", "MACD_signal"
]

#Pivot the features to a wide format (time, asset-feature) 
pivot_features = train_data.reset_index().pivot_table( 
index="Date", #the rows are the dates
columns="Ticker", #the columns are the tickers
values=feature_cols #the values are the features
)

#fill the missing values 
pivot_features = pivot_features.ffill().bfill() 

# Flatten columns (change them from (asset, feature) to (asset_feature), for example from SPY,Close to SPY_Close)
pivot_features.columns = [
    f"{col[0]}_{col[1]}" for col in pivot_features.columns 
]


#Create the returns dataframe (time, asset) - this is the price data for each asset at each time step
price_df = train_data.reset_index().pivot(
index="Date",
columns="Ticker",
values="Close" #the values are the close prices
)

returns = price_df.pct_change().dropna() #calculate the returns for each asset at each time step. Formula: (price_t - price_t-1) / price_t-1 (percentage change)
# Align features (remove the first row of the pivot_features dataframe since it was NaN)
features = pivot_features.iloc[1:]

returns = returns[["SPY", "QQQ", "TLT"]] #only keep the SPY, QQQ, and TLT returns



#Print the shape and columns of the features and returns dataframes
print("Features shape:", features.shape) 
print("Returns shape:", returns.shape)
print("Returns columns:", returns.columns)


#Create the environment
env = trading_env(features, returns)

#Reset the environment
obs, _ = env.reset()
#Print the shape of the observation
print("\nObservation shape:", obs.shape)


for i in range(5):
    action = env.action_space.sample() #sample a random action from the action space
    obs, reward, done, _, _ = env.step(action) #step the environment forward by one time step

    print(f"\nStep {i+1}")
    print("Reward:", reward)
    print("Done:", done)

    if done:
        break
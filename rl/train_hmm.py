import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import pickle
"""
Why are we using regime switch?
We are using regime switching as a risk manager. Because markets can be bullish or bearish
we have to acommodate for that. Regime switching allows us to allocate more funds during.
Think of it like wearing different clothes(weights) based on the weather(markets).

Why are we using hidden markov models?
We are using HMM because because they have zero lag. Using a 50 day moving average would
have alot of lag. The HMM looks at todays returns and see todays "stress". 

What is a markov model?
Markov model is used to describe transition between different states. 

Why is it hidden?
There is no actual announcement about the current state we are in.(Bull or Bear). We have
to use the price of SPY to get the symptoms of the regime.

Hmmlearn is a Python library for building hidden markov models. 
"""
#1. Load Data
# We only want to train the HMM on the overall market (SPY) returns
print("Loading training data...")
data = pd.read_pickle("data_files/engineered.pkl")
train_data = data["train"]
df_reset = train_data.reset_index() 

# Pivot to get just the Close prices
price_df = df_reset.pivot(index="Date", columns="Ticker", values="Close")

# Calculate daily returns for SPY
spy_returns = price_df["SPY"].pct_change().dropna().values.reshape(-1, 1)


#2. Build and Train the HMM
print("Training Hidden Markov Model (HMM)...")
# We specify 3 components (Calm, Moderate, Stress)
# covariance_type="diag" means we assume variance is independent
# n_iter=100 is the number of loops it will run to find the best fit
hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)

# Fit the model to the SPY returns
hmm_model.fit(spy_returns)


#3. Identify Which State is Which
# The HMM randomly assigns states as 0, 1, and 2. 
# We need to sort them by volatility (variance) so we know which one is "Stress"
# hmm_model.covars_ holds the variance of each state
variances = hmm_model.covars_.flatten()

# This sorts the indices of the states from lowest variance to highest variance
sorted_state_indices = np.argsort(variances)

# We create a dictionary mapping the HMM's internal state ID to our human-readable names
state_map = {
    sorted_state_indices[0]: "Calm",      # Lowest volatility
    sorted_state_indices[1]: "Moderate",  # Medium volatility
    sorted_state_indices[2]: "Stress"     # Highest volatility
}

print("\n--- HMM State Identification ---")
for internal_id, name in state_map.items():
    vol = np.sqrt(variances[internal_id]) * np.sqrt(252) # Annualized Volatility
    print(f"State {internal_id} -> {name} (Annualized Vol: {vol:.2%})")


#4. Save the Model and the Map
print("\nSaving HMM model to rl/model/hmm_model.pkl...")
with open("rl/model/hmm_model.pkl", "wb") as f:
    # We save both the model AND the map so evaluate.py knows what state means what
    pickle.dump((hmm_model, state_map), f)
    
print("Done! You can now use the HMM in evaluate.py to scale your risk.")

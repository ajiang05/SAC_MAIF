# Reinforcement Learning Training Basics

Knowing when to wipe the slate clean vs when to keep building on an existing "brain" is one of the most important concepts in Reinforcement Learning model development.

Here is your definitive guide for when to restart vs. resume training for your SAC Agent:

## 🚨 ALWAYS Restart From Scratch When:

If you change the "rules of the game" or the data structures, the old model will become completely confused (or just instantly crash due to shape mismatches). You **must** train a brand new model if you change any of the following:

### 1. You Add or Remove Features (Observation Space)
* **Example:** You add a new indicator like the VIX, or you add `ret_1` and `ret_5` to your dataframe. 
* **Why:** The neural network's input layer is physically sized to match the number of features. If it was built to accept 39 inputs and you suddenly hand it 45, it crashes with an `Unexpected observation shape` error.

### 2. You Change What the Agent Can Do (Action Space)
* **Example:** You change the action space from `[0, 1]` (Long Only) to `[-1, 1]` (Shorting Allowed), or you add a 4th stock like AAPL to your portfolio.
* **Why:** The agent's output layer expects a specific format. If its past experience says "0.0 is the lowest I can go," and suddenly the environment maps that to a different real-world action, the strategy falls apart.

### 3. You Change How the Agent Makes Money (Reward Function)
* **Example:** You add a volatility penalty, or you change the turnover fee from `0.0003` to `0.0010`.
* **Why:** The agent spent hours learning a delicate mathematical balance to maximize its old reward. If you change the math, the agent will keep executing its old strategy, but it will suddenly start losing points and won't understand why.

---

## 🟢 RESUME Training (Pick up where you left off) When:

You should resume training (using `model = SAC.load(...)` before calling `.learn()`) when the structure of the game is exactly the same, but you just want the agent to get more experience.

### 1. You Need More Timesteps
* **Example:** You trained for 100,000 steps, ran the evaluation, and got a decent Sharpe Ratio. You think it could do better, so you resume training for another 400,000 steps to see if it finds a sharper strategy.

### 2. You Have New (But Identically Formatted) Data
* **Example:** It's the year 2025. You download the new 2024 stock data. You run it through your `engineered.pkl` pipeline so the columns are identical, and you feed it to your 2023 model so it can learn the recent trends. 

### 3. The Training Was Interrupted
* **Example:** You started a massive 1,000,000 timestep training run, but your computer crashed halfway through. As long as the code hasn't changed, you can load the partially trained `.zip` file and pick up right where it left off.

---

## Summary Checklist

Before you run `.learn()` on an existing model, ask yourself:
* *"Did I edit `env.py`?"* ➡️ Restart.
* *"Did I edit the feature columns in the data pipeline?"* ➡️ Restart.
* *"Did I just want to let it train longer because it needs more practice?"* ➡️ Resume!

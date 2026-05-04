# SAC-MAIF: Algorithmic Trading with Soft Actor-Critic and HMM Regime Overlays

This repository contains the implementation of a dual-brain algorithmic trading pipeline. It combines a Reinforcement Learning agent (Soft Actor-Critic) for continuous portfolio optimization with a Hidden Markov Model (HMM) for regime-switching risk management.

## Project Architecture

The system is separated into two independent models to prevent the AI from overfitting to a specific market condition:
1. **The Alpha Generator (SAC Agent):** Focuses entirely on maximizing risk-adjusted returns by dynamically allocating portfolio weights across aggressive, baseline, and safe-haven assets.
2. **The Risk Manager (HMM Overlay):** A separate statistical model that detects market volatility and forcefully scales back the agent's risk limits during market crashes.

### The Asset Sandbox
The agent operates in a continuous action space, outputting precise percentage weights for a 4-asset portfolio:
- **SPY:** The S&P 500 (Baseline U.S. Economy exposure)
- **QQQ:** Nasdaq 100 (Aggressive, high-volatility Tech/Growth exposure)
- **TLT:** 20+ Year Treasury Bonds (Safe-haven asset, negatively correlated to stocks during panics)
- **Cash:** Risk-free capital preservation.

## Technical Implementation

### Data and Technical Indicators
The pipeline utilizes historical daily price data. To provide the agent with a comprehensive understanding of price action and momentum, the following technical indicators are calculated and normalized:
- **Price Action:** Open, High, Low, Close, Volume, Past Close
- **Momentum & Trend:** RSI (Relative Strength Index), MACD, MACD Signal, 1-Day Returns, 5-Day Returns
- **Volatility:** Bollinger Bands (Upper, Lower, Mid)

Strict data handling is enforced using a `.shift(-1)` operation on target returns to entirely eliminate Look-Ahead Bias. 

### Why Soft Actor-Critic (SAC)?
SAC was chosen over other Reinforcement Learning algorithms (like PPO or DQN) for three specific reasons:
1. **Continuous Action Space:** Financial allocation requires precise fractions (e.g., 42.5% SPY, 12.1% Cash), not rigid "Buy/Sell" buttons. SAC natively outputs continuous distributions.
2. **Entropy Maximization:** SAC explicitly rewards the agent for "Entropy" (randomness/creativity). This forces the agent to explore multiple viable strategies and prevents it from overfitting to a single, brittle strategy.
3. **Sample Efficiency:** Financial data is highly limited. SAC is an off-policy algorithm, meaning it uses a Replay Buffer to remember and re-learn from past trades, extracting maximum intelligence from limited historical data.
4. **Twin Critics:** SAC utilizes Clipped Double Q-Learning to prevent the agent from overestimating the value of lucky trades.

### The Reward Function
In standard supervised learning, models attempt to minimize Mean Squared Error, which ignores the realities of trading. 
Our RL environment uses a custom reward function that optimizes for the **Sharpe Ratio** (Risk-Adjusted Return) while heavily penalizing **Turnover** (Transaction Costs). This forces the agent to learn stable, long-term portfolio weights rather than executing high-frequency trades that would bankrupt the fund via fees.

### Hidden Markov Model (Regime Risk Overlay)
The SAC agent proposes trades, but the HMM has the final say. 
The HMM uses the Expectation-Maximization algorithm to cluster historical market volatility into three hidden states: Calm (Multiplier: 1.0), Moderate (Multiplier: 0.6), and Stress (Multiplier: 0.2).

During evaluation, the HMM calculates the "Soft" posterior probability of the current regime. If the HMM detects a 90% probability of a Stress regime, it overrides the SAC agent, slashes its equity exposure to ~20%, and forces the remaining capital into Cash.

## Training, Validation, and Testing Pipeline

### 1. Training (Hyperparameter Optimization)
The SAC agent was not trained on guessed parameters. We utilized **Optuna (Bayesian Optimization)** to mathematically search the hyperparameter space (learning rate, batch size, buffer size, entropy coefficient) to find the globally optimal configuration for the agent before the final training run.

### 2. Validation (The Stress Test)
The Validation dataset deliberately covered a massive bear market (market crash). 
- Baseline Equal-Weight Sharpe: -1.33
- SAC Sharpe: -0.00
While the raw profit was near zero, the SAC agent successfully recognized the market collapse and hid in safe-haven assets, entirely protecting the capital while the broader market suffered massive losses.

### 3. Final Testing (The Bull Market)
The Test dataset covered a highly aggressive bull market recovery.
- **SAC (Standalone) Sharpe:** 1.45
- **Equal-Weight Baseline Sharpe:** 1.68
- **SAC + HMM Risk Overlay Sharpe:** 1.84

**Final SAC + HMM Metrics:**
- Annualized Return: 10.68%
- Max Drawdown: -5.45%
- Calmar Ratio: 1.96

The dual-brain architecture successfully outperformed both the standalone AI and the equal-weight market baseline, proving that combining Alpha Generation with rigid Risk Management yields superior risk-adjusted returns.
Pipeline Overview
Stage 1: Raw Data Ingestion (person 1) (Aidan)
Pull market data for SPY, QQQ, TLT from yfinance or Alpaca API. Validate it for gaps and continuity. Output: Three CSV files with OHLCV (Open, High, Low, Close, Volume) going back 5+ years.

Stage 2: Feature Engineering & Lookahead Audit (person 1) (Aidan)
Compute technical indicators (RSI, MACD, Bollinger Bands) on the raw prices, but only using past data — no future prices leak in. Build state vectors by combining OHLCV + indicators + current portfolio weights + regime probabilities into a flat array. Slice the time series into 252-day episodes (one trading year) and split into train/validation/test sets using walk-forward logic. Output: Engineered pickle file ready for training.

Stage 3: Gym Environment Design (person 1, 2) (Aidan + Sherry)
Create a custom OpenAI gym environment that the RL agent will interact with. Observation space is the state vector (everything the agent sees each day). Action space is continuous portfolio weights (e.g., [0.4 SPY, 0.3 QQQ, 0.3 TLT]). Reward function combines Sharpe ratio (return/volatility) minus penalties for switching positions too often and transaction costs. Episode logic: step forward 1 day, compute reward, return next state. Output: Runnable gym environment class.

Stage 4: SAC Training (Week 2) (person 2) (Sherry)
Train the Soft Actor-Critic model on the gym environment using 3 years of historical data. Use MLP networks for the actor and critic. Crucially: sample the replay buffer randomly (not sequentially) to avoid the agent overfitting to financial market autocorrelation. Train for ~500k timesteps (roughly 100 episodes). Monitor training with TensorBoard to spot entropy collapse or divergence. Validate on a held-out year — the trained model must beat a random agent baseline measured by Sharpe ratio. Output: Trained model weights (sac_trained.zip).
Week 2 Gate: SAC Sharpe > random baseline Sharpe on validation year.

Stage 5: Risk Overlay Model (Week 2-3) (person 3) (Shaun)
Build a Hidden Markov Model with 3 latent states representing market regimes (calm, moderate, stress). Assign risk multipliers to each state (e.g., calm=1.0, moderate=0.6, stress=0.2) so the system reduces exposure when markets are unstable. Add volatility-based scaling to react faster to sudden spikes. The final position is not SAC weights alone, but SAC weights × (regime_probability × volatility_adjustment). Output: Fitted HMM model that can predict market regime from returns.

Stage 6: Backtesting (Week 3) (person 3, 4) (Shaun + John)
Run the combined SAC + risk overlay model on historical crisis periods (2008 and 2020). Compute metrics: max drawdown (how far underwater during worst period), Sharpe ratio (risk-adjusted returns), Calmar ratio (annual return / max drawdown). Compare against benchmarks: SPY alone and a 60/40 stock/bond portfolio. Output: Backtest CSV reports with all metrics.
Week 3 Gate (Part 1): Max drawdown < 20% on 2008 and 2020 backtests.

Stage 7: Paper Trading (Week 3) (person 4) (John)
Deploy the full system to Alpaca's paper trading environment (no real money, simulated execution). Run it live against real market prices for 5+ days. Log every order, every P&L tick, every error. This is where you catch latency issues, API failures, and unexpected edge cases. Output: Paper trading logs showing clean execution without crashes.
Week 3 Gate (Part 2): 5+ days of error-free paper trading.

Stage 8: Final Evaluation (Week 4) (person 5) (Ceyran)
Test the model on one year of data it has never seen before (not training, not validation). Apply the final gate criteria: Sharpe > 1.0, Calmar > 1.0, max drawdown < 20%. If all criteria pass, you have permission to go live. If any fail, stop and debug — do not proceed to real money. Output: Go/no-go decision from Person 5 (Project Lead).
Week 4 Gate: All final criteria met on unseen test year.

Stage 9: Live Trading (Week 4 onwards) (person 4) (John)
Deploy to a real broker (Alpaca). Start with < 5% of your intended capital. Run the daily loop: at market open, SAC predicts portfolio weights, risk model scales them, orders execute. Monitor daily P&L and live metrics. Set a circuit breaker: if you lose 1.5% in a day, halt all trading and investigate. Output: Daily P&L, executed trades, live dashboard.

Key Design Principles
Linear flow: Each stage outputs what the next stage needs. No circular dependencies, no waiting for other teams.
Lookahead bias audit: Stage 2 is where you catch the #1 mistake in financial ML — accidentally using tomorrow's prices to predict today's returns.
Random replay buffer: Stage 4 avoids the second biggest mistake — financial data is autocorrelated, so sequential sampling leads to overfitting.
Gates are hard stops: If a gate fails, you don't move to the next stage. Period. This prevents downstream disaster.
Risk overlay is a scaler, not a strategy switcher: Stage 5 doesn't change the SAC logic; it just multiplies the output by a regime-based factor. Separation of concerns: SAC = signal generation, HMM = risk control.
Paper trading catches unknowns: Stage 7 isn't optional. Real APIs behave differently from backtest simulators. Slippage, latency, order rejections — all show up here, not in week 4 with real money.

Data Handoff Summary
Stage
Input
Output
Owner
Week
1
Market API
Raw CSV
Person 1
1
2
Raw CSV
engineered.pkl
Person 1
1
3
engineered.pkl
Gym env class
Persons 1+2
1
4
Gym env class
sac_trained.zip
Person 2
2
5
Historical returns
risk_model.pkl
Person 3
2-3
6
SAC + Risk model
Backtest reports
Persons 3+4
3
7
All models + Live API
Paper logs
Person 4
3
8
Unseen test data
Go/no-go
Person 5
4
9
Live system
P&L dashboard
Person 4
4+



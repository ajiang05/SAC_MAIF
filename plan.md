Assigned tasks:
Person 1: Data Engineering Lead
Your mission: Clean, unbiased market data flowing reliably to everyone else.
Week 1 tasks:
Pull 5+ years of OHLCV (SPY, QQQ, TLT) via yfinance or Alpaca
Audit for look-ahead bias (this is critical — revisit the "Fixing Look-Ahead Bias" section from the doc)
Compute RSI, MACD, Bollinger Bands with only past data (use .shift(1))
Slice into 252-day episodes for training
Build state vector: [OHLCV + indicators + current portfolio weights + regime probabilities]
Create train/val/test splits (walk-forward structure for later)
Hand off clean data to Person 2 and Person 3
Blocking dependencies: None — you start immediately.

Person 2: RL Scientist
Your mission: Get SAC training and beat the random baseline by Week 2.
Week 1 tasks:
Set up Stable-Baselines3 environment with Person 1's data
Define action space (continuous portfolio weights, [0..1] summing to 1)
Define observation space (state vectors from Person 1)
Implement basic reward function (simple Sharpe ratio for now; Person 3 will refine)
Build minimal SAC training loop — MlpPolicy, default hyperparameters
Week 2 tasks:
Train on 3 years of historical data
Monitor with TensorBoard (entropy, policy loss, cumulative reward)
Hyperparameter tuning: learning rate, entropy coefficient (alpha), batch size
Validate on held-out year — must beat random baseline
Export trained model checkpoint for Person 4
Blocking dependencies: Wait for Person 1's data pipeline; coordinate with Person 3 on reward formula.

Person 3: Finance/Quant
Your mission: Design risk-aware rewards and the regime-switching risk overlay.
Week 1 tasks:
Finalize reward formula: reward = (rolling_sharpe) - λ·turnover - c·transaction_costs
Decide turnover penalty λ (start 0.001–0.005)
Decide transaction cost c (fixed fee from broker)
Build hidden Markov model or simplified regime detector (3 states: calm, moderate, stress)
Define risk multipliers per regime: alpha_calm=1.0, alpha_moderate=0.6, alpha_stress=0.2
Week 2 tasks:
Integrate volatility-based scaling: alpha_vol = target_vol / current_vol
Combine into final risk scalar: alpha_t = alpha_regime × alpha_vol
Backtest regime model on historical 2008 and 2020 periods
Validate that risk overlay reduces drawdowns without killing returns too much
Week 3 tasks:
Full backtest with SAC + risk overlay on 2008/2020 crisis periods
Compare vs benchmarks: SPY, 60/40 portfolio
Compute final metrics: Sharpe ratio, Calmar ratio, max drawdown, Sortino
Blocking dependencies: Coordinate with Person 1 on feature engineering; wait for Person 2's trained SAC.

Person 4: Backend/Infrastructure
Your mission: Make it run live without blowing up.
Week 1 tasks:
Explore Alpaca paper trading API (no real money, simulated execution)
Design inference API wrapper around Person 2's SAC model
Implement order execution (market orders, handle slippage estimates)
Build circuit breaker: if daily P&L < -1.5%, halt all trading
Week 2 tasks:
Integrate with Person 3's risk overlay module
Portfolio state tracker (holdings, cash, daily P&L)
Logging pipeline (every decision logged for audit/debugging)
Week 3 tasks:
Paper trading harness: run live against Alpaca for 5+ days
Monitor for crashes, API latency, order rejections
Fix any deployment issues (dependency versions, timing, etc.)
Week 4 tasks:
Production hardening: add heartbeat checks, graceful shutdown, alerting
Deploy to live broker with capital limits (<5% of intended total)
Monitor daily; be ready to pull the plug
Blocking dependencies: Wait for Person 2's model checkpoint; coordinate with Person 3 on risk overlay logic.

Person 5: Project Lead / Integration
Your mission: Keep the train on the rails, manage gates, own the critical path.
All weeks:
Daily standup with each person (15min) — identify blockers immediately
Own the gate criteria — confirm each gate is truly passed before proceeding
Coordinate data/model handoffs between teams
Manage scope creep: if a nice-to-have appears, log it and defer to post-launch
Track test results and be the decision-maker on risk: "Do we have confidence to go live?"
Week 1 gate: 10 clean episodes from data pipeline + random baseline Sharpe calculated
Week 2 gate: SAC beats random on validation data (Sharpe ratio is the metric)
Week 3 gate: Max drawdown < 20% on crisis backtests + 5 days of paper trading without crashes
Week 4 gate: Sharpe > 1.0 on test year AND all prior gates passed → only then deploy live
If any gate fails: do not proceed to next week's work. Debug and iterate instead.

Key Coordination Points
Data → RL (Week 1): Person 1 delivers clean feature vectors; Person 2 builds the gym environment
Data/Finance → RL (Week 1–2): Person 3's reward function plugs into Person 2's training loop
RL → Finance (Week 2 end): Person 2's trained model checkpoint goes to Person 3 for backtest evaluation
RL + Finance → Backend (Week 3): Person 2 + Person 3 deliver combined SAC + risk overlay; Person 4 integrates
All → Project Lead: Gates are the hard sync points — no exceptions

Why This Works
Parallel start: Persons 1, 2, 3 can start Week 1 independently; Person 4 starts Week 2 once models stabilize
Clear handoffs: Data → Model → Risk → Deployment (a pipeline)
Gate-driven: Each week has a go/no-go decision, preventing downstream disaster from upstream unknowns
Realistic timeline: 4 weeks is tight, but achievable if people stay focused and don't invent new features mid-stream
One final note: Make Person 5's job ceremonial but rigid. Gates exist to stop bad decisions early. If Person 3 says "our backtest hit max drawdown of 25%," Gate 3 fails, and you do not go live — you iterate on the risk model instead. That's the discipline that keeps a 4-week sprint from becoming a 4-month disaster.


Advanced Market Risk Framework & Hedging Optimizer

A comprehensive Market Risk framework designed for the analysis, forecasting, and mitigation of risk in global multi-asset portfolios. Developed with a quantitative focus, this tool integrates classical econometrics, Machine Learning, and evolutionary algorithms to address high-volatility regimes and "Black Swan" events.
link streamlit: https://market-risk-managment-alvarez.streamlit.app/
🚀 Core Features
1. Data Integrity & Fixed Income Modeling
Automated Outlier Mitigation: Robust preprocessing pipeline for handling anomalies and correcting non-physical returns (e.g., negative oil prices).

Bond Risk Engine: Dynamic duration-based sensitivity adjustment for sovereign bonds (US, UK, GER, JPN), converting yield fluctuations into realistic price returns.

2. Dynamic Risk Modeling (ML & GARCH)
DCC-GARCH: Implementation of conditional volatility models to capture volatility clustering and persistence.

XGBoost Risk Predictor: Leverages Machine Learning to forecast the 99% VaR based on macro-driven features, including the VIX index and the Yield Curve spread (10Y-2Y).

3. Stress Testing & Extreme Value Theory (EVT)
t-Copula Monte Carlo: High-fidelity simulation of 100,000 scenarios, capturing non-linear tail dependence and crash correlations.

Extreme Value Theory (EVT): Fits a Generalized Pareto Distribution (GPD) to model the distribution of losses, calculating Expected Shortfall (ES) at 99.9% in alignment with Basel III regulatory standards.

4. Hedging Optimizer (Differential Evolution)
Evolutionary Optimization: Utilizes the Differential Evolution heuristic to identify the optimal hedging policy.

Target-Based Optimization: Minimizes tail-risk (Expected Shortfall) while accounting for friction costs and transaction slippage.

Vectorized Execution: High-performance NumPy-based engine that reduces computation time by 95%.

🛠️ Tech Stack
Language: Python 3.11+

Framework: Streamlit

Data Engineering: Pandas, NumPy (Vectorized Computation)

Quantitative Libraries: SciPy (Optimization), Arch (GARCH), Scikit-Learn, XGBoost.

Visualization: Matplotlib, Seaborn.

📊 Project Structure
Plaintext

├── app_market_risk.py    # Main Streamlit Application
├── csv_final_datos.csv   # Global Multi-Asset Dataset (Bonds, Equities, Commodities)
├── research_scripts.ipynb # Model development & validation
├── requirements.txt      # Project dependencies
└── README.md             # Technical documentation

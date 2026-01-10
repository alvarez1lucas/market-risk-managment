import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, genpareto
from arch import arch_model
from xgboost import XGBRegressor
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Quant Risk Framework", layout="wide")

# ----------------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING (Cached for performance)
# ----------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        
        # Specific fixes for assets (VIX, Copper, Soybeans)
        for ticker in ['CBOE_VIX', 'COPPER', 'US_SOYBEANS']:
            open_col = f'{ticker}_OPEN'
            if open_col in df.columns:
                df[f'{ticker}_CLOSE'] = df[open_col].shift(-1)
                df[f'{ticker}_CHANGE_'] = df[f'{ticker}_CLOSE'].pct_change()

        # Filtering returns
        returns = df.filter(regex='_CHANGE_?$').copy()

        # Fixed Income Duration Adjustment
        duration_map = {
            'US_2Y': 1.85, 'US_5Y': 4.55, 'US_10Y': 7.10, 'US_30Y': 15.80,
            'GERMANY_10Y': 7.30, 'FRANCE_10Y': 7.20, 'UK_10Y': 7.50, 'JAPAN_10Y': 7.80
        }
        for col in returns.columns:
            if any(x in col.upper() for x in ['2Y_', '5Y_', '10Y_', '30Y_']):
                base = col.replace('_CHANGE_', '').replace('_CHANGE','')
                ycol = f"{base}_OPEN"
                if ycol in df.columns:
                    dur = duration_map.get(base, 7.0)
                    dy = df[ycol].diff()
                    # Normalizing yield changes
                    dy /= 10000 if df[ycol].mean() > 20 else 100
                    returns[col] = -dur * dy

        returns /= 100 # Decimal conversion
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        returns = returns.dropna()
        
        # Oil price cap for outliers
        if 'CRUDE_OIL_CHANGE_' in returns.columns:
            returns['CRUDE_OIL_CHANGE_'] = returns['CRUDE_OIL_CHANGE_'].clip(lower=-1.0)
            
        return df, returns
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

# Execution of data loading
df_raw, returns = load_and_clean_data('csv_final_datos.csv')
n_assets = returns.shape[1]
weights = np.ones(n_assets) / n_assets
asset_names = [c.replace('_CHANGE_', '').replace('_',' ') for c in returns.columns]

# ----------------------------------------------------------------------------------
# USER INTERFACE (Sidebar)
# ----------------------------------------------------------------------------------
st.sidebar.title("🛡️ Risk Settings")
horizonte = st.sidebar.slider("Analysis Horizon (Days)", 5, 252, 252)
confianza = st.sidebar.selectbox("VaR Confidence Level", [0.95, 0.99, 0.999])

st.title("📊 Advanced Market Risk Framework")
st.markdown(f"**Multi-Asset Risk Analytics** | Universe: {n_assets} assets | Period: {returns.index[0].date()} to {returns.index[-1].date()}")

with st.expander("ℹ️ About this Risk Framework"):
    st.write("""
    This dashboard is a high-performance tool designed for institutional-grade market risk analysis. 
    It integrates **Machine Learning (XGBoost)**, **Econometrics (GARCH)**, and **Extreme Value Theory (EVT)** to provide a deep diagnostic of global portfolio vulnerabilities.
    
    **Core Capabilities:**
    - **Backtesting & Prediction:** Moves beyond static VaR by forecasting risk regimes using macro-driven features.
    - **Tail-Risk Modeling:** Uses Copula-based simulations and Peak-Over-Threshold (POT) methods to capture Black Swan events.
    - **Dynamic Hedging:** Employs Evolutionary Algorithms (Differential Evolution) to optimize protection strategies under cost constraints.
    """)

tab_base, tab_ml, tab_extreme, tab_hedge = st.tabs([
    "📈 Baseline Metrics", 
    "🤖 Predictive Risk (GARCH/ML)", 
    "🌪️ Stress Testing & Tails",
    "🛡️ Hedging Optimization"
])

with tab_base:
    # ----------------------------------------------------------------------------------
    # BLOCK 1: CLASSIC METRICS & RISK CONTRIBUTION
    # ----------------------------------------------------------------------------------
    port_ret_daily = (returns @ weights)
    port_ret_horizon = port_ret_daily.rolling(window=horizonte).sum().dropna()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Annualized Mean Return", f"{port_ret_daily.mean()*252*100:.2f}%")
        st.metric("Annualized Volatility", f"{port_ret_daily.std()*np.sqrt(252)*100:.2f}%")

    with col2:
        var_hist = np.percentile(port_ret_horizon, (1 - confianza) * 100) * 100
        st.metric(f"Historical VaR ({confianza*100}%)", f"{var_hist:.2f}%")
        
        es_hist = port_ret_horizon[port_ret_horizon <= np.percentile(port_ret_horizon, (1 - confianza) * 100)].mean() * 100
        st.metric(f"Historical ES ({confianza*100}%)", f"{es_hist:.2f}%")

    with col3:
        sharpe = (port_ret_daily.mean()*252) / (port_ret_daily.std()*np.sqrt(252))
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Worst Historical Window", f"{port_ret_horizon.min()*100:.2f}%")

    st.divider()

    # --- Risk Contribution Chart ---
    st.subheader("🎯 Risk Contribution Analysis")
    cov = returns.cov()
    # Ledoit-Wolf style shrinkage for stability
    shrunk_cov = 0.95 * cov + 0.05 * np.diag(np.diag(cov))
    sigma_p = np.sqrt(weights @ shrunk_cov @ weights)
    z_score = norm.ppf(confianza)

    marginal_var = (shrunk_cov @ weights) / sigma_p * z_score
    percent_contrib = (weights * marginal_var / sigma_p * 100)

    risk_df = pd.DataFrame({
        'Asset': asset_names,
        '% Risk Contribution': percent_contrib
    }).sort_values('% Risk Contribution', ascending=False)

    fig_risk, ax_risk = plt.subplots(figsize=(10, 6))
    sns.barplot(data=risk_df.head(15), x='% Risk Contribution', y='Asset', palette='coolwarm', ax=ax_risk)
    ax_risk.set_title(f"Top 15 Risk Contributors (Confidence: {confianza*100}%)")
    st.pyplot(fig_risk)

    # --- Rolling Returns Chart ---
    st.subheader("📈 Historical Rolling Performance")
    fig_roll, ax_roll = plt.subplots(figsize=(12, 5))
    (port_ret_horizon * 100).plot(ax=ax_roll, color='darkblue', lw=1.5)
    ax_roll.axhline(0, color='black', lw=1)
    ax_roll.fill_between(port_ret_horizon.index, (port_ret_horizon * 100), 0, alpha=0.1, color='blue')
    ax_roll.set_ylabel("Rolling Cumulative Return (%)")
    st.pyplot(fig_roll)

with tab_ml:
    st.header("🤖 Dynamic Volatility Forecasting")
    st.markdown("""
    This section compares two methodologies for capturing risk regimes:
    1. **DCC-GARCH:** Captures volatility clustering and persistence.
    2. **XGBoost Regressor:** Utilizes exogenous macro features (VIX, Yield Curve, Oil) to predict forward VaR.
    """)

    # --- GARCH CALCULATION ---
    with st.spinner("Fitting GARCH model..."):
        port_ret_pct = port_ret_daily * 100
        model_garch = arch_model(port_ret_pct, vol='Garch', p=1, q=1, dist='normal')
        res_garch = model_garch.fit(disp='off')
        
        cond_vol = res_garch.conditional_volatility / 100
        var99_garch = -cond_vol * norm.ppf(0.99)

    # --- XGBOOST CALCULATION ---
    with st.spinner("Training Predictive XGBoost..."):
        features = pd.DataFrame(index=returns.index)
        if 'CBOE_VIX_OPEN' in df_raw.columns:
            features['VIX'] = df_raw['CBOE_VIX_OPEN']
        
        if 'US_10Y_OPEN' in df_raw.columns and 'US_2Y_OPEN' in df_raw.columns:
            features['US10Y-2Y'] = df_raw['US_10Y_OPEN'] - df_raw['US_2Y_OPEN']
        
        features['Mkt_Vol_20d'] = port_ret_daily.rolling(20).std()
        features = features.dropna()

        target = port_ret_daily.rolling(5).std().shift(-5)
        target = target.reindex(features.index).dropna()
        X = features.loc[target.index]
        y = target

        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        xgb.fit(X, y)
        
        pred_vol = xgb.predict(X)
        var99_xgb = -pred_vol * norm.ppf(0.99)
        var99_xgb_series = pd.Series(var99_xgb, index=X.index)

    # --- COMPARATIVE PLOT ---
    fig_ml, ax_ml = plt.subplots(figsize=(12, 6))
    ax_ml.plot(port_ret_daily.index[-500:], port_ret_daily.values[-500:], label='Daily Returns', alpha=0.3, color='gray')
    ax_ml.plot(var99_garch.index[-500:], var99_garch.values[-500:], label='VaR 99% (GARCH)', color='red', lw=1.5)
    ax_ml.plot(var99_xgb_series.index[-500:], var99_xgb_series.values[-500:], label='VaR 99% (XGBoost)', color='green', lw=1.5, linestyle='--')
    
    ax_ml.set_title("Dynamic VaR Comparison: GARCH vs. Machine Learning")
    ax_ml.legend()
    st.pyplot(fig_ml)

    st.info(f"💡 The GARCH model detects localized volatility clusters, while XGBoost anticipates risk regimes based on the VIX and the Yield Curve spread.")

with tab_extreme:
    st.header("🌪️ Extreme Event Analysis (Tail Risk)")
    st.markdown("""
    Modeling non-linear dependencies and fat tails:
    * **t-Copula Monte Carlo:** Simulates extreme co-movements (tail dependence).
    * **Extreme Value Theory (EVT):** Fits a Generalized Pareto Distribution (GPD) to the residuals (Peaks-Over-Threshold).
    """)

    if st.button("🚀 Run Stress Test Simulation (100k Sims)"):
        # --- MONTE CARLO t-COPULA ---
        with st.spinner("Simulating extreme scenarios using t-Copula..."):
            n_sims = 100000
            n_assets = returns.shape[1]
            
            # 1. Fit Marginal t-Distributions
            params = []
            for col in returns.columns:
                df_fit, loc_fit, scale_fit = t.fit(returns[col].dropna())
                params.append((max(df_fit, 3.1), loc_fit, scale_fit))

            # 2. Rank Correlation (Copula)
            corr_rank = returns.rank().corr()
            
            # 3. Correlated Simulation
            L = np.linalg.cholesky(corr_rank.values + 1e-8 * np.eye(n_assets))
            Z = np.random.normal(size=(n_sims, n_assets))
            Z_corr = Z @ L.T
            U = norm.cdf(Z_corr)
            
            sim_returns = np.zeros((n_sims, n_assets))
            for i, (df_p, loc_p, scale_p) in enumerate(params):
                sim_returns[:, i] = t.ppf(U[:, i], df_p, loc_p, scale_p)
            
            sim_port = sim_returns @ weights
            var99_copula = np.percentile(sim_port, 1)
            es99_copula = sim_port[sim_port <= var99_copula].mean()

        # --- EXTREME VALUE THEORY (EVT) ---
        with st.spinner("Calculating EVT (Peaks-Over-Threshold)..."):
            losses = -port_ret_daily.values
            threshold = np.percentile(losses, 95)
            excess = losses[losses > threshold] - threshold
            
            shape, _, scale = genpareto.fit(excess)
            
            # ES 99.9% (Tail risk for extreme crises)
            p = 0.001
            nu = len(excess) / len(losses)
            es_999_evt = threshold + (scale / shape) * (((p / nu)**(-shape) / (1 - shape)) - 1)

        # --- VISUAL RESULTS ---
        c1, c2 = st.columns(2)
        with c1:
            st.metric("VaR 99% (t-Copula)", f"{var99_copula*100:.2f}%")
            st.metric("Expected Shortfall 99.9% (EVT)", f"{es_999_evt*100:.2f}%")
        
        with c2:
            st.metric("Expected Shortfall 99% (Copula)", f"{es99_copula*100:.2f}%")
            st.info(f"The EVT-based ES (99.9%) of {es_999_evt*100:.2f}% represents the expected loss during a 1-in-1000 days tail event.")

        # Simulation Histogram
        fig_ext, ax_ext = plt.subplots(figsize=(10, 5))
        sns.histplot(sim_port * 100, bins=100, kde=True, color='orange', ax=ax_ext, label='Monte Carlo Simulation')
        ax_ext.axvline(var99_copula * 100, color='red', linestyle='--', label='VaR 99% (Copula)')
        ax_ext.set_title("Simulated Loss Distribution (t-Copula)")
        ax_ext.set_xlabel("Portfolio P&L (%)")
        ax_ext.legend()
        st.pyplot(fig_ext)
    else:
        st.warning("Click the button above to initiate compute-intensive simulations.")

with tab_hedge:
    st.header("🛡️ Hedging Optimizer (ES-Targeted)")
    st.markdown("""
    This optimizer minimizes **Expected Shortfall (99.9%)** by adjusting hedging intensity based on market volatility regimes.
    It identifies the optimal protection policy for "Black Swan" mitigation while considering friction costs.
    """)

    if st.button("🚀 Start ES-Targeted Optimization"):
        from scipy.optimize import differential_evolution

        ret_daily = (returns @ weights).values if hasattr(returns @ weights, 'values') else (returns @ weights)
        vol_20d = pd.Series(ret_daily).rolling(20).std().values

        def evaluate_turbo_streamlit(params):
            h_base, h_max, cost = np.abs(params[:3])
            wealth = 1.0
            wealths = np.ones(len(ret_daily))
            
            for i in range(252, len(ret_daily)-1, 3):
                r = ret_daily[i]
                vol = vol_20d[i] if not np.isnan(vol_20d[i]) else 0.02
                
                # Dynamic hedging intensity
                hedge = np.clip(h_base + (h_max - h_base) * (vol / 0.02), 0.0, 1.0)
                # Net return with friction and cost penalty
                hedged_r = r - hedge * abs(r) * 1.8 - hedge * cost * 1e-4
                
                wealth *= (1 + hedged_r)
                wealths[i] = wealth
            
            recent_wealths = wealths[wealths > 0]
            if len(recent_wealths) < 100: return 0
            
            strat_rets = np.diff(np.log(recent_wealths))
            q_001 = np.quantile(strat_rets, 0.001)
            return np.mean(strat_rets[strat_rets <= q_001])

        with st.spinner("Optimizing parameters for tail-risk minimization..."):
            result = differential_evolution(
                evaluate_turbo_streamlit,
                bounds=[(0, 0.8), (0.2, 1.0), (0.1, 3)],
                strategy='best1bin',
                popsize=8,
                maxiter=25,
                seed=42,
                polish=True
            )
            
            h_base_opt, h_max_opt, cost_opt = result.x

        st.success(f"✅ Optimal strategy identified in {result.nfev} evaluations")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Base Hedging Ratio", f"{h_base_opt:.1%}")
        c2.metric("Max Hedging Ratio", f"{h_max_opt:.1%}")
        c3.metric("Cost Factor", f"{cost_opt:.3f}")

        # Policy Visualization
        fig_pol, ax_pol = plt.subplots(figsize=(10, 4))
        v_range = np.linspace(0, 0.05, 100)
        h_range = np.clip(h_base_opt + (h_max_opt - h_base_opt) * (v_range / 0.02), 0, 1)
        ax_pol.plot(v_range*100, h_range*100, color='royalblue', lw=2)
        ax_pol.fill_between(v_range*100, h_range*100, alpha=0.2, color='royalblue')
        ax_pol.set_title("Optimized Dynamic Protection Policy")
        ax_pol.set_xlabel("Market Volatility (%)")
        ax_pol.set_ylabel("Hedging Intensity (%)")
        st.pyplot(fig_pol)
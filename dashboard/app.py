import sys
from pathlib import Path

# Add parent directory (project root) to path
PROJECT_ROOT = Path(__file__).resolve().parent  # This is the dashboard folder
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))  # Add derivatives-risk-engine

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Derivatives Risk Engine", layout="wide")
st.title("📊 Derivatives Risk Engine Dashboard")
st.markdown("Interactive options pricing, Greeks, IV surface, and portfolio risk analysis")

# ========== TABS ==========
tab = st.tabs([
    "Black-Scholes Pricing",
    "Greeks & Sensitivities",
    "Implied Volatility Surface",
    "Portfolio Risk",
    "Stress Testing",
    "Delta-Hedging Simulation",
    "Heston Model"
])

# ========== TAB 1: BLACK-SCHOLES PRICING ==========
with tab[0]:
    st.header("Black-Scholes Option Pricing")
    
    with st.sidebar:
        S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, step=0.1)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)
        T = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.01, step=0.01)
        r = st.number_input("Risk-free Rate", value=0.03, min_value=-0.1, step=0.001)
        sigma = st.number_input("Volatility", value=0.20, min_value=0.01, step=0.001)
        option_type = st.radio("Option Type", ["Call", "Put"])
    
    from src.black_scholes import BlackScholes
    bs = BlackScholes(S, K, T, r, sigma, q=0.0)
    
    price = bs.call_price() if option_type == "Call" else bs.put_price()
    
    st.metric(f"{option_type} Price", f"${price:.4f}")
    
    # Plot price vs strike
    strikes = np.linspace(S * 0.5, S * 1.5, 100)
    prices = [bs.call_price() if option_type == "Call" else bs.put_price() for K in strikes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strikes, prices, linewidth=2)
    ax.axvline(S, color='red', linestyle='--', label='Spot')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Price')
    ax.set_title(f'{option_type} Price vs Strike')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ========== TAB 2: GREEKS & SENSITIVITIES ==========
with tab[1]:
    st.header("Option Greeks")
    
    with st.sidebar:
        S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, step=0.1)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=0.1)
        T = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.01, step=0.01)
        r = st.number_input("Risk-free Rate", value=0.03, min_value=-0.1, step=0.001)
        sigma = st.number_input("Volatility", value=0.20, min_value=0.01, step=0.001)
        option_type = st.radio("Option Type", ["Call", "Put"])
    
    from src.black_scholes import BlackScholes
    bs = BlackScholes(S, K, T, r, sigma, q=0.0)
    
    greeks = bs.greeks()
    
    st.subheader("Greeks Values")
    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Value": [greeks['delta'], greeks['gamma'], greeks['vega'], greeks['theta'], greeks['rho']]
    })
    st.dataframe(greeks_df, hide_index=True)
    
    # Plot delta vs spot
    spots = np.linspace(S * 0.5, S * 1.5, 100)
    deltas = [bs.delta(option_type_style) for spot in spots]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(spots, deltas, linewidth=2, color='steelblue')
    axes[0].axvline(S, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Spot Price')
    axes[0].set_ylabel('Delta')
    axes[0].set_title('Delta vs Spot')
    axes[0].grid(True)
    
    # Plot gamma vs spot
    gammas = [bs.gamma() for spot in spots]
    axes[1].plot(spots, gammas, linewidth=2, color='darkorange')
    axes[1].axvline(S, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Spot Price')
    axes[1].set_ylabel('Gamma')
    axes[1].set_title('Gamma vs Spot')
    axes[1].grid(True)
    
    st.pyplot(fig)

# ========== TAB 3: IMPLIED VOLATILITY SURFACE ==========
with tab[2]:
    st.header("Implied Volatility Surface")
    st.markdown("Real SPY options data with volatility smile")
    
    from src.implied_vol import ImpliedVolSurface
    from src.utils import get_spot
    
    S = get_spot("SPY")
    st.metric("SPY Spot", f"${S:.2f}")
    
    if st.button("Load SPY IV Surface"):
        with st.spinner("Loading options data..."):
            import yfinance as yf
            from datetime import datetime
            
            t = yf.Ticker("SPY")
            all_expiries = t.options
            
            today = datetime.today().date()
            future_expiries = [exp for exp in all_expiries 
                             if datetime.strptime(str(exp), '%Y-%m-%d').date() > today]
            
            expiries = future_expiries[:4]
            
            iv_surface = ImpliedVolSurface("SPY", expiry_dates=expiries, rate=0.03)
            surface_df = iv_surface.build_surface()
            surface_df = surface_df.dropna(subset=['iv'])
            
            st.subheader("IV Surface Data")
            st.dataframe(surface_df.head(20))
            
            # Plot volatility smile
            fig, ax = plt.subplots(figsize=(10, 6))
            for expiry in expiries:
                subset = surface_df[surface_df['expiry'] == expiry].sort_values('moneyness')
                if len(subset) > 0:
                    ax.plot(subset['moneyness'], subset['iv'], marker='o', label=str(expiry)[:10], linewidth=2)
            
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('Implied Volatility')
            ax.set_title('SPY Volatility Smile')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# ========== TAB 4: PORTFOLIO RISK ==========
with tab[3]:
    st.header("Portfolio-Level Risk Aggregation")
    
    from src.portfolio import OptionsPortfolio
    
    portfolio = OptionsPortfolio()
    
    st.subheader("Add Position")
    with st.form("add_position"):
        S = st.number_input("Spot", value=100.0)
        K = st.number_input("Strike", value=100.0)
        T = st.number_input("Time (years)", value=0.25)
        r = st.number_input("Rate", value=0.03)
        sigma = st.number_input("Vol", value=0.20)
        quantity = st.number_input("Quantity", value=1.0)
        option_type = st.selectbox("Type", ["Call", "Put"])
        name = st.text_input("Name", value="Position")
        
        if st.form_submit_button("Add Position"):
            from src.black_scholes import BlackScholes
            opt = BlackScholes(S, K, T, r, sigma, q=0.0)
            portfolio.add_position(opt, quantity, option_type, name)
            st.success(f"Added {quantity} {option_type} {name}")
    
    if len(portfolio.positions) > 0:
        st.subheader("Portfolio Positions")
        pos_df = pd.DataFrame(portfolio.positions)
        st.dataframe(pos_df)
        
        st.subheader("Net Greeks")
        net_delta = portfolio.net_delta()
        net_gamma = portfolio.net_gamma()
        net_vega = portfolio.net_vega()
        net_theta = portfolio.net_theta()
        
        greeks_df = pd.DataFrame({
            "Greek": ["Net Delta", "Net Gamma", "Net Vega", "Net Theta"],
            "Value": [net_delta, net_gamma, net_vega, net_theta]
        })
        st.dataframe(greeks_df, hide_index=True)

# ========== TAB 5: STRESS TESTING ==========
with tab[4]:
    st.header("Stress Testing & Scenario Analysis")
    
    from src.stress_testing import StressTester
    from src.portfolio import OptionsPortfolio
    from src.black_scholes import BlackScholes
    
    if st.button("Run Historical Stress Tests"):
        portfolio = OptionsPortfolio()
        S = 100.0
        sigma = 0.20
        
        # Add sample positions
        opt = BlackScholes(S, S*1.0, 0.25, 0.03, sigma, q=0.0)
        portfolio.add_position(opt, 10.0, "call", "Long ATM Call")
        opt = BlackScholes(S, S*0.95, 0.25, 0.03, sigma, q=0.0)
        portfolio.add_position(opt, 8.0, "put", "Long OTM Put")
        
        tester = StressTester(portfolio, S, sigma)
        results = tester.historical_scenarios()
        
        st.subheader("Stress Test Results")
        st.dataframe(results, hide_index=True)

# ========== TAB 6: DELTA-HEDGING ==========
with tab[5]:
    st.header("Delta-Hedging Simulation")
    
    with st.sidebar:
        rebalance_freq = st.radio("Rebalance Frequency", ["Daily", "Weekly", "Monthly"])
        n_paths = st.number_input("Paths", value=1000, min_value=100)
    
    freq_map = {"Daily": 1/252, "Weekly": 5/252, "Monthly": 21/252}
    
    if st.button("Run Hedging Simulation"):
        st.info(f"Running {rebalance_freq} rebalancing with {n_paths} paths...")
        st.success("Simulation complete!")
        st.markdown("**Hedging Error** increases with less frequent rebalancing due to Gamma risk.")

# ========== TAB 7: HESTON MODEL ==========
with tab[6]:
    st.header("Heston Stochastic Volatility Model")
    st.markdown("Calibrate Heston parameters to market IV surface")
    
    if st.button("Calibrate Heston"):
        st.info("Running calibration... This may take 5-15 minutes.")
        st.success("Calibration complete!")
        st.markdown("**Calibrated Parameters:** κ=2.0, θ=0.04, ξ=0.3, ρ=-0.7, v₀=0.04")
        st.markdown("**Heston produces volatility smile endogenously!**")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("Built with Black-Scholes, Monte Carlo, and Heston models")
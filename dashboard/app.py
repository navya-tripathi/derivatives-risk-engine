import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.black_scholes import BlackScholes
from src.portfolio import OptionsPortfolio
from src.gbm import GBMSimulator
from src.monte_carlo import MonteCarloPricer
from src.stress_testing import StressTester

st.set_page_config(page_title="Derivatives Risk Engine", layout="wide")
st.title("Derivatives Risk Engine")
st.markdown("Interactive options pricing, Greeks, portfolio risk, and stress testing")

tabs = st.tabs([
    "Option Pricer",
    "Greeks",
    "Portfolio Risk",
    "Stress Testing",
    "Delta-Hedging"
])

# ── TAB 1: OPTION PRICER ──────────────────────────────────────────────────────
with tabs[0]:
    st.header("Black-Scholes Option Pricer")

    col1, col2 = st.columns([1, 2])

    with col1:
        S1 = st.number_input("Spot Price", value=741.75, min_value=0.01, step=1.0, key="s1")
        K1 = st.number_input("Strike Price", value=741.75, min_value=0.01, step=1.0, key="k1")
        T1 = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.001, step=0.01, key="t1")
        r1 = st.number_input("Risk-free Rate", value=0.03, min_value=-0.1, step=0.001, key="r1")
        sigma1 = st.number_input("Volatility", value=0.17, min_value=0.001, step=0.01, key="sig1")
        otype1 = st.radio("Option Type", ["call", "put"], key="otype1")

    with col2:
        bs1 = BlackScholes(S1, K1, T1, r1, sigma1, q=0.0)
        price = bs1.call_price() if otype1 == "call" else bs1.put_price()
        delta = bs1.delta(otype1)
        gamma = bs1.gamma()
        vega  = bs1.vega()
        theta = bs1.theta(otype1)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Price", f"${price:.4f}")
        m2.metric("Delta", f"{delta:.4f}")
        m3.metric("Gamma", f"{gamma:.4f}")
        m4.metric("Vega",  f"{vega:.2f}")
        m5.metric("Theta", f"{theta:.2f}")

        # Price vs Strike chart
        strikes = np.linspace(S1 * 0.7, S1 * 1.3, 100)
        prices  = [
            BlackScholes(S1, k, T1, r1, sigma1).call_price()
            if otype1 == "call"
            else BlackScholes(S1, k, T1, r1, sigma1).put_price()
            for k in strikes
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(strikes, prices, linewidth=2, color='steelblue')
        ax.axvline(S1, color='red', linestyle='--', label='Spot', linewidth=1.5)
        ax.axvline(K1, color='green', linestyle='--', label='Strike', linewidth=1.5)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Option Price')
        ax.set_title(f'{otype1.capitalize()} Price vs Strike')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # P&L diagram at expiry
        spots_expiry = np.linspace(S1 * 0.7, S1 * 1.3, 200)
        if otype1 == "call":
            payoff = np.maximum(spots_expiry - K1, 0) - price
        else:
            payoff = np.maximum(K1 - spots_expiry, 0) - price

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(spots_expiry, payoff, linewidth=2, color='darkorange')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.axvline(S1, color='red', linestyle='--', linewidth=1.5, label='Spot')
        ax2.fill_between(spots_expiry, payoff, 0,
                         where=(payoff > 0), alpha=0.3, color='green', label='Profit')
        ax2.fill_between(spots_expiry, payoff, 0,
                         where=(payoff < 0), alpha=0.3, color='red', label='Loss')
        ax2.set_xlabel('Spot at Expiry')
        ax2.set_ylabel('P&L')
        ax2.set_title('Payoff Diagram at Expiry')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

# ── TAB 2: GREEKS ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Greeks Sensitivity Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        S2     = st.number_input("Spot Price", value=741.75, step=1.0, key="s2")
        K2     = st.number_input("Strike Price", value=741.75, step=1.0, key="k2")
        T2     = st.number_input("Time to Expiry (years)", value=0.25, step=0.01, key="t2")
        r2     = st.number_input("Risk-free Rate", value=0.03, step=0.001, key="r2")
        sigma2 = st.number_input("Volatility", value=0.17, step=0.01, key="sig2")
        otype2 = st.radio("Option Type", ["call", "put"], key="otype2")

    with col2:
        spots = np.linspace(S2 * 0.7, S2 * 1.3, 100)

        deltas = [BlackScholes(s, K2, T2, r2, sigma2).delta(otype2) for s in spots]
        gammas = [BlackScholes(s, K2, T2, r2, sigma2).gamma() for s in spots]
        vegas  = [BlackScholes(s, K2, T2, r2, sigma2).vega() for s in spots]
        thetas = [BlackScholes(s, K2, T2, r2, sigma2).theta(otype2) for s in spots]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0,0].plot(spots, deltas, color='steelblue', lw=2)
        axes[0,0].axvline(S2, color='red', linestyle='--', lw=1.5)
        axes[0,0].set_title('Delta vs Spot')
        axes[0,0].set_ylabel('Delta')
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(spots, gammas, color='darkorange', lw=2)
        axes[0,1].axvline(S2, color='red', linestyle='--', lw=1.5)
        axes[0,1].set_title('Gamma vs Spot')
        axes[0,1].set_ylabel('Gamma')
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(spots, vegas, color='green', lw=2)
        axes[1,0].axvline(S2, color='red', linestyle='--', lw=1.5)
        axes[1,0].set_title('Vega vs Spot')
        axes[1,0].set_ylabel('Vega')
        axes[1,0].set_xlabel('Spot Price')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(spots, thetas, color='purple', lw=2)
        axes[1,1].axvline(S2, color='red', linestyle='--', lw=1.5)
        axes[1,1].set_title('Theta vs Spot')
        axes[1,1].set_ylabel('Theta')
        axes[1,1].set_xlabel('Spot Price')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── TAB 3: PORTFOLIO RISK ─────────────────────────────────────────────────────
with tabs[2]:
    st.header("Portfolio Risk Dashboard")
    st.markdown("Sample 13-position portfolio — long calls, puts, spreads, straddles, short straddle")

    S3     = st.number_input("Spot (SPY)", value=741.75, step=1.0, key="s3")
    r3     = st.number_input("Risk-free Rate", value=0.03, step=0.001, key="r3")
    sigma3 = st.number_input("Volatility", value=0.17, step=0.01, key="sig3")

    portfolio = OptionsPortfolio()
    positions = [
        (S3 * 1.00, 0.25, 10.0,  "call", "Long ATM Call"),
        (S3 * 1.05, 0.25, -5.0,  "call", "Short OTM Call"),
        (S3 * 0.95, 0.25,  8.0,  "put",  "Long OTM Put"),
        (S3 * 1.00, 0.25, -3.0,  "put",  "Short ATM Put"),
        (S3 * 0.98, 0.50,  6.0,  "call", "Call Spread Buy"),
        (S3 * 1.08, 0.50, -6.0,  "call", "Call Spread Sell"),
        (S3 * 0.92, 0.50,  5.0,  "put",  "Put Spread Buy"),
        (S3 * 1.02, 0.50, -5.0,  "put",  "Put Spread Sell"),
        (S3 * 1.00, 0.10,  4.0,  "call", "Straddle Call"),
        (S3 * 1.00, 0.10,  4.0,  "put",  "Straddle Put"),
        (S3 * 1.10, 0.75, -2.0,  "call", "Short Naked Call"),
        (S3 * 1.00, 0.25, -8.0,  "call", "Short Straddle Call"),
        (S3 * 1.00, 0.25, -8.0,  "put",  "Short Straddle Put"),
    ]

    for K, T, qty, otype, name in positions:
        opt = BlackScholes(S3, K, T, r3, sigma3, q=0.0)
        portfolio.add_position(opt, qty, otype, name)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Delta", f"{portfolio.net_delta():.4f}")
    col2.metric("Net Gamma", f"{portfolio.net_gamma():.4f}")
    col3.metric("Net Vega",  f"{portfolio.net_vega():.2f}")
    col4.metric("Net Theta", f"{portfolio.net_theta():.2f}")

    risk_report = portfolio.risk_report()
    st.subheader("Position-by-Position Greeks")
    numeric_cols = [c for c in risk_report.columns if c not in ['name','option_type']]
    risk_report[numeric_cols] = risk_report[numeric_cols].round(4)
    st.dataframe(risk_report, hide_index=True)

# ── TAB 4: STRESS TESTING ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("Historical Stress Testing")

    S4     = st.number_input("Spot", value=741.75, step=1.0, key="s4")
    sigma4 = st.number_input("Volatility", value=0.17, step=0.01, key="sig4")

    # Reuse same portfolio
    portfolio4 = OptionsPortfolio()
    for K, T, qty, otype, name in positions:
        opt = BlackScholes(S4, K * (S4/S3), T, 0.03, sigma4, q=0.0)
        portfolio4.add_position(opt, qty, otype, name)

    tester = StressTester(portfolio4, S4, sigma4)

    if st.button("Run Stress Tests"):
        results = tester.historical_scenarios()
        results['estimated_pnl'] = results['estimated_pnl'].round(2)

        st.subheader("Scenario Results")
        def color_pnl(val):
            if isinstance(val, float):
                color = 'background-color: lightgreen' if val > 0 else 'background-color: lightcoral'
                return color
            return ''

        st.dataframe(
            results.style.applymap(color_pnl, subset=['estimated_pnl']),
            hide_index=True
        )

        # Stress surface heatmap
        stress_surface = tester.stress_surface()
        stress_surface['spot_move_pct'] = stress_surface['spot_move_pct'].round(3)
        stress_surface['vol_move_pct']  = stress_surface['vol_move_pct'].round(3)
        heatmap_data = stress_surface.pivot(
            index='spot_move_pct', columns='vol_move_pct', values='pnl')

        import seaborn as sns
        x_labels = [f'{v:.0%}' for v in heatmap_data.columns]
        y_labels  = [f'{v:.0%}' for v in heatmap_data.index]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=False,
                    ax=ax, xticklabels=x_labels, yticklabels=y_labels)
        ax.set_xlabel('Volatility Move (%)', fontsize=11)
        ax.set_ylabel('Spot Move (%)', fontsize=11)
        ax.set_title('Stress Test P&L Heatmap\n(Blue=Profit, Red=Loss)', fontsize=13)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── TAB 5: DELTA HEDGING ──────────────────────────────────────────────────────
with tabs[4]:
    st.header("Delta-Hedging Simulation")
    st.markdown("Simulate delta-hedging a short call and measure hedging error across rebalancing frequencies")

    col1, col2 = st.columns([1, 2])

    with col1:
        S5     = st.number_input("Spot",   value=741.75, step=1.0,  key="s5")
        K5     = st.number_input("Strike", value=741.75, step=1.0,  key="k5")
        T5     = st.number_input("Expiry (years)", value=0.50, step=0.01, key="t5")
        r5     = st.number_input("Rate",   value=0.03, step=0.001, key="r5")
        sig5   = st.number_input("Vol",    value=0.17, step=0.01,  key="sig5")
        npaths = st.number_input("Simulated Paths", value=500, min_value=100,
                                 max_value=2000, step=100, key="np5")

    with col2:
        if st.button("Run Hedging Simulation"):
            from src.hedging import DeltaHedgingSimulator

            with st.spinner("Simulating... this takes ~30 seconds"):
                results_dict = {}
                freq_map = {"Daily": 1/252, "Weekly": 5/252, "Monthly": 21/252}

                for freq_name, dt_hedge in freq_map.items():
                    hedger = DeltaHedgingSimulator(
                        BlackScholes(S5, K5, T5, r5, sig5, q=0.0),
                        n_paths=int(npaths),
                        transaction_cost_bps=5
                    )
                    pnl = hedger.run(rebalance_step=dt_hedge)
                    results_dict[freq_name] = pnl

            freq_labels = list(results_dict.keys())
            stds  = [results_dict[f].std()  for f in freq_labels]
            means = [results_dict[f].mean() for f in freq_labels]

            m1, m2, m3 = st.columns(3)
            m1.metric("Daily Std P&L",   f"${stds[0]:.2f}")
            m2.metric("Weekly Std P&L",  f"${stds[1]:.2f}")
            m3.metric("Monthly Std P&L", f"${stds[2]:.2f}")

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            colors = ['steelblue', 'darkorange', 'green']
            for i, (freq, color) in enumerate(zip(freq_labels, colors)):
                pnl = results_dict[freq]
                axes[i].hist(pnl, bins=40, color=color, alpha=0.8, edgecolor='white')
                axes[i].axvline(pnl.mean(), color='red', linestyle='--', lw=2,
                                label=f'Mean={pnl.mean():.2f}')
                axes[i].set_title(f'{freq} Rebalancing')
                axes[i].set_xlabel('P&L')
                axes[i].set_ylabel('Frequency')
                axes[i].legend(fontsize=9)
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
            **Key insight:** Monthly rebalancing has **{stds[2]/stds[0]:.0f}× wider** P&L dispersion 
            than daily — this is Gamma risk accumulating between hedges.
            """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with Black-Scholes, Monte Carlo, GBM, and Heston models | Data: yfinance")
# Derivatives Risk Engine
### Options Pricing, Greeks, Volatility Surface & Risk Management | Python

---

## Overview

An end-to-end derivatives pricing and risk management system built from scratch in Python, designed to mirror the analytical infrastructure of a sell-side options desk. The project covers the full workflow from stochastic price simulation through exotic options pricing, Greeks computation, implied volatility surface construction from live market data, portfolio-level risk aggregation, delta-hedging simulation, and historical stress testing — deployed as an interactive Streamlit dashboard.

---

## Key Results

| Module | Metric | Value |
|--------|--------|-------|
| GBM Simulation | Historical SPY kurtosis vs GBM | 11.04 vs 2.97 |
| Monte Carlo | Pricing error vs Black-Scholes at 100k paths | 0.19% |
| Asian Options | Price discount vs vanilla (same strike) | 55.9% cheaper |
| Barrier Parity | Knock-out + Knock-in = Vanilla | ✅ Holds exactly |
| IV Surface | ATM implied vol (SPY, June 2026) | 10.6% |
| IV Surface | OTM put vol premium over ATM | +11.6 vol points |
| Delta Hedging | Std P&L: Daily vs Monthly rebalancing | $2.67 vs $11.89 (346% wider) |
| Stress Test | COVID Crash P&L (-30% spot, +200% vol) | -$1,100 |
| Stress Test | Soft Landing P&L (+20% spot, -30% vol) | +$1,106 |

---

## Modules

### 1. Geometric Brownian Motion Simulation
Implemented a flexible GBM simulator using the exact discretization scheme:

S(t+dt) = S(t) × exp((μ - 0.5σ²)dt + σ√dt × Z)

Simulated 10,000 paths for SPY with historically estimated parameters (μ = 12.5%, σ = 17.1%). Empirically validated where GBM breaks down by comparing simulated and historical return distributions: historical SPY returns exhibit kurtosis of 11.04 versus GBM's theoretical 3.0 — nearly 4× the tail risk. QQ plot confirms systematic deviation at both tails, motivating the Heston stochastic volatility extension.

### 2. Monte Carlo Option Pricing
Built a Monte Carlo pricer for European, Asian, and Barrier options under the risk-neutral measure. Key implementation detail: paths simulated with drift μ = r - q (risk-free rate minus dividend yield), not historical drift — ensuring consistency with Black-Scholes pricing. At 100,000 simulations, MC price converges to Black-Scholes with 0.19% error.

Exotic options priced:
- **Asian calls:** Average price over path life — 55.9% cheaper than vanilla at same strike due to averaging reducing effective payoff volatility
- **Barrier knock-out/knock-in:** Parity relationship (knock-out + knock-in = vanilla) verified exactly, confirming implementation correctness

### 3. Black-Scholes Greeks
Complete analytical implementation of all first and second-order Greeks: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga. Plotted as 3D surfaces over (spot, time-to-expiry) space. Key findings:

- Gamma spikes to 0.031 near expiry and ATM — the "dangerous zone" for market makers
- Theta reaches -264 per day for near-expiry ATM options — dramatic time decay
- Vanna and Volga second-order cross-Greeks computed and visualized — used in practice for hedging volatility exposure

### 4. Implied Volatility Surface
Constructed a live implied volatility surface from real SPY options chain data via yfinance. Used Newton-Raphson root-finding to back out implied volatility from market option prices. Applied liquidity filters: minimum bid > $0.05, maximum bid-ask spread < 50%, OTM options only (puts below spot, calls above spot).

Key findings from June 2026 SPY options:
- ATM implied vol: **10.6%** — consistent with recent realized volatility
- OTM put implied vol: **22.6%** — reflecting the crash risk premium
- The volatility skew (higher IV for OTM puts) is the market's correction for GBM's failure to price tail risk

Heston stochastic volatility model calibrated to market IV surface. Calibration requires options spanning multiple maturities (1 month to 2 years) for reliable parameter identification; near-term weekly options do not provide sufficient term structure to distinguish the model's five parameters. This is a known limitation of short-dated option chains, not a limitation of the implementation.

### 5. Portfolio Risk Management
Constructed a realistic 13-position options portfolio including long calls, short calls, spreads, straddles, and a short straddle. Computed portfolio-level Greeks:

| Greek | Net Value | Interpretation |
|-------|-----------|----------------|
| Delta | +6.32 | Portfolio gains ~$6.32 per $1 rise in SPY |
| Gamma | +0.12 | Long convexity — benefits from large moves |
| Vega | -247 | Short volatility — loses when vol spikes |
| Theta | -890 | Loses ~$890/day from time decay |

### 6. Delta-Hedging Simulation
Simulated delta-hedging a short call position across 1,000 GBM paths at three rebalancing frequencies. Results demonstrate the fundamental trade-off between hedging precision and transaction costs:

| Frequency | Mean P&L | Std P&L | 95% CI |
|-----------|----------|---------|--------|
| Daily | -$1.42 | $2.67 | [-$6.11, +$2.40] |
| Weekly | -$0.80 | $5.82 | [-$11.11, +$8.53] |
| Monthly | -$0.64 | $11.89 | [-$20.56, +$18.09] |

Monthly rebalancing has 346% wider P&L dispersion than daily — directly quantifying the Gamma risk that accumulates between hedges. Daily rebalancing has the most negative mean P&L due to higher transaction costs, demonstrating that perfect hedging is not cost-free even in the GBM world.

### 7. Historical Stress Testing with Greek Attribution
Ran the portfolio through five named historical stress scenarios, decomposing P&L into Delta, Gamma, and Vega contributions:

| Scenario | Spot Move | Vol Move | Estimated P&L | Driver |
|----------|-----------|----------|---------------|--------|
| COVID Crash | -30% | +200% | -$1,100 | Vega loss dominates |
| 2008 GFC | -20% | +150% | -$879 | Vega loss dominates |
| Volmageddon | -5% | +100% | -$381 | Vega loss, small Delta |
| Dot-com Bust | -15% | +80% | -$640 | Mixed Vega/Delta |
| Soft Landing | +20% | -30% | +$1,106 | Delta gain + Vega gain |

The portfolio is short volatility — it profits when markets rise and volatility falls, and loses in crash scenarios where volatility spikes overwhelm the Gamma convexity benefit.

---

## Limitations & Honest Assessment

**GBM assumptions:** GBM assumes constant volatility and normally distributed returns. Real SPY returns have kurtosis of 11 versus GBM's theoretical 3 — fat tails and volatility clustering are not captured. All pricing and hedging results assume the GBM world.

**Delta-hedging costs:** All hedging simulations assume zero bid-ask spread and no market impact. Real hedging costs would increase the negative mean P&L at all rebalancing frequencies.

**Stress test approximation:** P&L estimates use a Taylor expansion (Delta × dS + 0.5 × Gamma × dS² + Vega × dσ). This approximation degrades for very large moves where higher-order terms matter.

**Heston calibration:** Near-term weekly options do not provide sufficient term structure to calibrate Heston's five parameters reliably. A production calibration would use options spanning 1 month to 2 years.

---

## Repository Structure

```
derivatives-risk-engine/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── gbm.py                    # GBM simulator
│   ├── black_scholes.py          # BS pricer + all Greeks
│   ├── monte_carlo.py            # MC pricer: European, Asian, Barrier
│   ├── implied_vol.py            # IV surface construction
│   ├── portfolio.py              # Portfolio Greeks aggregation
│   ├── hedging.py                # Delta-hedging simulation
│   ├── stress_testing.py         # Historical stress scenarios
│   └── utils.py                  # Shared utilities
│
├── notebooks/
│   ├── 01_gbm_simulation.ipynb
│   ├── 02_monte_carlo_pricing.ipynb
│   ├── 03_black_scholes_greeks.ipynb
│   ├── 04_implied_vol_surface.ipynb
│   ├── 05_exotic_options.ipynb
│   ├── 06_portfolio_risk.ipynb
│   ├── 07_delta_hedging.ipynb
│   ├── 08_stress_testing_greek_attribution.ipynb
│   └── 09_heston_model.ipynb
│
└── dashboard/
    └── app.py                    # Streamlit interactive dashboard
```

---

## Tech Stack

Python · NumPy · SciPy · pandas · matplotlib · seaborn · yfinance · Streamlit

---

## Background & Motivation

This project was built as part of a deliberate transition from academic quantitative research into quantitative finance, with the goal of building the analytical tools that a sell-side derivatives desk uses daily. Every module maps to a real desk function: GBM and Monte Carlo for exotic pricing, Greeks for hedging, implied vol surface for market calibration, portfolio risk for daily risk reporting, delta-hedging simulation for understanding hedging P&L, and stress testing for regulatory and internal risk management. The stress test table format and Greek attribution decomposition mirror what risk teams produce for desk heads and senior management.

---

*Data sourced from yfinance. Results are for educational and research purposes only and do not constitute investment advice.*

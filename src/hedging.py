import numpy as np
import pandas as pd
from src.black_scholes import BlackScholes


class DeltaHedgingSimulator:
    def __init__(self, option, n_paths=1000, transaction_cost_bps=5, rebalance_step=1 / 252, seed=None):
        self.option = option
        self.n_paths = int(n_paths)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.rebalance_step = float(rebalance_step)
        self.seed = seed

    def run(self):
        opt0 = self.option
        rng = np.random.default_rng(self.seed)
        n_steps = int(np.ceil(opt0.T / self.rebalance_step))
        dt = opt0.T / n_steps
        sigma = opt0.sigma
        drift = (opt0.r - opt0.q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        out = []

        for _ in range(self.n_paths):
            S = opt0.S
            opt = opt0
            cash = opt.call_price() - opt.delta("call") * S
            shares = opt.delta("call")

            for _ in range(n_steps):
                S = S * np.exp(drift + vol * rng.standard_normal())
                new_opt = BlackScholes(S, opt.K, max(opt.T - dt, 1e-12), opt.r, opt.sigma, opt.q)
                new_delta = new_opt.delta("call")
                trade = new_delta - shares
                tc = abs(trade) * S * self.transaction_cost_bps / 10000.0
                cash = cash * np.exp(opt.r * dt) - trade * S - tc
                shares = new_delta
                opt = new_opt

            payoff = max(S - opt.K, 0)
            pnl = cash + shares * S - payoff
            out.append(pnl)

        return pd.Series(out, name="hedging_pnl")
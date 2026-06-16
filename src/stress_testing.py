import pandas as pd
import numpy as np


class StressTester:
    def __init__(self, portfolio, S, sigma):
        self.portfolio = portfolio
        self.S = float(S)
        self.sigma = float(sigma)

    def scenario_pnl(self, delta_S_pct, delta_sigma_pct):
        dS = self.S * delta_S_pct
        dvol = self.sigma * delta_sigma_pct
        d = self.portfolio.net_delta()
        g = self.portfolio.net_gamma()
        v = self.portfolio.net_vega()
        pnl = d * dS + 0.5 * g * dS**2 + v * dvol
        return pnl

    def historical_scenarios(self):
        scenarios = {
            "COVID Crash": (-0.30, 2.00),
            "2008 GFC": (-0.20, 1.50),
            "Volmageddon": (-0.05, 1.00),
            "Dot-com Bust": (-0.15, 0.80),
            "Soft Landing": (0.20, -0.30),
        }
        rows = []
        for name, (s, v) in scenarios.items():
            rows.append(
                {
                    "scenario": name,
                    "spot_move_pct": s,
                    "vol_move_pct": v,
                    "estimated_pnl": self.scenario_pnl(s, v),
                }
            )
        return pd.DataFrame(rows)

    def stress_surface(self, spot_grid=np.linspace(-0.3, 0.3, 25), vol_grid=np.linspace(-0.5, 1.5, 25)):
        rows = []
        for s in spot_grid:
            for v in vol_grid:
                rows.append({"spot_move_pct": s, "vol_move_pct": v, "pnl": self.scenario_pnl(s, v)})
        return pd.DataFrame(rows)
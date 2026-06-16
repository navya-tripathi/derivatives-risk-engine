import pandas as pd
import numpy as np


class OptionsPortfolio:
    def __init__(self):
        self.positions = []

    def add_position(self, option, quantity=1.0, option_type="call", name=None):
        self.positions.append({"option": option, "quantity": float(quantity), "option_type": option_type, "name": name})

    def _agg(self, greek_name):
        total = 0.0
        for pos in self.positions:
            opt = pos["option"]
            qty = pos["quantity"]
            typ = pos["option_type"]
            val = getattr(opt, greek_name)
            g = val(typ) if greek_name in ["delta", "theta", "rho"] else val()
            total += qty * g
        return total

    def net_delta(self):
        return self._agg("delta")

    def net_gamma(self):
        return self._agg("gamma")

    def net_vega(self):
        return self._agg("vega")

    def net_theta(self):
        return self._agg("theta")

    def net_rho(self):
        return self._agg("rho")

    def dollar_greeks(self, S):
        return {
            "dollar_delta": self.net_delta() * S,
            "dollar_gamma": self.net_gamma() * 0.5 * S**2,
            "dollar_vega_per_1pct": self.net_vega() * 0.01,
        }

    def risk_report(self):
        rows = []
        for pos in self.positions:
            opt = pos["option"]
            typ = pos["option_type"]
            qty = pos["quantity"]
            rows.append(
                {
                    "name": pos["name"],
                    "quantity": qty,
                    "option_type": typ,
                    "delta": qty * opt.delta(typ),
                    "gamma": qty * opt.gamma(),
                    "vega": qty * opt.vega(),
                    "theta": qty * opt.theta(typ),
                    "rho": qty * opt.rho(typ),
                }
            )
        df = pd.DataFrame(rows)
        net = pd.DataFrame(
            [
                {
                    "name": "NET",
                    "quantity": df["quantity"].sum() if len(df) else 0,
                    "option_type": "",
                    "delta": df["delta"].sum() if len(df) else 0,
                    "gamma": df["gamma"].sum() if len(df) else 0,
                    "vega": df["vega"].sum() if len(df) else 0,
                    "theta": df["theta"].sum() if len(df) else 0,
                    "rho": df["rho"].sum() if len(df) else 0,
                }
            ]
        )
        return pd.concat([df, net], ignore_index=True)
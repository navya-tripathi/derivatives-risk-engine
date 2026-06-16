import pandas as pd
import numpy as np
import yfinance as yf
from src.utils import implied_vol, year_fraction, interpolate_surface, get_spot


class ImpliedVolSurface:
    def __init__(self, ticker, expiry_dates=None, rate=0.03):
        self.ticker = ticker
        self.expiry_dates = expiry_dates
        self.rate = rate
        self.surface = None

    def _chain_for_expiry(self, expiry):
        t = yf.Ticker(self.ticker)
        chain = t.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        calls["option_type"] = "call"
        puts["option_type"] = "put"
        return pd.concat([calls, puts], ignore_index=True)

    def compute_iv(self, market_price, S, K, T, r, option_type):
        return implied_vol(market_price, S, K, T, r, option_type=option_type)

    def build_surface(self):
        S = get_spot(self.ticker)
        rows = []
        expiries = self.expiry_dates or list(yf.Ticker(self.ticker).options[:4])
        for expiry in expiries:
            chain = self._chain_for_expiry(expiry)
            T = year_fraction(pd.Timestamp.today(), expiry)
            for _, row in chain.iterrows():
                price = row.get("lastPrice", np.nan)
                bid = row.get("bid", np.nan)
                ask = row.get("ask", np.nan)
                if (not np.isfinite(price) or price <= 0) and np.isfinite(bid) and np.isfinite(ask) and ask > bid:
                    price = 0.5 * (bid + ask)
                if not np.isfinite(price) or price <= 0:
                    continue
                iv = self.compute_iv(price, S, row["strike"], T, self.rate, row["option_type"])
                rows.append(
                    {
                        "expiry": expiry,
                        "ttm": T,
                        "strike": row["strike"],
                        "moneyness": row["strike"] / S if S else np.nan,
                        "iv": iv,
                        "option_type": row["option_type"],
                        "lastPrice": price,
                        "bid": row.get("bid", np.nan),
                        "ask": row.get("ask", np.nan),
                        "volume": row.get("volume", 0),
                    }
                )
        self.surface = pd.DataFrame(rows)
        return self.surface

    def plot_smile_data(self, expiry):
        if self.surface is None:
            self.build_surface()
        return self.surface[self.surface["expiry"] == expiry].copy()

    def plot_surface_data(self):
        if self.surface is None:
            self.build_surface()
        return interpolate_surface(self.surface)
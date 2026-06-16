import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
import yfinance as yf

TRADING_DAYS = 252


def year_fraction(start, end, basis=TRADING_DAYS):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return max((end - start).days / basis, 0.0)


def annualize_vol(returns, basis=TRADING_DAYS):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(basis)


def safe_div(a, b):
    return np.where(np.abs(b) > 1e-12, a / b, np.nan)


def get_spot(ticker):
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period="5d", auto_adjust=False)
        if len(hist):
            for col in ["Close", "Adj Close"]:
                if col in hist.columns:
                    return float(hist[col].dropna().iloc[-1])
    except Exception:
        pass
    info = getattr(t, "fast_info", {}) or {}
    for k in ["lastPrice", "regularMarketPrice"]:
        if k in info and info[k] is not None:
            return float(info[k])
    return np.nan


def get_hist_prices(ticker, period="2y"):
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            s = df.xs("Close", axis=1, level=-1).iloc[:, 0]
    else:
        s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return s.dropna().rename("price")


def get_risk_free_rate(default=0.03):
    return default


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, q=0.0, option_type="call", bounds=(1e-6, 5.0)):
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    def objective(sig):
        return (bs_price(S, K, T, r, sig, q=q, option_type=option_type) - price) ** 2

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(res.x) if res.success else np.nan


def interpolate_surface(df, x_col="moneyness", y_col="ttm", z_col="iv", grid_size=40):
    d = df[[x_col, y_col, z_col]].dropna().copy()
    if d.empty:
        return None, None, None
    xi = np.linspace(d[x_col].min(), d[x_col].max(), grid_size)
    yi = np.linspace(d[y_col].min(), d[y_col].max(), grid_size)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((d[x_col], d[y_col]), d[z_col], (X, Y), method="linear")
    return X, Y, Z
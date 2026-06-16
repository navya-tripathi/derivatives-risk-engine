import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)

    def d1(self):
        if self.T <= 0 or self.sigma <= 0:
            return np.nan
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        d1, d2 = self.d1(), self.d2()
        return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self):
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        d1, d2 = self.d1(), self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

    def delta(self, option_type="call"):
        d1 = self.d1()
        if option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)

    def gamma(self):
        d1 = self.d1()
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1 = self.d1()
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self, option_type="call"):
        d1, d2 = self.d1(), self.d2()
        first = -(self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
        if option_type == "call":
            return first + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return first - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

    def rho(self, option_type="call"):
        d2 = self.d2()
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

    def vanna(self):
        return self.vega() * (1 - self.d1() / (self.sigma * np.sqrt(self.T))) / self.S

    def volga(self):
        d1, d2 = self.d1(), self.d2()
        return self.vega() * d1 * d2 / self.sigma

    def greeks(self):
        return {
            "delta_call": self.delta("call"),
            "delta_put": self.delta("put"),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta_call": self.theta("call"),
            "theta_put": self.theta("put"),
            "rho_call": self.rho("call"),
            "rho_put": self.rho("put"),
            "vanna": self.vanna(),
            "volga": self.volga(),
        }
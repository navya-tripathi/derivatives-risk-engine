import numpy as np
from src.black_scholes import BlackScholes


class MonteCarloPricer:
    def __init__(self, simulator, r, q=0.0):
        self.simulator = simulator
        self.r = float(r)
        self.q = float(q)

    def _discount(self, payoff):
        return np.exp(-self.r * self.simulator.T) * np.asarray(payoff)

    def price_european_call(self, K):
        S_T = self.simulator.terminal_distribution()
        return float(self._discount(np.maximum(S_T - K, 0)).mean())

    def price_european_put(self, K):
        S_T = self.simulator.terminal_distribution()
        return float(self._discount(np.maximum(K - S_T, 0)).mean())

    def price_asian_call(self, K):
        if self.simulator.paths is None:
            self.simulator.simulate()
        avg = self.simulator.paths.mean(axis=1)
        return float(np.exp(-self.r * self.simulator.T) * np.maximum(avg - K, 0).mean())

    def price_barrier_call(self, K, barrier, barrier_type="knock_out"):
        if self.simulator.paths is None:
            self.simulator.simulate()
        hit = self.simulator.paths.max(axis=1) >= barrier
        payoff = np.maximum(self.simulator.paths[:, -1] - K, 0)
        if barrier_type == "knock_out":
            payoff = np.where(hit, 0.0, payoff)
        else:
            payoff = np.where(hit, payoff, 0.0)
        return float(np.exp(-self.r * self.simulator.T) * payoff.mean())

    def confidence_interval(self, K, option_type="call", n_bootstrap=1000, alpha=0.05):
        rng = np.random.default_rng(42)
        S_T = self.simulator.terminal_distribution()
        payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
        disc = np.exp(-self.r * self.simulator.T) * payoff
        boot = [rng.choice(disc, size=len(disc), replace=True).mean() for _ in range(n_bootstrap)]
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    def bs_reference(self, S, K, T, sigma, option_type="call"):
        bs = BlackScholes(S, K, T, self.r, sigma, q=self.q)
        return bs.call_price() if option_type == "call" else bs.put_price()
import numpy as np
import matplotlib.pyplot as plt


class GBMSimulator:
    def __init__(self, S0, mu, sigma, T, dt, n_paths):
        # S0: initial price
        # mu: drift (annualized)
        # sigma: volatility (annualized)
        # T: time horizon in years
        # dt: time step (1/252 for daily)
        # n_paths: number of simulation paths
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.T = float(T)
        self.dt = float(dt)
        self.n_paths = int(n_paths)
        self.paths = None

    def simulate(self):
        # Returns array of shape (n_paths, n_steps)
        # Uses exact GBM discretization:
        # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # where Z ~ N(0,1)
        n_steps = int(np.ceil(self.T / self.dt)) + 1
        paths = np.zeros((self.n_paths, n_steps))
        paths[:, 0] = self.S0
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)
        Z = np.random.standard_normal((self.n_paths, n_steps - 1))
        for t in range(1, n_steps):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * Z[:, t - 1])
        self.paths = paths
        return paths

    def terminal_distribution(self):
        # Returns terminal prices S(T) across all paths
        if self.paths is None:
            self.simulate()
        return self.paths[:, -1]

    def plot_paths(self, n_display=50, band=0.95):
        # Plots sample paths with mean and confidence bands
        if self.paths is None:
            self.simulate()

        lower_q = (1 - band) / 2
        upper_q = 1 - lower_q
        mean_path = self.paths.mean(axis=0)
        lower = np.quantile(self.paths, lower_q, axis=0)
        upper = np.quantile(self.paths, upper_q, axis=0)

        plt.figure(figsize=(10, 6))
        for i in range(min(n_display, self.n_paths)):
            plt.plot(self.paths[i], color="steelblue", alpha=0.15, linewidth=0.8)

        plt.plot(mean_path, color="black", linewidth=2, label="Mean")
        plt.fill_between(
            range(len(mean_path)), lower, upper, color="gray", alpha=0.25, label=f"{int(band*100)}% band"
        )
        plt.title("GBM Simulated Paths")
        plt.xlabel("Step")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
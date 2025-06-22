from datetime import date
from typing import Dict, List

import cvxpy as cp
import numpy as np
from loguru import logger


# calculate asset allocation
class PortfolioOptimizer:
    def __init__(
        self,
        action_date: Dict[str, date],
        position: Dict[str, int],
        trading_price_history: Dict[str, List[float]],
        buying_power: float,
    ):
        self.action_date = action_date
        self.position = position
        self.trading_price = trading_price_history
        self.buying_power = buying_power
        self.returns = self._calculate_returns()

    # * pass: calculate the return series
    def _calculate_returns(self):
        returns = {}
        for symbol, prices in self.trading_price.items():
            if len(self.trading_price[symbol]) >= 7:
                symbol_returns = np.diff(prices) / prices[:-1]
                returns[symbol] = np.array(symbol_returns)
            else:
                logger.warning(
                    "Portfolio Optimization not enough data to calculate returns."
                )
        return returns

    def _shrinkage_estimates(self):
        n = len(self.returns)
        rets = np.vstack(
            [
                self.returns[symbol]
                for symbol in self.returns
                if self.returns[symbol] is not None
            ]
        )
        # Calculate sample covariance and mean
        sample_cov = np.cov(rets)
        mean_rets = np.mean(rets, axis=1)

        # Shrinkage target (e.g., scaled identity matrix)
        avg_var = np.trace(sample_cov) / n
        target = np.eye(n) * avg_var

        # Shrinkage intensity parameter (example calculation, can be optimized)
        beta = 0.1

        # Shrinkage estimator for covariance
        shrunk_cov = beta * target + (1 - beta) * sample_cov

        # Shrinkage for means (shrink towards overall mean)
        overall_mean = np.mean(mean_rets)
        shrunk_means = beta * overall_mean + (1 - beta) * mean_rets

        return shrunk_means, shrunk_cov

    # * pass: Markowitz portfolio optimization
    def _optimize_weights(self):
        if not self.returns:
            return
        n = len(self.returns)

        # Optimization variables
        w = cp.Variable(n)

        mean_returns, cov_matrix = self._shrinkage_estimates()
        # Define the objective
        risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(mean_returns.T @ w - risk)

        # Constraints
        constraints = []  # [cp.sum(w) == 1]
        for i, symbol in enumerate(self.returns):
            position = self.position.get(symbol, 0)
            if position == 1:  # type: ignore
                constraints.extend((w[i] >= 0, w[i] <= 1))
            elif position == -1:  # type: ignore
                constraints.extend((w[i] <= 0, w[i] >= -1))
            elif position == 0:
                constraints.append(w[i] == 0)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return w.value

    # * pass: Markowitz portfolio optimization + make small val to zero
    def _process_weights(self):
        if self.returns:
            threshold = 1e-7
            weights = self._optimize_weights()
            weights = np.where(abs(weights) < threshold, 0, weights)  # type: ignore
            return weights

    def calculate_weights(self):
        weight_dict = {}
        if self.returns:
            weights = self._process_weights()
            symbols = list(self.returns.keys())
            for i, symbol in enumerate(symbols):
                weight_rounded = np.round(weights[i], 4)  # type: ignore
                weight_dict[symbol] = weight_rounded.item()
        else:
            all_symbols = list(self.trading_price.keys())
            for symbol in all_symbols:
                weight_dict[symbol] = 0.0
            logger.warning(
                "Portfolio Optimization no symbols returns, do not have weights and shares."
            )

        return weight_dict

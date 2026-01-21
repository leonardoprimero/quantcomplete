"""
Monte Carlo Portfolio Simulator
Performs random portfolio simulations for efficient frontier analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class MonteCarloSimulator:
    """Performs Monte Carlo portfolio simulation."""
    
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, num_portfolios: int = 50000):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.num_portfolios = num_portfolios
        self.results = None
        self.weights_record = None
        
    def run_simulation(self, seed: int = 42) -> None:
        """Run Monte Carlo simulation for random portfolios."""
        np.random.seed(seed)
        num_assets = len(self.mean_returns)
        
        self.results = np.zeros((3, self.num_portfolios))
        self.weights_record = []
        
        for i in range(self.num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            self.weights_record.append(weights)
            
            portfolio_return, portfolio_volatility = self._calculate_portfolio_stats(weights)
            sharpe_ratio = portfolio_return / portfolio_volatility
            
            self.results[0, i] = portfolio_return
            self.results[1, i] = portfolio_volatility
            self.results[2, i] = sharpe_ratio
            
        self.weights_record = np.array(self.weights_record)
        
    def _calculate_portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio return and volatility."""
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
        
    def get_optimal_portfolios(self) -> dict:
        """Find max Sharpe and min volatility portfolios from simulation."""
        max_sharpe_idx = np.argmax(self.results[2])
        min_vol_idx = np.argmin(self.results[1])
        
        return {
            'max_sharpe': {
                'return': self.results[0, max_sharpe_idx],
                'volatility': self.results[1, max_sharpe_idx],
                'sharpe': self.results[2, max_sharpe_idx],
                'weights': self.weights_record[max_sharpe_idx]
            },
            'min_vol': {
                'return': self.results[0, min_vol_idx],
                'volatility': self.results[1, min_vol_idx],
                'sharpe': self.results[2, min_vol_idx],
                'weights': self.weights_record[min_vol_idx]
            }
        }

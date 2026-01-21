"""
Markowitz Portfolio Optimizer
Performs mathematical optimization for efficient frontier.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple


class MarkowitzOptimizer:
    """Performs Markowitz portfolio optimization."""
    
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.num_assets = len(mean_returns)
        
    def _portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio return and volatility."""
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
        
    def _neg_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        ret, vol = self._portfolio_stats(weights)
        return -ret / vol
        
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Portfolio volatility for minimization."""
        return self._portfolio_stats(weights)[1]
        
    def optimize_max_sharpe(self) -> dict:
        """Optimize for maximum Sharpe ratio."""
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets]
        
        result = minimize(self._neg_sharpe, initial_guess, 
                         args=(), method='SLSQP', bounds=bounds, constraints=constraints)
        
        ret, vol = self._portfolio_stats(result.x)
        return {
            'return': ret,
            'volatility': vol,
            'sharpe': ret / vol,
            'weights': result.x
        }
        
    def optimize_min_volatility(self) -> dict:
        """Optimize for minimum volatility."""
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets]
        
        result = minimize(self._portfolio_volatility, initial_guess,
                         args=(), method='SLSQP', bounds=bounds, constraints=constraints)
        
        ret, vol = self._portfolio_stats(result.x)
        return {
            'return': ret,
            'volatility': vol,
            'sharpe': ret / vol,
            'weights': result.x
        }
        
    def generate_efficient_frontier(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate efficient frontier curve."""
        min_vol_portfolio = self.optimize_min_volatility()
        target_returns = np.linspace(min_vol_portfolio['return'], self.mean_returns.max(), num_points)
        efficient_vols = []
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = self.num_assets * [1. / self.num_assets]
        
        for target_ret in target_returns:
            constraints_ef = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, tr=target_ret: self._portfolio_stats(x)[0] - tr}
            ]
            result = minimize(self._portfolio_volatility, initial_guess,
                            args=(), method='SLSQP', bounds=bounds, constraints=constraints_ef)
            
            if result.success:
                efficient_vols.append(result.fun)
            else:
                efficient_vols.append(np.nan)
                
        return np.array(efficient_vols), target_returns

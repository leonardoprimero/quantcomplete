"""
Portfolio Data Handler
Manages financial data download and preprocessing.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List


class PortfolioData:
    """Handles financial data download and preprocessing."""
    
    def __init__(self, tickers: List[str], years: int = 3, start_date: str = None, end_date: str = None):
        self.tickers = tickers
        self.years = years
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.log_returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def download_data(self) -> None:
        """Download historical price data from Yahoo Finance."""
        if self.end_date:
            end = self.end_date
        else:
            end = datetime.now()
            
        if self.start_date:
            start = self.start_date
        else:
            start = end - timedelta(days=self.years * 365)
        
        raw_data = yf.download(self.tickers, start=start, end=end, auto_adjust=False)
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            self.data = raw_data['Adj Close']
        else:
            self.data = raw_data
            
    def calculate_returns(self, trading_days: int = 252) -> None:
        """Calculate annualized returns and covariance matrix."""
        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()
        self.mean_returns = self.log_returns.mean() * trading_days
        self.cov_matrix = self.log_returns.cov() * trading_days
        
    def get_individual_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate individual asset volatility and Sharpe ratios."""
        individual_vols = np.sqrt(np.diag(self.cov_matrix))
        individual_sharpe = self.mean_returns / individual_vols
        return individual_vols, individual_sharpe

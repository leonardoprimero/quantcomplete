"""
Results Printer
Handles console output formatting for portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import List


class ResultsPrinter:
    """Handles console output formatting."""
    
    @staticmethod
    def print_header(tickers: List[str], num_portfolios: int) -> None:
        """Print analysis header."""
        print("\n" + "=" * 70)
        print("EFFICIENT FRONTIER ANALYSIS - MONTE CARLO + MARKOWITZ OPTIMIZATION")
        print("=" * 70)
        print(f"\nTickers: {', '.join(tickers)}")
        print(f"Monte Carlo Simulations: {num_portfolios:,}")
        
    @staticmethod
    def print_section(title: str) -> None:
        """Print section header."""
        print(f"\n{title}")
        print("-" * 80)
        print()
        
    @staticmethod
    def print_individual_assets(tickers: List[str], mean_returns: pd.Series, 
                               individual_vols: np.ndarray, individual_sharpe: np.ndarray) -> None:
        """Print individual asset statistics."""
        print("\n" + "-" * 70)
        print("INDIVIDUAL ASSETS PERFORMANCE")
        print("-" * 70)
        for i, ticker in enumerate(tickers):
            print(f"{ticker:6s} | Return: {mean_returns.iloc[i]:7.4f} ({mean_returns.iloc[i]*100:6.2f}%) | "
                  f"Vol: {individual_vols[i]:.4f} ({individual_vols[i]*100:5.2f}%) | "
                  f"Sharpe: {individual_sharpe[i]:.4f}")
            
    @staticmethod
    def print_portfolio(title: str, portfolio: dict, tickers: List[str]) -> None:
        """Print portfolio details."""
        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)
        print(f"Expected Annual Return: {portfolio['return']:.4f} ({portfolio['return']*100:.2f}%)")
        print(f"Annual Volatility: {portfolio['volatility']:.4f} ({portfolio['volatility']*100:.2f}%)")
        print(f"Sharpe Ratio: {portfolio['sharpe']:.4f}")
        print("\nWeights:")
        for ticker, weight in zip(tickers, portfolio['weights']):
            print(f"  {ticker}: {weight:.4f}")
            
    @staticmethod
    def print_footer() -> None:
        """Print footer."""
        print("=" * 70 + "\n")

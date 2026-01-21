"""
Efficient Frontier Visualizer
Handles all visualization and plotting for portfolio analysis.
Uses institutional color palette and adjustText for professional presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, List
from adjustText import adjust_text
from quant.visualization.styles import InstitutionalColors as IC


class EfficientFrontierVisualizer:
    """Handles visualization of efficient frontier analysis."""
    
    def __init__(self, tickers: List[str], mc_results: np.ndarray, 
                 efficient_frontier: Tuple[np.ndarray, np.ndarray],
                 individual_stats: Tuple[np.ndarray, np.ndarray, pd.Series],
                 markowitz_portfolios: dict, mc_portfolios: dict):
        self.tickers = tickers
        self.mc_results = mc_results
        self.efficient_vols, self.efficient_returns = efficient_frontier
        self.individual_vols, self.individual_sharpe, self.mean_returns = individual_stats
        self.markowitz = markowitz_portfolios
        self.mc = mc_portfolios
        
    def create_plot(self, save_path: str, num_portfolios: int) -> None:
        """Create and save the efficient frontier visualization."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        self._plot_monte_carlo(ax)
        self._plot_efficient_frontier(ax)
        self._plot_capital_market_line(ax)
        self._plot_individual_assets(ax)
        self._plot_optimal_portfolios(ax)
        self._add_annotations(ax)
        self._format_plot(ax, num_portfolios)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced chart saved to: {save_path}")
        plt.show()
        
    def _plot_monte_carlo(self, ax) -> None:
        """Plot Monte Carlo simulation scatter."""
        # Use institutional color gradient
        scatter = ax.scatter(self.mc_results[1, :], self.mc_results[0, :], 
                           c=self.mc_results[2, :], cmap='RdYlGn',  # Red-Yellow-Green for Sharpe
                           marker='o', s=8, alpha=0.25, edgecolors='none', 
                           label='Monte Carlo Portfolios')
        cbar = plt.colorbar(scatter, ax=ax, label='Sharpe Ratio', pad=0.02)
        cbar.ax.tick_params(labelsize=10, colors=IC.DARK_GRAY)
        
    def _plot_efficient_frontier(self, ax) -> None:
        """Plot Markowitz efficient frontier."""
        valid_mask = ~np.isnan(self.efficient_vols)
        ax.plot(self.efficient_vols[valid_mask], self.efficient_returns[valid_mask],
               color=IC.DANGER_RED, linewidth=3, label='Markowitz Efficient Frontier', zorder=10)
        
    def _plot_capital_market_line(self, ax) -> None:
        """Plot Capital Market Line."""
        max_sharpe = self.markowitz['max_sharpe']
        max_vol_for_cml = max_sharpe['volatility'] * 1.5
        cml_x = np.linspace(0, max_vol_for_cml, 100)
        cml_y = (max_sharpe['return'] / max_sharpe['volatility']) * cml_x
        ax.plot(cml_x, cml_y, color=IC.WARNING_ORANGE, linewidth=2, linestyle='--',
               label='Capital Market Line', zorder=9, alpha=0.8)
        
    def _plot_individual_assets(self, ax) -> None:
        """Plot individual assets with adjustText to prevent overlap."""
        colors = IC.get_color_palette()
        texts = []
        
        for i, ticker in enumerate(self.tickers):
            # Use modulo to cycle through colors if more tickers than colors
            color_idx = i % len(colors)
            ax.scatter(self.individual_vols[i], self.mean_returns.iloc[i], 
                      marker='D', s=200, color=colors[color_idx], edgecolors=IC.DARK_GRAY, 
                      linewidths=2, zorder=11, label=f'{ticker}')
            
            # Collect text objects for adjustText
            text = ax.text(self.individual_vols[i], self.mean_returns.iloc[i], ticker,
                          fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=IC.DARK_GRAY, alpha=0.9))
            texts.append(text)
        
        # Use adjustText to prevent overlap
        adjust_text(texts, ax=ax, 
                   arrowprops=dict(arrowstyle='->', color=IC.MEDIUM_GRAY, lw=0.5),
                   expand_points=(1.2, 1.2), expand_text=(1.2, 1.2))
        
    def _plot_optimal_portfolios(self, ax) -> None:
        """Plot optimal portfolios."""
        max_sharpe = self.markowitz['max_sharpe']
        min_vol = self.markowitz['min_vol']
        
        ax.scatter(max_sharpe['volatility'], max_sharpe['return'], 
                  marker='*', color=IC.GOLD, s=800, edgecolors=IC.DARK_GRAY, 
                  linewidths=2.5, label='Markowitz Max Sharpe', zorder=12)
        ax.scatter(min_vol['volatility'], min_vol['return'],
                  marker='*', color=IC.PRIMARY_BLUE, s=800, edgecolors=IC.DARK_GRAY,
                  linewidths=2.5, label='Markowitz Min Volatility', zorder=12)
        
    def _add_annotations(self, ax) -> None:
        """Add annotations for optimal portfolios."""
        max_sharpe = self.markowitz['max_sharpe']
        min_vol = self.markowitz['min_vol']
        
        ax.annotate(f"Max Sharpe\nReturn: {max_sharpe['return']*100:.1f}%\n"
                   f"Vol: {max_sharpe['volatility']*100:.1f}%\nSR: {max_sharpe['sharpe']:.2f}",
                   xy=(max_sharpe['volatility'], max_sharpe['return']),
                   xytext=(20, -40), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=IC.GOLD, alpha=0.8,
                            edgecolor=IC.DARK_GRAY),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                  lw=1.5, color=IC.DARK_GRAY))
        
        ax.annotate(f"Min Vol\nReturn: {min_vol['return']*100:.1f}%\n"
                   f"Vol: {min_vol['volatility']*100:.1f}%\nSR: {min_vol['sharpe']:.2f}",
                   xy=(min_vol['volatility'], min_vol['return']),
                   xytext=(-120, 30), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=IC.PRIMARY_BLUE, alpha=0.8,
                            edgecolor=IC.DARK_GRAY),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                                  lw=1.5, color=IC.DARK_GRAY))
        
    def _format_plot(self, ax, num_portfolios: int) -> None:
        """Format plot labels, title, and legend with institutional styling."""
        ax.set_xlabel('Annual Volatility (Standard Deviation)', fontsize=13, 
                     fontweight='bold', color=IC.DARK_GRAY)
        ax.set_ylabel('Expected Annual Return', fontsize=13, 
                     fontweight='bold', color=IC.DARK_GRAY)
        ax.set_title('Efficient Frontier: Monte Carlo Simulation + Markowitz Optimization\n' +
                    f'{", ".join(self.tickers)} | {num_portfolios:,} Simulated Portfolios | 3-Year Historical Data',
                    fontsize=15, fontweight='bold', pad=20, color=IC.PRIMARY_BLUE)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        
        # Apply institutional styling
        IC.apply_institutional_style(ax)
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2,
                 edgecolor=IC.DARK_GRAY, fancybox=True)
        
        # Add info box with institutional styling
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        info_text = (f'Risk-Free Rate: 0%\n'
                    f'Trading Days: 252\n'
                    f'Data Period: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=IC.LIGHT_GRAY, alpha=0.9,
                        edgecolor=IC.DARK_GRAY))

"""
Portfolio Backtester
Performs historical backtesting of portfolio strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class PortfolioBacktester:
    """Performs backtesting analysis on portfolio strategies."""
    
    def __init__(self, data: pd.DataFrame, returns: pd.DataFrame):
        self.data = data
        self.returns = returns
        
    def backtest_portfolio(self, weights: np.ndarray, tickers: list, 
                          initial_capital: float = 10000) -> dict:
        """Backtest a portfolio with given weights."""
        # Calculate portfolio returns
        portfolio_returns = (self.returns[tickers] * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # Calculate metrics
        total_return = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
        annual_return = portfolio_returns.mean() * 252 * 100
        annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
        
        # Calculate drawdown
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (portfolio_returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = (portfolio_returns.mean() * 252) / abs(drawdown.min()) if drawdown.min() != 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'cumulative_returns': cumulative_returns,
            'portfolio_returns': portfolio_returns,
            'drawdown': drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'initial_capital': initial_capital,
            'final_capital': portfolio_value.iloc[-1]
        }
        
    def create_backtest_chart(self, backtest_results: dict, benchmark_ticker: str,
                             save_path: str, portfolio_name: str = "Portfolio Óptimo"):
        """Create comprehensive backtest visualization."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio value evolution
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_portfolio_value(ax1, backtest_results, benchmark_ticker)
        
        # 2. Cumulative returns comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_cumulative_returns(ax2, backtest_results, benchmark_ticker)
        
        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown(ax3, backtest_results)
        
        # 4. Monthly returns
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_monthly_returns(ax4, backtest_results)
        
        # 5. Performance metrics table
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_metrics_table(ax5, backtest_results, benchmark_ticker)
        
        plt.suptitle(f'Backtesting: {portfolio_name}', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_portfolio_value(self, ax, results, benchmark_ticker):
        """Plot portfolio value evolution."""
        portfolio_value = results['portfolio_value']
        
        # Benchmark
        bench_returns = self.returns[benchmark_ticker]
        bench_value = results['initial_capital'] * (1 + bench_returns).cumprod()
        
        ax.plot(portfolio_value.index, portfolio_value, label='Portfolio', 
               linewidth=2, color='#2c5aa0')
        ax.plot(bench_value.index, bench_value, label=f'{benchmark_ticker} (Benchmark)', 
               linewidth=2, linestyle='--', alpha=0.7, color='orange')
        ax.axhline(y=results['initial_capital'], color='gray', linestyle=':', 
                  label='Capital Inicial', alpha=0.5)
        
        ax.set_title('Evolución del Valor del Portfolio', fontweight='bold', fontsize=12)
        ax.set_ylabel('Valor ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
    def _plot_cumulative_returns(self, ax, results, benchmark_ticker):
        """Plot cumulative returns."""
        cum_returns = results['cumulative_returns']
        bench_cum = (1 + self.returns[benchmark_ticker]).cumprod()
        
        ax.plot(cum_returns.index, (cum_returns - 1) * 100, label='Portfolio', linewidth=2)
        ax.plot(bench_cum.index, (bench_cum - 1) * 100, label=benchmark_ticker, 
               linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_title('Retornos Acumulados', fontweight='bold', fontsize=12)
        ax.set_ylabel('Retorno (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
        
    def _plot_drawdown(self, ax, results):
        """Plot drawdown."""
        drawdown = results['drawdown'] * 100
        
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
        
        ax.set_title(f'Drawdown (Máx: {results["max_drawdown"]:.2f}%)', 
                    fontweight='bold', fontsize=12)
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
    def _plot_monthly_returns(self, ax, results):
        """Plot monthly returns distribution."""
        portfolio_returns = results['portfolio_returns']
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        ax.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
        
        ax.set_title('Retornos Mensuales', fontweight='bold', fontsize=12)
        ax.set_ylabel('Retorno (%)')
        ax.set_xlabel('Mes')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_metrics_table(self, ax, results, benchmark_ticker):
        """Plot performance metrics table."""
        # Calculate benchmark metrics
        bench_returns = self.returns[benchmark_ticker]
        bench_annual_ret = bench_returns.mean() * 252 * 100
        bench_annual_vol = bench_returns.std() * np.sqrt(252) * 100
        bench_sharpe = (bench_returns.mean() * 252) / (bench_returns.std() * np.sqrt(252))
        
        metrics_data = [
            ['Métrica', 'Portfolio', benchmark_ticker],
            ['Retorno Total', f"{results['total_return']:.2f}%", 
             f"{((1 + bench_returns).prod() - 1) * 100:.2f}%"],
            ['Retorno Anual', f"{results['annual_return']:.2f}%", f"{bench_annual_ret:.2f}%"],
            ['Volatilidad Anual', f"{results['annual_volatility']:.2f}%", f"{bench_annual_vol:.2f}%"],
            ['Sharpe Ratio', f"{results['sharpe_ratio']:.3f}", f"{bench_sharpe:.3f}"],
            ['Sortino Ratio', f"{results['sortino_ratio']:.3f}", '-'],
            ['Calmar Ratio', f"{results['calmar_ratio']:.3f}", '-'],
            ['Máx Drawdown', f"{results['max_drawdown']:.2f}%", '-'],
            ['Win Rate', f"{results['win_rate']:.2f}%", '-'],
            ['Capital Final', f"${results['final_capital']:,.0f}", '-']
        ]
        
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                        colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#2c5aa0')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metrics_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Métricas de Performance', fontweight='bold', fontsize=12, pad=20)

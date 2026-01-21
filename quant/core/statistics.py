"""
Asset Analyzer
Performs detailed quantitative analysis for individual assets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import seaborn as sns


class AssetAnalyzer:
    """Performs comprehensive analysis on individual assets."""
    
    def __init__(self, data: pd.DataFrame, tickers: list, benchmark_ticker: str = 'SPY'):
        self.data = data
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.returns = np.log(data / data.shift(1)).dropna()
        
    def create_individual_analysis_charts(self, save_dir: str) -> dict:
        """Create comprehensive charts for each asset."""
        chart_paths = {}
        
        for ticker in self.tickers:
            if ticker == self.benchmark_ticker:
                continue
                
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            
            # 1. Price evolution
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_price_evolution(ax1, ticker)
            
            # 2. Returns distribution (Gaussian bell curve)
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_returns_distribution(ax2, ticker)
            
            # 3. QQ Plot (normality test)
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_qq(ax3, ticker)
            
            # 4. Rolling volatility
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_rolling_volatility(ax4, ticker)
            
            # 5. Cumulative returns
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_cumulative_returns(ax5, ticker)
            
            # 6. Drawdown
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_drawdown(ax6, ticker)
            
            # 7. Monthly returns heatmap
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_monthly_heatmap(ax7, ticker)
            
            # 8. Correlation with benchmark
            ax8 = fig.add_subplot(gs[3, 0])
            self._plot_benchmark_correlation(ax8, ticker)
            
            # 9. Volume analysis
            ax9 = fig.add_subplot(gs[3, 1])
            self._plot_volume(ax9, ticker)
            
            # 10. Risk-Return scatter
            ax10 = fig.add_subplot(gs[3, 2])
            self._plot_risk_return(ax10, ticker)
            
            plt.suptitle(f'Análisis Cuantitativo Completo: {ticker}', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            path = f"{save_dir}/{ticker}_analysis.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            
            chart_paths[ticker] = path
            
        return chart_paths
        
    def _plot_price_evolution(self, ax, ticker):
        """Plot price evolution with moving averages."""
        prices = self.data[ticker]
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()
        
        ax.plot(prices.index, prices, label='Precio', linewidth=1.5, color='#2c5aa0')
        ax.plot(ma50.index, ma50, label='MA 50', linewidth=1, linestyle='--', alpha=0.7)
        ax.plot(ma200.index, ma200, label='MA 200', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_title('Evolución del Precio', fontweight='bold')
        ax.set_ylabel('Precio ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    def _plot_returns_distribution(self, ax, ticker):
        """Plot returns distribution with Gaussian overlay."""
        returns = self.returns[ticker].dropna()
        
        # Histogram
        n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.7, 
                                   color='skyblue', edgecolor='black')
        
        # Fit Gaussian
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        
        ax.set_title('Distribución de Retornos', fontweight='bold')
        ax.set_xlabel('Retorno Diario')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_qq(self, ax, ticker):
        """QQ plot for normality test."""
        returns = self.returns[ticker].dropna()
        stats.probplot(returns, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Test de Normalidad)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    def _plot_rolling_volatility(self, ax, ticker):
        """Plot rolling volatility."""
        returns = self.returns[ticker]
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        
        ax.plot(rolling_vol.index, rolling_vol, color='orange', linewidth=1.5)
        ax.fill_between(rolling_vol.index, rolling_vol, alpha=0.3, color='orange')
        ax.set_title('Volatilidad Móvil (30 días)', fontweight='bold')
        ax.set_ylabel('Volatilidad Anualizada')
        ax.grid(True, alpha=0.3)
        
    def _plot_cumulative_returns(self, ax, ticker):
        """Plot cumulative returns vs benchmark."""
        cum_returns = (1 + self.returns[ticker]).cumprod()
        cum_bench = (1 + self.returns[self.benchmark_ticker]).cumprod()
        
        ax.plot(cum_returns.index, cum_returns, label=ticker, linewidth=2)
        ax.plot(cum_bench.index, cum_bench, label=self.benchmark_ticker, 
               linewidth=2, linestyle='--', alpha=0.7)
        ax.set_title('Retornos Acumulados', fontweight='bold')
        ax.set_ylabel('Valor ($1 inicial)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_drawdown(self, ax, ticker):
        """Plot drawdown analysis."""
        cum_returns = (1 + self.returns[ticker]).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
    def _plot_monthly_heatmap(self, ax, ticker):
        """Plot monthly returns heatmap."""
        returns = self.returns[ticker]
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot = monthly_returns.groupby([monthly_returns.index.year, 
                                        monthly_returns.index.month]).mean().unstack()
        
        if len(pivot) > 0:
            sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                       ax=ax, cbar_kws={'label': 'Retorno'})
            ax.set_title('Retornos Mensuales', fontweight='bold')
            ax.set_xlabel('Mes')
            ax.set_ylabel('Año')
        
    def _plot_benchmark_correlation(self, ax, ticker):
        """Plot correlation with benchmark."""
        asset_returns = self.returns[ticker].dropna()
        bench_returns = self.returns[self.benchmark_ticker].dropna()
        
        # Align indices
        common_idx = asset_returns.index.intersection(bench_returns.index)
        asset_returns = asset_returns.loc[common_idx]
        bench_returns = bench_returns.loc[common_idx]
        
        ax.scatter(bench_returns, asset_returns, alpha=0.5, s=10)
        
        # Linear regression
        slope, intercept = np.polyfit(bench_returns, asset_returns, 1)
        x_line = np.array([bench_returns.min(), bench_returns.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f'β = {slope:.2f}')
        
        corr = asset_returns.corr(bench_returns)
        ax.set_title(f'Correlación con {self.benchmark_ticker} (ρ={corr:.2f})', 
                    fontweight='bold')
        ax.set_xlabel(f'Retorno {self.benchmark_ticker}')
        ax.set_ylabel(f'Retorno {ticker}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_volume(self, ax, ticker):
        """Plot volume analysis (if available)."""
        ax.text(0.5, 0.5, 'Análisis de Volumen\n(Requiere datos adicionales)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Volumen de Operaciones', fontweight='bold')
        ax.axis('off')
        
    def _plot_risk_return(self, ax, ticker):
        """Plot risk-return profile."""
        annual_return = self.returns[ticker].mean() * 252
        annual_vol = self.returns[ticker].std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        
        bench_return = self.returns[self.benchmark_ticker].mean() * 252
        bench_vol = self.returns[self.benchmark_ticker].std() * np.sqrt(252)
        
        ax.scatter(annual_vol, annual_return, s=200, c='blue', marker='o', 
                  label=ticker, edgecolors='black', linewidths=2)
        ax.scatter(bench_vol, bench_return, s=200, c='red', marker='s', 
                  label=self.benchmark_ticker, edgecolors='black', linewidths=2)
        
        ax.set_title(f'Perfil Riesgo-Retorno (Sharpe={sharpe:.2f})', fontweight='bold')
        ax.set_xlabel('Volatilidad Anualizada')
        ax.set_ylabel('Retorno Anualizado')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def get_statistics_table(self) -> pd.DataFrame:
        """Generate comprehensive statistics table for all assets."""
        stats_dict = {}
        
        for ticker in self.tickers:
            returns = self.returns[ticker].dropna()
            
            stats_dict[ticker] = {
                'Retorno Anual (%)': returns.mean() * 252 * 100,
                'Volatilidad Anual (%)': returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Asimetría': returns.skew(),
                'Curtosis': returns.kurtosis(),
                'Máximo Drawdown (%)': self._calculate_max_drawdown(ticker) * 100,
                'VaR 95% (%)': np.percentile(returns, 5) * 100,
                'CVaR 95% (%)': returns[returns <= np.percentile(returns, 5)].mean() * 100,
            }
            
        return pd.DataFrame(stats_dict).T
        
    def _calculate_max_drawdown(self, ticker):
        """Calculate maximum drawdown."""
        cum_returns = (1 + self.returns[ticker]).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

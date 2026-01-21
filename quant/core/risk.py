"""
Advanced Risk Metrics Module
Calculates institutional-level risk metrics including VaR, CVaR, Alpha, Beta, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, List
import seaborn as sns


class AdvancedRiskMetrics:
    """Calculates advanced risk metrics for institutional reporting."""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                 portfolio_weights: np.ndarray, asset_returns: pd.DataFrame):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.portfolio_weights = portfolio_weights
        self.asset_returns = asset_returns
        
    def calculate_alpha_beta(self) -> Dict:
        """Calculate Alpha and Beta vs benchmark."""
        # Align returns
        common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        port_ret = self.portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]
        
        # Linear regression: portfolio_return = alpha + beta * benchmark_return
        slope, intercept, r_value, p_value, std_err = stats.linregress(bench_ret, port_ret)
        
        # Annualize alpha
        alpha_daily = intercept
        alpha_annual = alpha_daily * 252
        
        beta = slope
        r_squared = r_value ** 2
        
        return {
            'alpha_annual': alpha_annual,
            'alpha_daily': alpha_daily,
            'beta': beta,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err
        }
    
    def calculate_tracking_error(self) -> float:
        """Calculate tracking error (annualized)."""
        common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        port_ret = self.portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]
        
        excess_returns = port_ret - bench_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return tracking_error
    
    def calculate_information_ratio(self) -> float:
        """Calculate Information Ratio."""
        common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        port_ret = self.portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]
        
        excess_returns = port_ret - bench_ret
        avg_excess = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error > 0:
            information_ratio = avg_excess / tracking_error
        else:
            information_ratio = 0
        
        return information_ratio
    
    def calculate_var_cvar(self, confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """Calculate VaR and CVaR at different confidence levels (historical method)."""
        results = {}
        
        for conf in confidence_levels:
            # VaR (percentile method)
            var = np.percentile(self.portfolio_returns, (1 - conf) * 100)
            
            # CVaR (Expected Shortfall)
            cvar = self.portfolio_returns[self.portfolio_returns <= var].mean()
            
            results[f'VaR_{int(conf*100)}'] = var
            results[f'CVaR_{int(conf*100)}'] = cvar
        
        return results
    
    def perform_stress_tests(self) -> Dict:
        """Perform institutional stress tests."""
        stress_results = {}
        
        # Current metrics
        current_return = self.portfolio_returns.mean() * 252
        current_vol = self.portfolio_returns.std() * np.sqrt(252)
        current_dd = self._calculate_max_drawdown(self.portfolio_returns)
        
        # 1. Market shock scenarios
        for shock in [-0.10, -0.20, -0.30]:
            shocked_returns = self.portfolio_returns + shock/252  # Daily shock
            
            stress_results[f'Market_Shock_{int(shock*100)}pct'] = {
                'return': shocked_returns.mean() * 252,
                'volatility': shocked_returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(shocked_returns),
                'sharpe': (shocked_returns.mean() * 252) / (shocked_returns.std() * np.sqrt(252))
            }
        
        # 2. Volatility shock
        for vol_mult in [1.5, 2.0, 2.5]:
            # Increase volatility while keeping mean
            mean_ret = self.portfolio_returns.mean()
            shocked_returns = mean_ret + (self.portfolio_returns - mean_ret) * vol_mult
            
            stress_results[f'Vol_Shock_{vol_mult}x'] = {
                'return': shocked_returns.mean() * 252,
                'volatility': shocked_returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(shocked_returns),
                'sharpe': (shocked_returns.mean() * 252) / (shocked_returns.std() * np.sqrt(252))
            }
        
        # 3. Tech sector shock (if we can identify tech stocks)
        # Assuming first 7 stocks are tech based on clustering
        tech_shock = -0.25  # 25% drop in tech
        shocked_weights = self.portfolio_weights.copy()
        # Apply shock to tech-heavy positions
        tech_indices = [0, 1, 2, 3, 4, 5, 6]  # AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
        
        shocked_asset_returns = self.asset_returns.copy()
        for idx in tech_indices:
            if idx < len(shocked_asset_returns.columns):
                shocked_asset_returns.iloc[:, idx] += tech_shock / 252
        
        shocked_portfolio_returns = (shocked_asset_returns * self.portfolio_weights).sum(axis=1)
        
        stress_results['Tech_Sector_Shock_-25pct'] = {
            'return': shocked_portfolio_returns.mean() * 252,
            'volatility': shocked_portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(shocked_portfolio_returns),
            'sharpe': (shocked_portfolio_returns.mean() * 252) / (shocked_portfolio_returns.std() * np.sqrt(252))
        }
        
        return stress_results
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def create_benchmark_comparison_chart(self, save_path: str) -> None:
        """Create benchmark comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Equity curves
        ax1 = axes[0, 0]
        common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        port_cum = (1 + self.portfolio_returns.loc[common_idx]).cumprod()
        bench_cum = (1 + self.benchmark_returns.loc[common_idx]).cumprod()
        
        ax1.plot(port_cum.index, port_cum, label='Portfolio', linewidth=2, color='#2c5aa0')
        ax1.plot(bench_cum.index, bench_cum, label='Benchmark', linewidth=2, 
                linestyle='--', alpha=0.7, color='orange')
        ax1.set_title('Equity Curves: Portfolio vs Benchmark', fontweight='bold')
        ax1.set_ylabel('Valor Acumulado ($1 inicial)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Excess returns (cumulative)
        ax2 = axes[0, 1]
        excess_cum = port_cum - bench_cum
        ax2.fill_between(excess_cum.index, excess_cum, 0, 
                        where=(excess_cum >= 0), color='green', alpha=0.3, label='Outperformance')
        ax2.fill_between(excess_cum.index, excess_cum, 0, 
                        where=(excess_cum < 0), color='red', alpha=0.3, label='Underperformance')
        ax2.plot(excess_cum.index, excess_cum, color='black', linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax2.set_title('Exceso de Retorno Acumulado', fontweight='bold')
        ax2.set_ylabel('Exceso ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Beta (60 days)
        ax3 = axes[1, 0]
        rolling_beta = self._calculate_rolling_beta(60)
        ax3.plot(rolling_beta.index, rolling_beta, color='purple', linewidth=1.5)
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Beta = 1')
        ax3.set_title('Beta Móvil (60 días)', fontweight='bold')
        ax3.set_ylabel('Beta')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter: Portfolio vs Benchmark returns
        ax4 = axes[1, 1]
        port_ret = self.portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]
        
        ax4.scatter(bench_ret * 100, port_ret * 100, alpha=0.5, s=10)
        
        # Regression line
        slope, intercept = np.polyfit(bench_ret, port_ret, 1)
        x_line = np.array([bench_ret.min(), bench_ret.max()])
        y_line = slope * x_line + intercept
        ax4.plot(x_line * 100, y_line * 100, 'r-', linewidth=2, 
                label=f'β = {slope:.2f}, α = {intercept*252*100:.2f}% anual')
        
        ax4.set_title('Retornos: Portfolio vs Benchmark', fontweight='bold')
        ax4.set_xlabel('Retorno Benchmark (%)')
        ax4.set_ylabel('Retorno Portfolio (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _calculate_rolling_beta(self, window: int = 60) -> pd.Series:
        """Calculate rolling beta."""
        common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        port_ret = self.portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]
        
        rolling_beta = pd.Series(index=port_ret.index, dtype=float)
        
        for i in range(window, len(port_ret)):
            window_port = port_ret.iloc[i-window:i]
            window_bench = bench_ret.iloc[i-window:i]
            
            if len(window_port) == window and len(window_bench) == window:
                slope, _ = np.polyfit(window_bench, window_port, 1)
                rolling_beta.iloc[i] = slope
        
        return rolling_beta.dropna()
    
    def create_tail_risk_chart(self, save_path: str) -> None:
        """Create tail risk visualization with VaR/CVaR."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Distribution with VaR/CVaR markers
        ax1 = axes[0]
        
        returns_pct = self.portfolio_returns * 100
        
        # Histogram
        n, bins, patches = ax1.hist(returns_pct, bins=50, density=True, 
                                    alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = returns_pct.mean(), returns_pct.std()
        x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
        
        # VaR and CVaR lines
        var_95 = np.percentile(returns_pct, 5)
        var_99 = np.percentile(returns_pct, 1)
        cvar_95 = returns_pct[returns_pct <= var_95].mean()
        cvar_99 = returns_pct[returns_pct <= var_99].mean()
        
        ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, 
                   label=f'VaR 95%: {var_95:.2f}%')
        ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, 
                   label=f'VaR 99%: {var_99:.2f}%')
        ax1.axvline(cvar_95, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'CVaR 95%: {cvar_95:.2f}%')
        ax1.axvline(cvar_99, color='red', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'CVaR 99%: {cvar_99:.2f}%')
        
        ax1.set_title('Distribución de Retornos con Riesgo de Cola', fontweight='bold')
        ax1.set_xlabel('Retorno Diario (%)')
        ax1.set_ylabel('Densidad')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot for tail analysis
        ax2 = axes[1]
        stats.probplot(returns_pct, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Análisis de Colas', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_stress_test_heatmap(self, stress_results: Dict, save_path: str) -> None:
        """Create stress test results heatmap."""
        # Prepare data for heatmap
        scenarios = list(stress_results.keys())
        metrics = ['return', 'volatility', 'max_drawdown', 'sharpe']
        
        data = []
        for scenario in scenarios:
            row = [
                stress_results[scenario]['return'] * 100,
                stress_results[scenario]['volatility'] * 100,
                stress_results[scenario]['max_drawdown'] * 100,
                stress_results[scenario]['sharpe']
            ]
            data.append(row)
        
        df = pd.DataFrame(data, index=scenarios, columns=metrics)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize for color mapping
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   ax=ax, cbar_kws={'label': 'Valor'}, linewidths=0.5)
        
        ax.set_title('Stress Testing: Impacto en Métricas de Portfolio', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Métricas', fontweight='bold')
        ax.set_ylabel('Escenarios de Stress', fontweight='bold')
        
        # Rotate labels
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

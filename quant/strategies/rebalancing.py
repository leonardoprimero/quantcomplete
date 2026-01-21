"""
Rebalancing Strategy Module
Implements portfolio monitoring and rebalancing strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import seaborn as sns


class RebalancingStrategy:
    """Implements various portfolio rebalancing strategies."""
    
    def __init__(self, data: pd.DataFrame, returns: pd.DataFrame, 
                 target_weights: np.ndarray, tickers: List[str]):
        self.data = data
        self.returns = returns
        self.target_weights = target_weights
        self.tickers = tickers
        
    def calendar_rebalancing(self, frequency: str = 'Q', 
                            initial_capital: float = 10000) -> Dict:
        """
        Calendar-based rebalancing (monthly, quarterly, annually).
        
        Args:
            frequency: 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
            initial_capital: Starting capital
        """
        portfolio_value = pd.Series(index=self.data.index, dtype=float)
        rebalance_dates = []
        transaction_costs = []
        
        # Initial allocation
        current_weights = self.target_weights.copy()
        portfolio_value.iloc[0] = initial_capital
        
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            prev_date = self.data.index[i-1]
            
            # Calculate daily returns
            daily_returns = (self.data.loc[date, self.tickers].values / 
                           self.data.loc[prev_date, self.tickers].values - 1)
            
            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + np.sum(current_weights * daily_returns))
            
            # Update weights due to price changes (drift)
            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / current_weights.sum()
            
            # Check if rebalancing date
            if self._is_rebalance_date(date, prev_date, frequency):
                # Calculate transaction cost (0.1% per trade)
                weight_diff = np.abs(current_weights - self.target_weights)
                cost = portfolio_value.iloc[i] * weight_diff.sum() * 0.001
                transaction_costs.append(cost)
                portfolio_value.iloc[i] -= cost
                
                # Rebalance to target weights
                current_weights = self.target_weights.copy()
                rebalance_dates.append(date)
        
        return {
            'portfolio_value': portfolio_value,
            'rebalance_dates': rebalance_dates,
            'num_rebalances': len(rebalance_dates),
            'total_transaction_costs': sum(transaction_costs),
            'final_value': portfolio_value.iloc[-1],
            'total_return': (portfolio_value.iloc[-1] / initial_capital - 1) * 100
        }
    
    def threshold_rebalancing(self, threshold: float = 0.05, 
                             initial_capital: float = 10000) -> Dict:
        """
        Threshold-based rebalancing (rebalance when weights drift > threshold).
        
        Args:
            threshold: Maximum allowed deviation from target weight (e.g., 0.05 = 5%)
            initial_capital: Starting capital
        """
        portfolio_value = pd.Series(index=self.data.index, dtype=float)
        rebalance_dates = []
        transaction_costs = []
        weight_drifts = []
        
        current_weights = self.target_weights.copy()
        portfolio_value.iloc[0] = initial_capital
        
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            prev_date = self.data.index[i-1]
            
            # Calculate daily returns
            daily_returns = (self.data.loc[date, self.tickers].values / 
                           self.data.loc[prev_date, self.tickers].values - 1)
            
            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + np.sum(current_weights * daily_returns))
            
            # Update weights due to price changes
            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / current_weights.sum()
            
            # Calculate maximum drift
            max_drift = np.max(np.abs(current_weights - self.target_weights))
            weight_drifts.append(max_drift)
            
            # Check if rebalancing needed
            if max_drift > threshold:
                # Calculate transaction cost
                weight_diff = np.abs(current_weights - self.target_weights)
                cost = portfolio_value.iloc[i] * weight_diff.sum() * 0.001
                transaction_costs.append(cost)
                portfolio_value.iloc[i] -= cost
                
                # Rebalance
                current_weights = self.target_weights.copy()
                rebalance_dates.append(date)
        
        return {
            'portfolio_value': portfolio_value,
            'rebalance_dates': rebalance_dates,
            'num_rebalances': len(rebalance_dates),
            'total_transaction_costs': sum(transaction_costs),
            'final_value': portfolio_value.iloc[-1],
            'total_return': (portfolio_value.iloc[-1] / initial_capital - 1) * 100,
            'weight_drifts': weight_drifts,
            'avg_drift': np.mean(weight_drifts)
        }
    
    def buy_and_hold(self, initial_capital: float = 10000) -> Dict:
        """Buy and hold strategy (no rebalancing)."""
        portfolio_value = pd.Series(index=self.data.index, dtype=float)
        current_weights = self.target_weights.copy()
        portfolio_value.iloc[0] = initial_capital
        
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            prev_date = self.data.index[i-1]
            
            daily_returns = (self.data.loc[date, self.tickers].values / 
                           self.data.loc[prev_date, self.tickers].values - 1)
            
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + np.sum(current_weights * daily_returns))
            
            # Update weights (drift)
            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / current_weights.sum()
        
        return {
            'portfolio_value': portfolio_value,
            'rebalance_dates': [],
            'num_rebalances': 0,
            'total_transaction_costs': 0,
            'final_value': portfolio_value.iloc[-1],
            'total_return': (portfolio_value.iloc[-1] / initial_capital - 1) * 100
        }
    
    def _is_rebalance_date(self, current_date, prev_date, frequency):
        """Check if current date is a rebalancing date."""
        if frequency == 'M':  # Monthly
            return current_date.month != prev_date.month
        elif frequency == 'Q':  # Quarterly
            return current_date.quarter != prev_date.quarter
        elif frequency == 'Y':  # Yearly
            return current_date.year != prev_date.year
        return False
    
    def compare_strategies(self, initial_capital: float = 10000) -> pd.DataFrame:
        """Compare different rebalancing strategies."""
        strategies = {
            'Buy & Hold': self.buy_and_hold(initial_capital),
            'Mensual': self.calendar_rebalancing('M', initial_capital),
            'Trimestral': self.calendar_rebalancing('Q', initial_capital),
            'Anual': self.calendar_rebalancing('Y', initial_capital),
            'Umbral 3%': self.threshold_rebalancing(0.03, initial_capital),
            'Umbral 5%': self.threshold_rebalancing(0.05, initial_capital),
            'Umbral 10%': self.threshold_rebalancing(0.10, initial_capital),
        }
        
        comparison = []
        for name, result in strategies.items():
            comparison.append({
                'Estrategia': name,
                'Retorno Total (%)': result['total_return'],
                'Valor Final ($)': result['final_value'],
                'Num. Rebalanceos': result['num_rebalances'],
                'Costos Transacción ($)': result['total_transaction_costs'],
                'Retorno Neto (%)': ((result['final_value'] / initial_capital) - 1) * 100
            })
        
        return pd.DataFrame(comparison), strategies
    
    def create_monitoring_dashboard(self, strategies_results: Dict, 
                                   save_path: str) -> None:
        """Create comprehensive monitoring dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # 1. Portfolio value comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_value_comparison(ax1, strategies_results)
        
        # 2. Returns comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_returns_comparison(ax2, strategies_results)
        
        # 3. Number of rebalances
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rebalance_frequency(ax3, strategies_results)
        
        # 4. Transaction costs
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_transaction_costs(ax4, strategies_results)
        
        # 5. Drawdown comparison
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_drawdown_comparison(ax5, strategies_results)
        
        # 6. Rolling Sharpe
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_rolling_sharpe(ax6, strategies_results)
        
        # 7. Volatility comparison
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_volatility_comparison(ax7, strategies_results)
        
        # 8. Risk-return scatter
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_risk_return_scatter(ax8, strategies_results)
        
        plt.suptitle('Dashboard de Monitoreo y Rebalanceo de Portfolio', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_value_comparison(self, ax, strategies):
        """Plot portfolio value evolution for all strategies."""
        for name, result in strategies.items():
            ax.plot(result['portfolio_value'].index, result['portfolio_value'], 
                   label=name, linewidth=2, alpha=0.8)
            
            # Mark rebalance dates
            if result['rebalance_dates']:
                rebal_values = [result['portfolio_value'].loc[d] for d in result['rebalance_dates']]
                ax.scatter(result['rebalance_dates'], rebal_values, 
                          s=30, alpha=0.6, zorder=5)
        
        ax.set_title('Evolución del Valor del Portfolio', fontweight='bold', fontsize=12)
        ax.set_ylabel('Valor ($)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _plot_returns_comparison(self, ax, strategies):
        """Plot returns comparison."""
        names = list(strategies.keys())
        returns = [strategies[name]['total_return'] for name in names]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax.barh(names, returns, color=colors, alpha=0.7)
        ax.set_title('Retorno Total por Estrategia', fontweight='bold', fontsize=11)
        ax.set_xlabel('Retorno (%)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, returns)):
            ax.text(val, i, f' {val:.1f}%', va='center', fontsize=9)
    
    def _plot_rebalance_frequency(self, ax, strategies):
        """Plot number of rebalances."""
        names = list(strategies.keys())
        rebalances = [strategies[name]['num_rebalances'] for name in names]
        
        ax.bar(range(len(names)), rebalances, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title('Número de Rebalanceos', fontweight='bold', fontsize=11)
        ax.set_ylabel('Cantidad')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_transaction_costs(self, ax, strategies):
        """Plot transaction costs."""
        names = list(strategies.keys())
        costs = [strategies[name]['total_transaction_costs'] for name in names]
        
        ax.bar(range(len(names)), costs, color='coral', alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title('Costos de Transacción', fontweight='bold', fontsize=11)
        ax.set_ylabel('Costo ($)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    def _plot_drawdown_comparison(self, ax, strategies):
        """Plot drawdown for all strategies."""
        for name, result in strategies.items():
            pv = result['portfolio_value']
            running_max = pv.expanding().max()
            drawdown = (pv - running_max) / running_max * 100
            ax.plot(drawdown.index, drawdown, label=name, linewidth=1.5, alpha=0.7)
        
        ax.set_title('Comparación de Drawdown', fontweight='bold', fontsize=12)
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    def _plot_rolling_sharpe(self, ax, strategies):
        """Plot rolling Sharpe ratio."""
        for name, result in strategies.items():
            returns = result['portfolio_value'].pct_change()
            rolling_sharpe = (returns.rolling(60).mean() * 252) / (returns.rolling(60).std() * np.sqrt(252))
            ax.plot(rolling_sharpe.index, rolling_sharpe, label=name, linewidth=1.5, alpha=0.7)
        
        ax.set_title('Sharpe Ratio Móvil (60 días)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    def _plot_volatility_comparison(self, ax, strategies):
        """Plot volatility comparison."""
        names = list(strategies.keys())
        vols = []
        for name in names:
            returns = strategies[name]['portfolio_value'].pct_change()
            vol = returns.std() * np.sqrt(252) * 100
            vols.append(vol)
        
        ax.bar(range(len(names)), vols, color='orange', alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title('Volatilidad Anualizada', fontweight='bold', fontsize=11)
        ax.set_ylabel('Volatilidad (%)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_risk_return_scatter(self, ax, strategies):
        """Plot risk-return scatter."""
        for name, result in strategies.items():
            returns = result['portfolio_value'].pct_change()
            annual_return = result['total_return'] / 3  # Assuming 3 years
            annual_vol = returns.std() * np.sqrt(252) * 100
            
            ax.scatter(annual_vol, annual_return, s=150, alpha=0.7, 
                      label=name, edgecolors='black', linewidths=1)
        
        ax.set_title('Perfil Riesgo-Retorno', fontweight='bold', fontsize=11)
        ax.set_xlabel('Volatilidad Anualizada (%)')
        ax.set_ylabel('Retorno Anualizado (%)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

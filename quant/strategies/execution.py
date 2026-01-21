"""
Execution and Liquidity Analysis Module
Analyzes execution costs, slippage, liquidity constraints, and AUM scalability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class ExecutionLiquidityAnalysis:
    """Analyzes execution costs and liquidity constraints."""
    
    def __init__(self, portfolio_weights: np.ndarray, tickers: List[str],
                 base_transaction_cost: float = 0.001):
        self.portfolio_weights = portfolio_weights
        self.tickers = tickers
        self.base_transaction_cost = base_transaction_cost
        
    def calculate_execution_costs(self, aum_levels: List[float],
                                  slippage_bps: float = 5.0) -> Dict:
        """
        Calculate execution costs at different AUM levels.
        
        Args:
            aum_levels: List of AUM levels to analyze (e.g., [10000, 100000, 1000000])
            slippage_bps: Slippage in basis points (default: 5 bps)
        """
        results = {}
        
        slippage_rate = slippage_bps / 10000  # Convert bps to decimal
        
        for aum in aum_levels:
            # Base transaction cost
            base_cost = aum * self.base_transaction_cost
            
            # Slippage cost (proportional to trade size)
            # Assume slippage increases with AUM
            slippage_cost = aum * slippage_rate
            
            # Market impact (simplified model: proportional to sqrt(AUM))
            # This is a simplified version of the square-root market impact model
            market_impact_factor = np.sqrt(aum / 10000)  # Normalized to 10k base
            market_impact_cost = aum * 0.0005 * market_impact_factor
            
            total_cost = base_cost + slippage_cost + market_impact_cost
            total_cost_bps = (total_cost / aum) * 10000
            
            results[f'AUM_${aum:,.0f}'] = {
                'aum': aum,
                'base_cost': base_cost,
                'slippage_cost': slippage_cost,
                'market_impact': market_impact_cost,
                'total_cost': total_cost,
                'total_cost_bps': total_cost_bps,
                'cost_percentage': (total_cost / aum) * 100
            }
        
        return results
    
    def apply_weight_constraints(self, max_weight: float = 0.30,
                                 min_weight: float = 0.0) -> Dict:
        """
        Apply weight constraints and recalculate portfolio.
        
        Args:
            max_weight: Maximum weight per asset (default: 30%)
            min_weight: Minimum weight per asset (default: 0%)
        """
        constrained_weights = self.portfolio_weights.copy()
        
        # Apply max constraint
        over_max = constrained_weights > max_weight
        if over_max.any():
            excess = constrained_weights[over_max].sum() - (over_max.sum() * max_weight)
            constrained_weights[over_max] = max_weight
            
            # Redistribute excess to other assets proportionally
            under_max = ~over_max
            if under_max.any():
                redistribution = excess * (constrained_weights[under_max] / constrained_weights[under_max].sum())
                constrained_weights[under_max] += redistribution
        
        # Apply min constraint
        under_min = constrained_weights < min_weight
        if under_min.any():
            deficit = (min_weight * under_min.sum()) - constrained_weights[under_min].sum()
            constrained_weights[under_min] = min_weight
            
            # Remove deficit from other assets proportionally
            over_min = ~under_min
            if over_min.any():
                reduction = deficit * (constrained_weights[over_min] / constrained_weights[over_min].sum())
                constrained_weights[over_min] -= reduction
        
        # Normalize to sum to 1
        constrained_weights = constrained_weights / constrained_weights.sum()
        
        return {
            'original_weights': self.portfolio_weights,
            'constrained_weights': constrained_weights,
            'max_constraint': max_weight,
            'min_constraint': min_weight,
            'weights_changed': not np.allclose(self.portfolio_weights, constrained_weights)
        }
    
    def create_execution_analysis_table(self, execution_costs: Dict) -> pd.DataFrame:
        """Create execution analysis summary table."""
        data = []
        
        for aum_label, costs in execution_costs.items():
            data.append({
                'AUM': f"${costs['aum']:,.0f}",
                'Costo Base ($)': f"${costs['base_cost']:.2f}",
                'Slippage ($)': f"${costs['slippage_cost']:.2f}",
                'Impacto Mercado ($)': f"${costs['market_impact']:.2f}",
                'Costo Total ($)': f"${costs['total_cost']:.2f}",
                'Costo Total (bps)': f"{costs['total_cost_bps']:.1f}",
                'Costo (%)': f"{costs['cost_percentage']:.3f}%"
            })
        
        return pd.DataFrame(data)
    
    def create_liquidity_constraints_table(self) -> pd.DataFrame:
        """Create liquidity constraints framework table."""
        data = [
            {
                'Parámetro': 'Costo de Transacción Base',
                'Valor': '0.10%',
                'Notas': 'Comisiones de broker + fees de exchange'
            },
            {
                'Parámetro': 'Slippage Estimado',
                'Valor': '5 bps',
                'Notas': 'Basado en liquidez de large-cap stocks'
            },
            {
                'Parámetro': 'Peso Máximo por Activo',
                'Valor': '30%',
                'Notas': 'Límite de concentración para diversificación'
            },
            {
                'Parámetro': 'Peso Mínimo por Activo',
                'Valor': '0%',
                'Notas': 'Permite exclusión de activos'
            },
            {
                'Parámetro': 'Impacto de Mercado',
                'Valor': 'Modelo √AUM',
                'Notas': 'Aumenta con raíz cuadrada del tamaño de orden'
            },
            {
                'Parámetro': 'Liquidez Mínima',
                'Valor': 'Large-cap',
                'Notas': 'Todos los activos son S&P 500 constituents'
            },
            {
                'Parámetro': 'Rebalanceo',
                'Valor': 'Trimestral',
                'Notas': '4 eventos de trading por año'
            },
            {
                'Parámetro': 'Escalabilidad',
                'Valor': 'Hasta $10M',
                'Notas': 'Sin restricciones significativas de liquidez'
            }
        ]
        
        return pd.DataFrame(data)
    
    def create_scalability_chart(self, execution_costs: Dict, save_path: str) -> None:
        """Create AUM scalability visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        aum_values = [costs['aum'] for costs in execution_costs.values()]
        total_costs = [costs['total_cost'] for costs in execution_costs.values()]
        cost_bps = [costs['total_cost_bps'] for costs in execution_costs.values()]
        base_costs = [costs['base_cost'] for costs in execution_costs.values()]
        slippage_costs = [costs['slippage_cost'] for costs in execution_costs.values()]
        impact_costs = [costs['market_impact'] for costs in execution_costs.values()]
        
        # 1. Total cost vs AUM
        ax1 = axes[0, 0]
        ax1.plot(aum_values, total_costs, marker='o', linewidth=2, markersize=8, color='#2c5aa0')
        ax1.set_title('Costo Total de Ejecución vs AUM', fontweight='bold')
        ax1.set_xlabel('AUM ($)')
        ax1.set_ylabel('Costo Total ($)')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Cost in bps vs AUM
        ax2 = axes[0, 1]
        ax2.plot(aum_values, cost_bps, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_title('Costo de Ejecución (bps) vs AUM', fontweight='bold')
        ax2.set_xlabel('AUM ($)')
        ax2.set_ylabel('Costo (basis points)')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cost breakdown stacked bar
        ax3 = axes[1, 0]
        x_pos = np.arange(len(aum_values))
        ax3.bar(x_pos, base_costs, label='Costo Base', color='skyblue')
        ax3.bar(x_pos, slippage_costs, bottom=base_costs, label='Slippage', color='lightcoral')
        ax3.bar(x_pos, impact_costs, 
               bottom=np.array(base_costs) + np.array(slippage_costs),
               label='Impacto Mercado', color='lightgreen')
        
        ax3.set_title('Desglose de Costos por AUM', fontweight='bold')
        ax3.set_xlabel('Nivel de AUM')
        ax3.set_ylabel('Costo ($)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'${aum/1000:.0f}K' if aum < 1000000 else f'${aum/1000000:.1f}M' 
                             for aum in aum_values], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Cost as percentage of AUM
        ax4 = axes[1, 1]
        cost_pct = [(cost / aum) * 100 for cost, aum in zip(total_costs, aum_values)]
        ax4.plot(aum_values, cost_pct, marker='D', linewidth=2, markersize=8, color='purple')
        ax4.set_title('Costo como % del AUM', fontweight='bold')
        ax4.set_xlabel('AUM ($)')
        ax4.set_ylabel('Costo (%)')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}%'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class GovernanceFramework:
    """Defines governance and monitoring framework."""
    
    @staticmethod
    def get_monitoring_checklist() -> List[Dict]:
        """Get operational monitoring checklist."""
        return [
            {
                'Actividad': 'Monitoreo de Pesos',
                'Frecuencia': 'Semanal',
                'Umbral': 'Desviación > 5% del peso objetivo',
                'Acción': 'Revisar causas, considerar rebalanceo anticipado',
                'Responsable': 'Portfolio Manager'
            },
            {
                'Actividad': 'Revisión de Performance',
                'Frecuencia': 'Mensual',
                'Umbral': 'Sharpe Ratio < 1.0 por 2 meses consecutivos',
                'Acción': 'Análisis de atribución, revisar supuestos',
                'Responsable': 'Quantitative Analyst'
            },
            {
                'Actividad': 'Análisis de Drawdown',
                'Frecuencia': 'Semanal',
                'Umbral': 'Drawdown > 15% desde máximo',
                'Acción': 'Evaluación de riesgo, posible de-risking',
                'Responsable': 'Risk Manager'
            },
            {
                'Actividad': 'Recalibración de Covarianza',
                'Frecuencia': 'Trimestral',
                'Umbral': 'Cambio significativo en correlaciones (>0.2)',
                'Acción': 'Re-optimizar portfolio con nueva matriz',
                'Responsable': 'Quantitative Analyst'
            },
            {
                'Actividad': 'Actualización de Retornos Esperados',
                'Frecuencia': 'Trimestral',
                'Umbral': 'Cambio en régimen de mercado',
                'Acción': 'Revisar estimaciones, ajustar si necesario',
                'Responsable': 'Research Team'
            },
            {
                'Actividad': 'Stress Testing',
                'Frecuencia': 'Mensual',
                'Umbral': 'Pérdida > 20% en escenario adverso',
                'Acción': 'Revisar exposiciones, ajustar hedges',
                'Responsable': 'Risk Manager'
            },
            {
                'Actividad': 'Rebalanceo Programado',
                'Frecuencia': 'Trimestral',
                'Umbral': 'Fechas: 31 Mar, 30 Jun, 30 Sep, 31 Dic',
                'Acción': 'Ejecutar rebalanceo a pesos objetivo',
                'Responsable': 'Trading Desk'
            },
            {
                'Actividad': 'Revisión de Liquidez',
                'Frecuencia': 'Mensual',
                'Umbral': 'Volumen diario < 10x posición',
                'Acción': 'Evaluar reducción de posición',
                'Responsable': 'Trading Desk'
            },
            {
                'Actividad': 'Análisis de Tracking Error',
                'Frecuencia': 'Mensual',
                'Umbral': 'TE > 8% anualizado',
                'Acción': 'Investigar fuentes, ajustar si necesario',
                'Responsable': 'Portfolio Manager'
            },
            {
                'Actividad': 'Revisión de Costos',
                'Frecuencia': 'Trimestral',
                'Umbral': 'Costos acumulados > 0.5% AUM',
                'Acción': 'Optimizar frecuencia de rebalanceo',
                'Responsable': 'Operations'
            }
        ]
    
    @staticmethod
    def get_trigger_events() -> List[Dict]:
        """Get list of trigger events for immediate action."""
        return [
            {
                'Evento': 'Drawdown > 20%',
                'Severidad': 'Alta',
                'Acción Inmediata': 'Convocar comité de riesgo, evaluar de-risking',
                'Plazo': '24 horas'
            },
            {
                'Evento': 'Sharpe Ratio < 0.5',
                'Severidad': 'Media',
                'Acción Inmediata': 'Análisis de atribución completo, revisar estrategia',
                'Plazo': '1 semana'
            },
            {
                'Evento': 'Desviación de peso > 10%',
                'Severidad': 'Media',
                'Acción Inmediata': 'Rebalanceo extraordinario',
                'Plazo': '2 días'
            },
            {
                'Evento': 'Correlación con benchmark < 0.3',
                'Severidad': 'Media',
                'Acción Inmediata': 'Revisar composición, validar estrategia',
                'Plazo': '1 semana'
            },
            {
                'Evento': 'VaR 99% excedido',
                'Severidad': 'Alta',
                'Acción Inmediata': 'Análisis de evento, stress test adicional',
                'Plazo': '48 horas'
            },
            {
                'Evento': 'Cambio regulatorio',
                'Severidad': 'Variable',
                'Acción Inmediata': 'Evaluación legal, ajustes de compliance',
                'Plazo': 'Según regulación'
            },
            {
                'Evento': 'Crisis de mercado (VIX > 40)',
                'Severidad': 'Alta',
                'Acción Inmediata': 'Monitoreo continuo, preparar plan de contingencia',
                'Plazo': 'Inmediato'
            },
            {
                'Evento': 'Pérdida de liquidez en activo',
                'Severidad': 'Alta',
                'Acción Inmediata': 'Evaluar salida ordenada, buscar sustitutos',
                'Plazo': '1 semana'
            }
        ]
    
    @staticmethod
    def create_governance_table(checklist: List[Dict]) -> pd.DataFrame:
        """Create governance checklist table."""
        return pd.DataFrame(checklist)
    
    @staticmethod
    def create_trigger_events_table(triggers: List[Dict]) -> pd.DataFrame:
        """Create trigger events table."""
        return pd.DataFrame(triggers)

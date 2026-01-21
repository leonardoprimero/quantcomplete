"""
Generate Rebalancing Strategy Report
Creates comprehensive monitoring and rebalancing analysis.

Usage:
    python generate_rebalancing_report.py
"""

from portfolio_data import PortfolioData
from markowitz import MarkowitzOptimizer
from rebalancing import RebalancingStrategy
from rebalancing_report import RebalancingReportGenerator


def main():
    """Generate rebalancing strategy report."""
    # Configuration
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'JPM', 'V', 'WMT'
    ]
    
    HORIZON_YEARS = 1  # Horizonte de inversión: 1 año
    BASE_DIR = '/Users/leguillo/prueba antigravity/quant'
    DASHBOARD_PATH = f'{BASE_DIR}/rebalancing_dashboard.png'
    REPORT_PATH = f'{BASE_DIR}/informe_rebalanceo.pdf'
    
    print("="*70)
    print("ANÁLISIS DE ESTRATEGIAS DE REBALANCEO")
    print("="*70)
    print(f"Horizonte de inversión: {HORIZON_YEARS} año")
    print(f"Activos: {len(TICKERS)}")
    print("="*70 + "\n")
    
    # 1. Download and prepare data
    print("📊 Descargando datos históricos...")
    portfolio_data = PortfolioData(TICKERS, years=3)
    portfolio_data.download_data()
    portfolio_data.calculate_returns()
    
    # 2. Get optimal weights from Markowitz
    print("⚡ Calculando pesos óptimos (Markowitz)...")
    optimizer = MarkowitzOptimizer(
        portfolio_data.mean_returns,
        portfolio_data.cov_matrix
    )
    optimal_portfolio = optimizer.optimize_max_sharpe()
    target_weights = optimal_portfolio['weights']
    
    print(f"   ✓ Sharpe Ratio del portfolio óptimo: {optimal_portfolio['sharpe']:.3f}")
    
    # 3. Initialize rebalancing strategy
    print("🔄 Analizando estrategias de rebalanceo...")
    rebalancer = RebalancingStrategy(
        portfolio_data.data,
        portfolio_data.log_returns,
        target_weights,
        TICKERS
    )
    
    # 4. Compare strategies
    comparison_df, strategies_results = rebalancer.compare_strategies(initial_capital=10000)
    
    print("\n" + "="*70)
    print("RESULTADOS DE ESTRATEGIAS")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70 + "\n")
    
    # 5. Create monitoring dashboard
    print("📊 Creando dashboard de monitoreo...")
    rebalancer.create_monitoring_dashboard(strategies_results, DASHBOARD_PATH)
    print(f"   ✓ Dashboard guardado en: {DASHBOARD_PATH}")
    
    # 6. Generate PDF report
    print("📄 Generando informe de rebalanceo...")
    report_generator = RebalancingReportGenerator(
        comparison_df,
        strategies_results,
        target_weights,
        TICKERS,
        horizon_years=HORIZON_YEARS
    )
    report_generator.generate_report(REPORT_PATH, DASHBOARD_PATH)
    
    # 7. Print recommendation
    print("\n" + "="*70)
    print("RECOMENDACIÓN PARA HORIZONTE DE 1 AÑO")
    print("="*70)
    
    if HORIZON_YEARS == 1:
        recommended = comparison_df[comparison_df['Estrategia'] == 'Trimestral'].iloc[0]
        print("Estrategia Recomendada: REBALANCEO TRIMESTRAL")
    else:
        recommended = comparison_df[comparison_df['Estrategia'] == 'Anual'].iloc[0]
        print("Estrategia Recomendada: REBALANCEO ANUAL")
    
    print(f"Retorno Esperado: {recommended['Retorno Neto (%)']:.2f}%")
    print(f"Valor Final: ${recommended['Valor Final ($)']:,.0f}")
    print(f"Número de Rebalanceos: {int(recommended['Num. Rebalanceos'])}")
    print(f"Costos de Transacción: ${recommended['Costos Transacción ($)']:.0f}")
    print("="*70 + "\n")
    
    print("✅ ANÁLISIS COMPLETADO")
    print(f"📊 Dashboard: {DASHBOARD_PATH}")
    print(f"📄 Informe: {REPORT_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()

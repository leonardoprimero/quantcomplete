"""
Master Script - Extended Professional Client Report Generation
JP Morgan Senior Quant Level Analysis with Advanced Sections

NEW SECTIONS ADDED:
1. Benchmarking y Performance Relativa (Alpha, Beta, Tracking Error, Information Ratio)
2. Stress Testing y Escenarios (Market shocks, volatility shocks, sector shocks)
3. Riesgo de Cola y Downside (VaR, CVaR at 95% and 99%)
4. Limitaciones del Modelo y Riesgo de Modelización
5. Ejecución, Liquidez y Escalabilidad (AUM analysis, slippage, constraints)
6. Gobernanza y Monitoreo (Operational checklist, trigger events)

Output: /quant_client_report/ directory
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Import all modules
# Import all modules
from quant.data.loader import PortfolioData
from quant.core.simulation import MonteCarloSimulator
from quant.core.optimization import MarkowitzOptimizer
from quant.core.statistics import AssetAnalyzer
from quant.strategies.backtest import PortfolioBacktester
from quant.strategies.rebalancing import RebalancingStrategy
from quant.core.clustering import ClusteringAnalyzer
from quant.core.risk import AdvancedRiskMetrics
from quant.strategies.execution import ExecutionLiquidityAnalysis, GovernanceFramework
from quant.reporting.printer import ResultsPrinter
from quant.config import *


def main(tickers: list = None, benchmark: str = None, start_date: str = None, end_date: str = None,
         num_simulations: int = None, initial_capital: float = None):
    """Generate complete professional client report package with extended sections."""
    
    # Use provided arguments or fallback to config defaults
    # Use provided arguments or fallback to config defaults
    analysis_tickers = tickers if tickers else TICKERS
    analysis_benchmark = benchmark if benchmark else BENCHMARK
    analysis_start = start_date if start_date else START_DATE
    analysis_end = end_date if end_date else END_DATE
    analysis_sims = num_simulations if num_simulations else NUM_PORTFOLIOS
    analysis_capital = initial_capital if initial_capital else INITIAL_CAPITAL
    
    print("="*80)
    print("GENERACIÓN DE INFORME PROFESIONAL EXTENDIDO PARA CLIENTE")
    print("Nivel: JP Morgan Senior Quant + Advanced Risk Sections")
    print("="*80)
    print(f"Tickers: {', '.join(analysis_tickers)}")
    print(f"Benchmark: {analysis_benchmark}")
    print(f"Benchmark: {analysis_benchmark}")
    print(f"Fecha Inicio: {analysis_start}")
    print(f"Fecha Fin: {analysis_end}")
    print(f"Capital Inicial: ${analysis_capital:,.2f}")
    print("="*80)
    print()
    """Main execution flow."""
    printer = ResultsPrinter()
    printer.print_header(analysis_tickers, analysis_sims)
    
    # Create directories
    for directory in [OUTPUT_DIR, CHARTS_DIR, INDIVIDUAL_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    print(f"\n📁 Directorio de salida: {os.path.abspath(OUTPUT_DIR)}\n")

    # 1. Data Acquisition
    printer.print_section("FASE 1: ADQUISICIÓN Y PROCESAMIENTO DE DATOS")
    print("📊 Descargando datos históricos...")
    portfolio_data = PortfolioData(analysis_tickers + [analysis_benchmark], start_date=analysis_start, end_date=analysis_end)
    portfolio_data.download_data()
    portfolio_data.calculate_returns()
    print(f"   ✓ Datos descargados: {len(portfolio_data.data)} días de trading")
    print(f"   ✓ Período: {portfolio_data.data.index[0].date()} a {portfolio_data.data.index[-1].date()}")
    
    individual_vols, individual_sharpe = portfolio_data.get_individual_stats()
    print(f"   ✓ Estadísticas calculadas para {len(analysis_tickers)} activos")
    print()
    
    # ========================================================================
    # PHASE 2: INDIVIDUAL ASSET ANALYSIS
    # ========================================================================
    print("FASE 2: ANÁLISIS INDIVIDUAL DE ACTIVOS")
    print("-" * 80)
    
    print("📈 Generando análisis cuantitativo detallado por activo...")
    print("   (10 gráficos por activo: precio, Gaussiana, QQ-plot, volatilidad, etc.)")
    
    asset_analyzer = AssetAnalyzer(portfolio_data.data, analysis_tickers, analysis_benchmark)
    asset_chart_paths = asset_analyzer.create_individual_analysis_charts(INDIVIDUAL_DIR)
    stats_table = asset_analyzer.get_statistics_table()
    
    print(f"   ✓ Generados {len(asset_chart_paths)} análisis individuales completos")
    print(f"   ✓ Ubicación: {INDIVIDUAL_DIR}/")
    print()
    
    # ========================================================================
    # PHASE 3: CLUSTERING ANALYSIS
    # ========================================================================
    print("FASE 3: ANÁLISIS DE CLUSTERING")
    print("-" * 80)
    
    print("🔬 Ejecutando clustering jerárquico...")
    clustering_analyzer = ClusteringAnalyzer(
        portfolio_data.log_returns[analysis_tickers],
        analysis_tickers
    )
    
    cluster_labels, cluster_info, linkage_matrix = clustering_analyzer.perform_hierarchical_clustering(n_clusters=3)
    
    print(f"   ✓ Identificados {len(cluster_info)} clusters de activos")
    for cluster_name, info in cluster_info.items():
        print(f"   • {cluster_name}: {info['size']} activos - {', '.join(info['tickers'])}")
    
    clustering_chart_path = f'{CHARTS_DIR}/clustering_analysis.png'
    clustering_analyzer.create_clustering_visualization(
        cluster_labels, linkage_matrix, clustering_chart_path
    )
    print(f"   ✓ Visualización guardada: clustering_analysis.png")
    print()
    
    # ========================================================================
    # PHASE 4: PORTFOLIO OPTIMIZATION
    # ========================================================================
    print("FASE 4: OPTIMIZACIÓN DE PORTFOLIO")
    print("-" * 80)
    
    print("⚡ Ejecutando optimización de Markowitz...")
    optimizer = MarkowitzOptimizer(
        portfolio_data.mean_returns[analysis_tickers],
        portfolio_data.cov_matrix.loc[analysis_tickers, analysis_tickers]
    )
    
    markowitz_portfolios = {
        'max_sharpe': optimizer.optimize_max_sharpe(),
        'min_vol': optimizer.optimize_min_volatility()
    }
    
    optimal_portfolio = markowitz_portfolios['max_sharpe']
    print(f"   ✓ Portfolio óptimo identificado")
    print(f"   • Sharpe Ratio: {optimal_portfolio['sharpe']:.4f}")
    print(f"   • Retorno esperado: {optimal_portfolio['return']*100:.2f}%")
    print(f"   • Volatilidad: {optimal_portfolio['volatility']*100:.2f}%")
    
    print("\n🎲 Ejecutando simulación Monte Carlo...")
    mc_simulator = MonteCarloSimulator(
        portfolio_data.mean_returns[analysis_tickers],
        portfolio_data.cov_matrix.loc[analysis_tickers, analysis_tickers],
        analysis_sims
    )
    mc_simulator.run_simulation()
    mc_portfolios = mc_simulator.get_optimal_portfolios()
    print(f"   ✓ {analysis_sims:,} portfolios simulados")
    
    efficient_vols, efficient_returns = optimizer.generate_efficient_frontier()
    print(f"   ✓ Frontera eficiente calculada ({len(efficient_vols)} puntos)")
    print()
    
    # ========================================================================
    # PHASE 5: BACKTESTING
    # ========================================================================
    print("FASE 5: BACKTESTING HISTÓRICO")
    print("-" * 80)
    
    print("🔬 Ejecutando backtesting del portfolio óptimo...")
    backtester = PortfolioBacktester(portfolio_data.data, portfolio_data.log_returns)
    backtest_results = backtester.backtest_portfolio(
        optimal_portfolio['weights'],
        analysis_tickers,
        initial_capital=analysis_capital
    )
    
    # Calculate portfolio returns for advanced metrics
    portfolio_returns = (portfolio_data.log_returns[analysis_tickers] * optimal_portfolio['weights']).sum(axis=1)
    
    print(f"   ✓ Backtesting completado")
    print(f"   • Capital inicial: ${backtest_results['initial_capital']:,.0f}")
    print(f"   • Capital final: ${backtest_results['final_capital']:,.0f}")
    print(f"   • Retorno total: {backtest_results['total_return']:.2f}%")
    print(f"   • Máximo drawdown: {backtest_results['max_drawdown']:.2f}%")
    print(f"   • Sortino Ratio: {backtest_results['sortino_ratio']:.3f}")
    
    backtest_chart_path = f'{CHARTS_DIR}/backtest_results.png'
    backtester.create_backtest_chart(
        backtest_results, analysis_benchmark, backtest_chart_path,
        "Portfolio Óptimo - Máximo Sharpe Ratio"
    )
    print(f"   ✓ Gráfico guardado: backtest_results.png")
    print()
    
    # ========================================================================
    # PHASE 6: ADVANCED RISK METRICS (NEW)
    # ========================================================================
    print("FASE 6: MÉTRICAS DE RIESGO AVANZADAS")
    print("-" * 80)
    
    print("📊 Calculando Alpha, Beta, Tracking Error, Information Ratio...")
    risk_analyzer = AdvancedRiskMetrics(
        portfolio_returns,
        portfolio_data.log_returns[analysis_benchmark],
        optimal_portfolio['weights'],
        portfolio_data.log_returns[analysis_tickers]
    )
    
    # Calculate metrics
    alpha_beta = risk_analyzer.calculate_alpha_beta()
    tracking_error = risk_analyzer.calculate_tracking_error()
    information_ratio = risk_analyzer.calculate_information_ratio()
    
    risk_metrics = {
        'alpha_annual': alpha_beta['alpha_annual'],
        'beta': alpha_beta['beta'],
        'r_squared': alpha_beta['r_squared'],
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    }
    
    print(f"   ✓ Alpha anualizado: {alpha_beta['alpha_annual']*100:.2f}%")
    print(f"   ✓ Beta: {alpha_beta['beta']:.3f}")
    print(f"   ✓ Tracking Error: {tracking_error*100:.2f}%")
    print(f"   ✓ Information Ratio: {information_ratio:.3f}")
    
    # Create benchmark comparison chart
    benchmark_chart_path = f'{CHARTS_DIR}/benchmark_comparison.png'
    risk_analyzer.create_benchmark_comparison_chart(benchmark_chart_path)
    print(f"   ✓ Gráfico de benchmark guardado: benchmark_comparison.png")
    print()
    
    # ========================================================================
    # PHASE 7: STRESS TESTING (NEW)
    # ========================================================================
    print("FASE 7: STRESS TESTING Y ESCENARIOS")
    print("-" * 80)
    
    print("⚠️  Ejecutando stress tests institucionales...")
    stress_results = risk_analyzer.perform_stress_tests()
    
    print(f"   ✓ {len(stress_results)} escenarios de stress evaluados")
    print(f"   • Market shocks: -10%, -20%, -30%")
    print(f"   • Volatility shocks: 1.5x, 2.0x, 2.5x")
    print(f"   • Tech sector shock: -25%")
    
    # Create stress test heatmap
    stress_chart_path = f'{CHARTS_DIR}/stress_test_heatmap.png'
    risk_analyzer.create_stress_test_heatmap(stress_results, stress_chart_path)
    print(f"   ✓ Heatmap de stress test guardado: stress_test_heatmap.png")
    print()
    
    # ========================================================================
    # PHASE 8: TAIL RISK ANALYSIS (NEW)
    # ========================================================================
    print("FASE 8: ANÁLISIS DE RIESGO DE COLA")
    print("-" * 80)
    
    print("📉 Calculando VaR y CVaR (método histórico)...")
    var_cvar = risk_analyzer.calculate_var_cvar([0.95, 0.99])
    
    print(f"   ✓ VaR 95%: {var_cvar['VaR_95']*100:.2f}%")
    print(f"   ✓ CVaR 95%: {var_cvar['CVaR_95']*100:.2f}%")
    print(f"   ✓ VaR 99%: {var_cvar['VaR_99']*100:.2f}%")
    print(f"   ✓ CVaR 99%: {var_cvar['CVaR_99']*100:.2f}%")
    
    # Create tail risk chart
    tail_risk_chart_path = f'{CHARTS_DIR}/tail_risk_analysis.png'
    risk_analyzer.create_tail_risk_chart(tail_risk_chart_path)
    print(f"   ✓ Gráfico de riesgo de cola guardado: tail_risk_analysis.png")
    print()
    
    # ========================================================================
    # PHASE 9: EXECUTION AND LIQUIDITY ANALYSIS (NEW)
    # ========================================================================
    print("FASE 9: ANÁLISIS DE EJECUCIÓN Y LIQUIDEZ")
    print("-" * 80)
    
    print("💰 Analizando costos de ejecución a diferentes niveles de AUM...")
    execution_analyzer = ExecutionLiquidityAnalysis(
        optimal_portfolio['weights'],
        analysis_tickers,
        base_transaction_cost=0.001
    )
    
    # Calculate execution costs at different AUM levels
    # Use analysis_capital as base and grow by 10x
    base_cap = analysis_capital
    aum_levels = [base_cap, base_cap*10, base_cap*100, base_cap*1000]
    execution_costs = execution_analyzer.calculate_execution_costs(aum_levels, slippage_bps=5.0)
    
    print(f"   ✓ Costos calculados para {len(aum_levels)} niveles de AUM")
    for aum_label, costs in execution_costs.items():
        print(f"   • {aum_label}: ${costs['total_cost']:.2f} ({costs['total_cost_bps']:.1f} bps)")
    
    # Create execution tables
    execution_table = execution_analyzer.create_execution_analysis_table(execution_costs)
    liquidity_table = execution_analyzer.create_liquidity_constraints_table()
    
    # Create scalability chart
    scalability_chart_path = f'{CHARTS_DIR}/aum_scalability.png'
    execution_analyzer.create_scalability_chart(execution_costs, scalability_chart_path)
    print(f"   ✓ Gráfico de escalabilidad guardado: aum_scalability.png")
    print()
    
    # ========================================================================
    # PHASE 10: GOVERNANCE FRAMEWORK (NEW)
    # ========================================================================
    print("FASE 10: FRAMEWORK DE GOBERNANZA")
    print("-" * 80)
    
    print("📋 Generando checklist de monitoreo y eventos gatillo...")
    governance = GovernanceFramework()
    monitoring_checklist = governance.get_monitoring_checklist()
    trigger_events = governance.get_trigger_events()
    
    monitoring_df = governance.create_governance_table(monitoring_checklist)
    triggers_df = governance.create_trigger_events_table(trigger_events)
    
    print(f"   ✓ {len(monitoring_checklist)} actividades de monitoreo definidas")
    print(f"   ✓ {len(trigger_events)} eventos gatillo especificados")
    print()
    
    # ========================================================================
    # PHASE 11: REBALANCING STRATEGY
    # ========================================================================
    print("FASE 11: ESTRATEGIA DE REBALANCEO")
    print("-" * 80)
    
    print("🔄 Analizando estrategias de rebalanceo...")
    rebalancer = RebalancingStrategy(
        portfolio_data.data[analysis_tickers],
        portfolio_data.log_returns[analysis_tickers],
        optimal_portfolio['weights'],
        analysis_tickers
    )
    
    comparison_df, strategies_results = rebalancer.compare_strategies(initial_capital=analysis_capital)
    
    # Recommended strategy for 1 year
    if HORIZON_YEARS == 1:
        recommended_strategy = 'Trimestral'
    else:
        recommended_strategy = 'Anual'
    
    rec_data = comparison_df[comparison_df['Estrategia'] == recommended_strategy].iloc[0]
    
    print(f"   ✓ Estrategia recomendada: {recommended_strategy}")
    print(f"   • Retorno esperado: {rec_data['Retorno Neto (%)']:.2f}%")
    print(f"   • Número de rebalanceos: {int(rec_data['Num. Rebalanceos'])}")
    print(f"   • Costos de transacción: ${rec_data['Costos Transacción ($)']:.0f}")
    
    rebalancing_chart_path = f'{CHARTS_DIR}/rebalancing_strategies.png'
    rebalancer.create_monitoring_dashboard(strategies_results, rebalancing_chart_path)
    print(f"   ✓ Dashboard guardado: rebalancing_strategies.png")
    print()
    
    # ========================================================================
    # PHASE 12: EFFICIENT FRONTIER VISUALIZATION
    # ========================================================================
    print("FASE 12: VISUALIZACIÓN DE FRONTERA EFICIENTE")
    print("-" * 80)
    
    print("🎨 Creando gráfico de frontera eficiente...")
    from quant.visualization.plotter import EfficientFrontierVisualizer
    
    visualizer = EfficientFrontierVisualizer(
        analysis_tickers,
        mc_simulator.results,
        (efficient_vols, efficient_returns),
        (individual_vols[:-1], individual_sharpe[:-1], portfolio_data.mean_returns[analysis_tickers]),
        markowitz_portfolios,
        mc_portfolios
    )
    
    efficient_frontier_path = f'{CHARTS_DIR}/efficient_frontier.png'
    visualizer.create_plot(efficient_frontier_path, analysis_sims)
    print(f"   ✓ Frontera eficiente guardada: efficient_frontier.png")
    print()
    
    # ========================================================================
    # PHASE 13: GENERATE EXTENDED PROFESSIONAL PDF REPORT
    # ========================================================================
    print("FASE 13: GENERACIÓN DE INFORME EJECUTIVO PDF EXTENDIDO")
    print("-" * 80)
    
    print("📄 Compilando informe profesional completo con secciones avanzadas...")
    print("   (Incluyendo benchmarking, stress testing, tail risk, limitaciones,")
    print("    ejecución/liquidez, y gobernanza...)")
    
    # Prepare data for complete report
    portfolio_summary = {
        'tickers': analysis_tickers,
        'expected_return': optimal_portfolio['return'],
        'volatility': optimal_portfolio['volatility'],
        'sharpe_ratio': optimal_portfolio['sharpe'],
        'benchmark_return': portfolio_data.mean_returns[analysis_benchmark],
        'num_clusters': len(cluster_info)
    }
    
    rebalancing_summary = {
        'strategy': recommended_strategy,
        'frequency': 4 if recommended_strategy == 'Trimestral' else (12 if recommended_strategy == 'Mensual' else 1),  # Annual frequency
        'total_rebalances_3yr': int(rec_data['Num. Rebalanceos']),  # Total over 3 years
        'expected_return': rec_data['Retorno Neto (%)'],
        'costs': rec_data['Costos Transacción ($)']
    }
    
    # Generate extended report with all sections
    # Generate extended report with all sections
    from quant.reporting.pdf_builder import CompleteProfessionalReport
    from quant.reporting.components import ExtendedReportSections
    
    report_path = f'{CLIENT_DIR}/INFORME_EJECUTIVO_PROFESIONAL.pdf'
    
    # Create extended report (this will be updated to include new sections)
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from quant.reporting.pdf_builder import NumberedCanvas
    
    doc = SimpleDocTemplate(
        report_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Build complete story with all sections
    base_report = CompleteProfessionalReport(report_path, CHARTS_DIR, INDIVIDUAL_DIR)
    extended_sections = ExtendedReportSections(base_report.styles, CHARTS_DIR)
    
    story = []
    
    # Original sections
    story.extend(base_report._create_cover_page())
    story.extend(base_report._create_toc())
    story.extend(base_report._create_executive_summary(portfolio_summary, backtest_results, rebalancing_summary))
    story.extend(base_report._create_optimization_section(portfolio_summary))
    story.extend(base_report._create_backtest_section(backtest_results))
    
    # NEW SECTION 1: Benchmarking
    story.extend(extended_sections.create_benchmarking_section(risk_metrics))
    
    # Continue with original sections
    story.extend(base_report._create_clustering_section(cluster_info))
    story.extend(base_report._create_individual_assets_section(stats_table, analysis_tickers))
    story.extend(base_report._create_rebalancing_section(rebalancing_summary))
    
    # NEW SECTION 2: Stress Testing
    story.extend(extended_sections.create_stress_testing_section(stress_results))
    
    # NEW SECTION 3: Tail Risk
    story.extend(extended_sections.create_tail_risk_section(var_cvar))
    
    # NEW SECTION 4: Model Limitations
    story.extend(extended_sections.create_model_limitations_section())
    
    # NEW SECTION 5: Execution and Liquidity
    story.extend(extended_sections.create_execution_liquidity_section(execution_table, liquidity_table))
    
    # NEW SECTION 6: Governance
    story.extend(extended_sections.create_governance_section(monitoring_df, triggers_df))
    
    # NEW SECTION 7: Data & Methodology
    story.extend(extended_sections.create_data_methodology_section())
    
    # Original sections
    story.extend(base_report._create_mathematical_framework())
    story.extend(base_report._create_conclusions(portfolio_summary, backtest_results))
    
    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    print(f"   ✓ Informe extendido generado: INFORME_EJECUTIVO_PROFESIONAL.pdf")
    print(f"   ✓ Secciones totales: 14 (8 originales + 6 nuevas)")
    print()
    
    # ========================================================================
    # PHASE 14: GENERATE SUMMARY DOCUMENT
    # ========================================================================
    print("FASE 14: DOCUMENTACIÓN Y RESUMEN")
    print("-" * 80)
    
    # Create comprehensive README
    readme_content = f"""# INFORME CUANTITATIVO PROFESIONAL EXTENDIDO

## Generado: {datetime.now().strftime('%d de %B de %Y, %H:%M')}

---

## 🆕 NUEVAS SECCIONES AGREGADAS

1. **Benchmarking y Performance Relativa**
   - Alpha: {risk_metrics['alpha_annual']*100:.2f}%
   - Beta: {risk_metrics['beta']:.3f}
   - Tracking Error: {risk_metrics['tracking_error']*100:.2f}%
   - Information Ratio: {risk_metrics['information_ratio']:.3f}

2. **Stress Testing y Escenarios**
   - {len(stress_results)} escenarios evaluados
   - Market shocks, volatility shocks, sector shocks

3. **Riesgo de Cola y Downside**
   - VaR 95%: {var_cvar['VaR_95']*100:.2f}%
   - CVaR 95%: {var_cvar['CVaR_95']*100:.2f}%
   - VaR 99%: {var_cvar['VaR_99']*100:.2f}%

4. **Limitaciones del Modelo**
   - Supuestos y riesgos de modelización
   - Estrategias de mitigación

5. **Ejecución y Liquidez**
   - Análisis de escalabilidad (4 niveles de AUM)
   - Slippage y market impact

6. **Gobernanza y Monitoreo**
   - {len(monitoring_checklist)} actividades de monitoreo
   - {len(trigger_events)} eventos gatillo

---

## 📊 PORTFOLIO ÓPTIMO

- **Sharpe Ratio:** {optimal_portfolio['sharpe']:.4f}
- **Retorno Anual:** {optimal_portfolio['return']*100:.2f}%
- **Volatilidad:** {optimal_portfolio['volatility']*100:.2f}%
- **Backtest (3 años):** +{backtest_results['total_return']:.2f}%

---

## 📁 GRÁFICOS GENERADOS

**Total: {len(os.listdir(CHARTS_DIR))} visualizaciones**

1. efficient_frontier.png
2. backtest_results.png
3. clustering_analysis.png
4. rebalancing_strategies.png
5. benchmark_comparison.png ⭐ NUEVO
6. stress_test_heatmap.png ⭐ NUEVO
7. tail_risk_analysis.png ⭐ NUEVO
8. aum_scalability.png ⭐ NUEVO

Plus {len(asset_chart_paths)} análisis individuales de activos.

---

**Confidencial** - Análisis Cuantitativo Profesional Extendido

**Nivel:** JP Morgan Senior Quant + Advanced Risk Management
"""
    
    readme_path = f'{CLIENT_DIR}/README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"   ✓ README.md generado")
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("="*80)
    print("✅ GENERACIÓN COMPLETADA EXITOSAMENTE")
    print("="*80)
    print()
    print(f"📁 Ubicación: {CLIENT_DIR}/")
    print()
    print("📄 Archivos generados:")
    print(f"   • INFORME_EJECUTIVO_PROFESIONAL.pdf - Documento principal EXTENDIDO")
    print(f"   • README.md - Resumen y guía")
    print(f"   • charts/ - {len(os.listdir(CHARTS_DIR))} visualizaciones (4 NUEVAS)")
    print(f"   • individual_assets/ - {len(asset_chart_paths)} análisis individuales")
    print()
    print("📊 Estadísticas del análisis:")
    print(f"   • Activos analizados: {len(analysis_tickers)}")
    print(f"   • Portfolios simulados: {analysis_sims:,}")
    print(f"   • Clusters identificados: {len(cluster_info)}")
    print(f"   • Sharpe Ratio óptimo: {optimal_portfolio['sharpe']:.4f}")
    print(f"   • Alpha vs benchmark: {risk_metrics['alpha_annual']*100:.2f}%")
    print(f"   • Information Ratio: {risk_metrics['information_ratio']:.3f}")
    print()
    print("🆕 Nuevas secciones incluidas:")
    print("   • Benchmarking y Performance Relativa")
    print("   • Stress Testing y Escenarios")
    print("   • Riesgo de Cola (VaR/CVaR)")
    print("   • Limitaciones del Modelo")
    print("   • Ejecución y Liquidez")
    print("   • Gobernanza y Monitoreo")
    print()
    print("="*80)
    print("INFORME EXTENDIDO LISTO PARA PRESENTACIÓN AL CLIENTE")
    print("="*80)


if __name__ == "__main__":
    main()

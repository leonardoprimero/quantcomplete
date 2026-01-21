"""
Efficient Frontier Analysis - Main Script
Orchestrates portfolio optimization using Monte Carlo simulation and Markowitz optimization.

Usage:
    python efficient_frontier_mc.py
"""

from portfolio_data import PortfolioData
from monte_carlo import MonteCarloSimulator
from markowitz import MarkowitzOptimizer
from printer import ResultsPrinter
from visualizer import EfficientFrontierVisualizer
from report_generator import PortfolioReportGenerator
from asset_analyzer import AssetAnalyzer
from backtester import PortfolioBacktester
import os


def main():
    """Main execution function."""
    # Configuration - 10+ activos incluyendo SPY como benchmark
    TICKERS = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'NVDA',   # NVIDIA
        'META',   # Meta (Facebook)
        'TSLA',   # Tesla
        'JPM',    # JPMorgan Chase
        'V',      # Visa
        'WMT',    # Walmart
        'SPY'     # S&P 500 ETF (Benchmark)
    ]
    
    BENCHMARK = 'SPY'
    NUM_PORTFOLIOS = 50000
    BASE_DIR = '/Users/leguillo/prueba antigravity/quant'
    CHART_PATH = f'{BASE_DIR}/efficient_frontier_mc.png'
    REPORT_PATH = f'{BASE_DIR}/informe_portfolio.pdf'
    BACKTEST_PATH = f'{BASE_DIR}/backtest_portfolio.png'
    
    # Create directory for individual asset charts
    ASSETS_DIR = f'{BASE_DIR}/analisis_activos'
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    print("="*70)
    print("ANÁLISIS CUANTITATIVO DE PORTFOLIO")
    print("="*70)
    print(f"Activos: {len(TICKERS)}")
    print(f"Benchmark: {BENCHMARK}")
    print("="*70 + "\n")
    
    # 1. Download and prepare data
    print("📊 Descargando datos históricos...")
    portfolio_data = PortfolioData(TICKERS, years=3)
    portfolio_data.download_data()
    portfolio_data.calculate_returns()
    individual_vols, individual_sharpe = portfolio_data.get_individual_stats()
    
    # 2. Analyze individual assets
    print("📈 Analizando activos individuales...")
    asset_analyzer = AssetAnalyzer(portfolio_data.data, TICKERS, BENCHMARK)
    asset_chart_paths = asset_analyzer.create_individual_analysis_charts(ASSETS_DIR)
    stats_table = asset_analyzer.get_statistics_table()
    print(f"   ✓ Generados {len(asset_chart_paths)} gráficos de análisis individual")
    
    # 3. Run Monte Carlo simulation (excluding benchmark)
    print("🎲 Ejecutando simulación Monte Carlo...")
    active_tickers = [t for t in TICKERS if t != BENCHMARK]
    mc_simulator = MonteCarloSimulator(
        portfolio_data.mean_returns[active_tickers], 
        portfolio_data.cov_matrix.loc[active_tickers, active_tickers], 
        NUM_PORTFOLIOS
    )
    mc_simulator.run_simulation()
    mc_portfolios = mc_simulator.get_optimal_portfolios()
    print(f"   ✓ Simulados {NUM_PORTFOLIOS:,} portfolios")
    
    # 4. Perform Markowitz optimization
    print("⚡ Optimizando con Markowitz...")
    optimizer = MarkowitzOptimizer(
        portfolio_data.mean_returns[active_tickers],
        portfolio_data.cov_matrix.loc[active_tickers, active_tickers]
    )
    markowitz_portfolios = {
        'max_sharpe': optimizer.optimize_max_sharpe(),
        'min_vol': optimizer.optimize_min_volatility()
    }
    efficient_vols, efficient_returns = optimizer.generate_efficient_frontier()
    print("   ✓ Portfolios óptimos identificados")
    
    # 5. Backtest optimal portfolio
    print("🔬 Ejecutando backtesting...")
    backtester = PortfolioBacktester(portfolio_data.data, portfolio_data.log_returns)
    backtest_results = backtester.backtest_portfolio(
        markowitz_portfolios['max_sharpe']['weights'],
        active_tickers,
        initial_capital=10000
    )
    backtester.create_backtest_chart(backtest_results, BENCHMARK, BACKTEST_PATH,
                                    "Portfolio Máximo Sharpe")
    print(f"   ✓ Retorno total: {backtest_results['total_return']:.2f}%")
    print(f"   ✓ Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    
    # 6. Print results
    print("\n" + "="*70)
    print("RESULTADOS DEL ANÁLISIS")
    print("="*70)
    printer = ResultsPrinter()
    printer.print_header(active_tickers, NUM_PORTFOLIOS)
    printer.print_individual_assets(active_tickers, 
                                   portfolio_data.mean_returns[active_tickers], 
                                   individual_vols[:-1],  # Exclude benchmark
                                   individual_sharpe[:-1])
    printer.print_portfolio("MARKOWITZ - PORTFOLIO MÁXIMO SHARPE RATIO",
                          markowitz_portfolios['max_sharpe'], active_tickers)
    printer.print_portfolio("MARKOWITZ - PORTFOLIO MÍNIMA VOLATILIDAD",
                          markowitz_portfolios['min_vol'], active_tickers)
    printer.print_footer()
    
    # 7. Create visualization
    print("🎨 Creando visualización de frontera eficiente...")
    visualizer = EfficientFrontierVisualizer(
        active_tickers, 
        mc_simulator.results,
        (efficient_vols, efficient_returns),
        (individual_vols[:-1], individual_sharpe[:-1], 
         portfolio_data.mean_returns[active_tickers]),
        markowitz_portfolios,
        mc_portfolios
    )
    visualizer.create_plot(CHART_PATH, NUM_PORTFOLIOS)
    
    # 8. Generate comprehensive PDF report
    print("📄 Generando informe completo en PDF...")
    report_generator = PortfolioReportGenerator(
        active_tickers,
        portfolio_data.mean_returns[active_tickers],
        portfolio_data.cov_matrix.loc[active_tickers, active_tickers],
        (individual_vols[:-1], individual_sharpe[:-1]),
        markowitz_portfolios,
        mc_portfolios,
        stats_table,
        backtest_results,
        asset_chart_paths,
        BENCHMARK
    )
    report_generator.generate_report(REPORT_PATH, CHART_PATH, BACKTEST_PATH)
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"📊 Gráfico frontera eficiente: {CHART_PATH}")
    print(f"📈 Gráfico backtesting: {BACKTEST_PATH}")
    print(f"📁 Análisis individual: {ASSETS_DIR}/ ({len(asset_chart_paths)} archivos)")
    print(f"📄 Informe completo (PDF): {REPORT_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()

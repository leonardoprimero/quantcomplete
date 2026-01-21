"""
Global Configuration
Centralized configuration for the quantitative analysis engine.
"""

# Tickers to analyze
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
    'META', 'TSLA', 'JPM', 'V', 'WMT'
]

# Market Data Parameters
START_DATE = '2023-01-20'  # 3 years of history
END_DATE = '2026-01-20'
RISK_FREE_RATE = 0.0  # Conservative approach

# Portfolio Optimization
INITIAL_CAPITAL = 10000
NUM_PORTFOLIOS = 50000  # Monte Carlo simulations

# Report Configuration
OUTPUT_DIR = 'quant_client_report'
CLIENT_DIR = OUTPUT_DIR
CHARTS_DIR = f'{OUTPUT_DIR}/charts'
INDIVIDUAL_DIR = f'{OUTPUT_DIR}/individual_assets'

# Strategy Parameters
BENCHMARK = 'SPY'
HORIZON_YEARS = 1

# Visualization
INSTITUTIONAL_COLORS = {
    'primary': '#1a5490',
    'secondary': '#4a4a4a',
    'accent': '#d4af37',
    'success': '#2e7d32',
    'warning': '#f57c00',
    'danger': '#c62828'
}

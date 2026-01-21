# Quant Complete

**Professional Quantitative Analysis Engine**

Quant Complete is a comprehensive Python-based framework designed for institutional-grade portfolio analysis. It combines advanced statistical methods, modern portfolio theory, and rigorous risk management metrics to deliver actionable investment insights.

## 🚀 Features

- **Data Acquisition**: Automated historical data downloading from Yahoo Finance.
- **Portfolio Optimization**:
  - **Markowitz Efficiency**: Calculation of the Efficient Frontier.
  - **Monte Carlo Simulation**: 50,000+ scenarios to identify optimal asset weights.
- **Advanced Analytics**:
  - **Clustering**: Hierarchical clustering to identify asset correlations.
  - **Risk Metrics**: VaR (Value at Risk), CVaR (Conditional Value at Risk), Alpha, Beta, Tracking Error, and Information Ratio.
  - **Stress Testing**: Simulation of market shocks (e.g., -20% market drop, volatility spikes).
- **Execution & Liquidity**: Analysis of transaction costs, slippage, and AUM scalability.
- **Professional Reporting**: Generates a detailed PDF report (`INFORME_EJECUTIVO_PROFESIONAL.pdf`) and high-quality visualizations.

## 🛠 Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/leonardoprimero/quantcomplete.git
    cd quantcomplete
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    _(Note: Ensure you have `pandas`, `numpy`, `yfinance`, `matplotlib`, `seaborn`, `scipy`, `sklearn`, and `reportlab` installed)_

## 🖥 Usage

### Interactive Mode (Recommended)

Run the interactive script to customize your analysis parameters (Assets, Benchmark, Date Range, Capital, etc.):

```bash
python3 run_analysis.py
```

### Quick Start

You can also configure defaults in `quant/config.py` and run the main module directly:

```bash
python3 -m quant.main
```

## 📊 Output

All results are generated in the `quant_client_report/` directory:

- **PDF Report**: A complete executive summary.
- **Charts**: Individual and aggregated portfolio visualizations.
- **Data**: Statistical tables and metrics.

---

Leonardo I (a.k.a. @leonardoprimero)

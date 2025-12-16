# Quant App

A quantitative trading application for real-time pairs trading analysis and hedge ratio estimation. Built with Streamlit, this app fetches tick data from a database, computes hedge ratios using various regression methods (OLS, TLS, Huber, Theil-Sen, Kalman Filter, Non-linear), and provides interactive visualizations for spread analysis, z-score monitoring, rolling correlations, and backtesting mean-reversion strategies.

## Features

- Real-time data ingestion and preparation
- Multiple regression techniques for hedge ratio calculation
- Spread and z-score analysis with alerts
- Interactive charts for prices, spreads, correlations, and volume
- Stationarity testing (ADF test) on spreads
- Mini backtest for mean-reversion trading
- Time-series statistics and data export
- Alter for filtering range for zscore and spread

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd quant_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

- Select two symbols and a timeframe for analysis.
- Choose a regression type for hedge ratio estimation.
- Monitor the spread, z-score, and set alerts.
- View charts and perform backtests.

## Requirements

- Python 3.11+
- Streamlit
- Pandas
- Statsmodels
- PyKalman
- Scikit-learn
- Plotly


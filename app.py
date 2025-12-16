import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from db import fetch_ticks
from analytics import *
from ingest import start_ingestion
from config import SYMBOLS

st.set_page_config(layout="wide")
st.title("📈 Real‑Time Quant Analytics Dashboard")

# Sidebar for alerts
st.sidebar.title("Alerts")
alert_enabled = st.sidebar.checkbox("Enable Alerts")
if alert_enabled:
    alert_condition = st.sidebar.selectbox("Condition", ["Z-Score >", "Z-Score <", "Spread >", "Spread <"])
    alert_threshold = st.sidebar.number_input("Threshold", value=2.0)

# Start ingestion once
if 'started' not in st.session_state:
    start_ingestion()
    st.session_state.started = True

# Controls
index_a = SYMBOLS.index('btcusdt') if 'btcusdt' in SYMBOLS else 0
index_b = SYMBOLS.index('ethusdt') if 'ethusdt' in SYMBOLS else 1
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    sym_a = st.selectbox("Symbol A", SYMBOLS, index=index_a)
with col2:
    sym_b = st.selectbox("Symbol B", SYMBOLS, index=index_b)
with col3:
    timeframe = st.selectbox("Timeframe", ['1s', '1m', '5m'])
with col4:
    window = st.slider("Rolling Window", 10, 100, 30)
with col5:
    reg_type = st.selectbox("Regression Type", ['OLS', 'TLS', 'Huber', 'Theil-Sen', 'Non-linear', 'Kalman'])

# Fetch and prepare data
df_a = prepare_df(fetch_ticks(sym_a), freq='1s')
df_b = prepare_df(fetch_ticks(sym_b), freq='1s')

if df_a.empty or df_b.empty:
    st.error("Invalid pair or no data available for the selected symbols.")
elif len(df_a) > 10 and len(df_b) > 10:
    # Resample
    a = resample_df(df_a, timeframe)
    b = resample_df(df_b, timeframe)

    # Compute analytics
    beta = hedge_ratio(a['price'], b['price'], reg_type)
    if pd.isna(beta):
        st.error("No overlapping data for the selected symbols and timeframe.")
    else:
        spread, z = spread_zscore(a['price'], b['price'], beta)
        corr = rolling_corr(a['price'], b['price'], window)

        # Check alerts
        if alert_enabled and not z.empty and not z.isna().all():
            z_last = z.iloc[-1]
            spread_last = spread.iloc[-1]

            if alert_condition == "Z-Score >" and z_last > alert_threshold:
                st.error(f"🚨 Alert: Z-Score {z_last:.2f} > {alert_threshold}")
            elif alert_condition == "Z-Score <" and z_last < alert_threshold:
                st.error(f"🚨 Alert: Z-Score {z_last:.2f} < {alert_threshold}")
            elif alert_condition == "Spread >" and spread_last > alert_threshold:
                st.error(f"🚨 Alert: Spread {spread_last:.2f} > {alert_threshold}")
            elif alert_condition == "Spread <" and spread_last < alert_threshold:
                st.error(f"🚨 Alert: Spread {spread_last:.2f} < {alert_threshold}")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last Price A", round(a['price'].iloc[-1], 2))
        m2.metric("Last Price B", round(b['price'].iloc[-1], 2))
        m3.metric("Hedge Ratio", round(beta, 4))
        if z.empty or z.isna().all():
            m4.metric("Z-Score", "N/A")
        else:
            m4.metric("Z-Score", round(z.iloc[-1], 2))

        # Prices chart
        st.subheader("Prices")
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(x=a.index, y=a['price'], name=sym_a))
        fig_prices.add_trace(go.Scatter(x=b.index, y=b['price'], name=sym_b))
        st.plotly_chart(fig_prices, use_container_width=True)

        # Spread & Z-score chart
        st.subheader("Spread & Z-Score")
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
        fig_spread.add_trace(go.Scatter(x=z.index, y=z, name="Z-Score"))
        st.plotly_chart(fig_spread, use_container_width=True)

        # Rolling correlation chart
        st.subheader(f"Rolling Correlation ({window} periods)")
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=corr.index, y=corr, name="Rolling Corr"))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Volume chart
        st.subheader("Volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=a.index, y=a['size'], name=sym_a))
        fig_vol.add_trace(go.Bar(x=b.index, y=b['size'], name=sym_b))
        st.plotly_chart(fig_vol, use_container_width=True)

        # ADF test trigger
        adf_trigger = st.checkbox("Run ADF Test on Spread", value=True)

        # Hedge ratio & ADF test info
        st.subheader("Hedge Ratio & Spread Stationarity")
        st.write(f"Hedge Ratio (β): {beta:.4f}")
        if adf_trigger:
            p_val, is_stationary = adf_test(spread)

            if p_val is None:
                st.warning("ADF test not applicable (constant or insufficient data)")
            else:
                st.write(f"Spread stationary: {is_stationary}, p-value={p_val:.4f}")
        # Backtest
        st.subheader("Mini Mean-Reversion Backtest")
        trades, total_pnl = backtest(spread, z)
        st.write(f"Total PnL: {total_pnl:.4f}")
        if trades:
            st.table(pd.DataFrame(trades))
        else:
            st.write("No trades triggered.")

        # Time-series Stats Table
        st.subheader("Time-Series Stats")
        # Resample to 1m for table
        stats_df = a.resample('1min').agg({'price': 'last', 'size': 'sum'}).dropna()
        stats_df['spread'] = spread.resample('1min').last()
        stats_df['z_score'] = z.resample('1min').last()
        stats_df['correlation'] = corr.resample('1min').last()
        stats_df = stats_df.reset_index()
        st.dataframe(stats_df)
        csv = stats_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "stats.csv", "text/csv")

        # Z-Score Distribution
        st.subheader("Z-Score Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=z, nbinsx=50, name="Z-Score"))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        corr_matrix = pd.DataFrame({'A': a['price'], 'B': b['price']}).corr()
        st.dataframe(corr_matrix)

else:
    st.warning("Waiting for data…")

# Live update every 500ms
time.sleep(0.5)
st.rerun()

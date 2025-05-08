import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Markowitz Portfolio Optimization Tool", layout="wide")
st.title("\U0001F4C8 Markowitz Portfolio Optimization Tool")

# === Step 1: User Inputs ===
st.subheader("Select Assets")

tickers_input = st.text_input("Enter Yahoo Finance tickers separated by commas (e.g., INFY.NS, TCS.NS, RELIANCE.NS):")
sectors_input = st.text_area("Enter corresponding sectors separated by commas (same order):")
opt_start = st.date_input("Optimization Start Date", value=datetime(2019, 4, 1))
opt_end = st.date_input("Optimization End Date", value=datetime(2022, 3, 31))
backtest_start = st.date_input("Backtest Start Date", value=datetime(2022, 4, 1))
backtest_end = st.date_input("Backtest End Date", value=datetime(2025, 3, 31))
initial_investment = st.number_input("Initial Investment (INR)", min_value=1000, value=100000)

if st.button("Run Optimization"):
    try:
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        sectors = [s.strip() for s in sectors_input.split(",")]
        if len(tickers) != len(sectors):
            st.error("Number of tickers and sectors must match.")
            st.stop()

        ticker_sector_map = dict(zip(tickers, sectors))

        # === Step 2: Fetch Data ===
        st.info("Fetching data from Yahoo Finance...")
        data = yf.download(tickers, start=opt_start, end=opt_end)

        # Fix: Extract only 'Close' prices
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']

        returns = data.pct_change().dropna()

        # === Step 3: Mean-Variance Optimization ===
        mu = returns.mean()
        cov = returns.cov()
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(len(mu))
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)

        # Normalize weights
        weights = weights / weights.sum()

        weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": weights,
            "Sector": [ticker_sector_map[t] for t in tickers]
        })

        st.success("\u2705 Optimization complete!")
        st.dataframe(weights_df)

        # === Step 4: Plot Sector Exposure ===
        st.subheader("Sector-wise Allocation")
        sector_allocation = weights_df.groupby("Sector")["Weight"].sum()
        fig, ax = plt.subplots()
        sector_allocation.plot(kind="bar", ax=ax, color="teal")
        ax.set_ylabel("Weight")
        ax.set_title("Portfolio Sector Exposure")
        st.pyplot(fig)

        # === Step 5: Backtest (optional enhancement) ===
        st.subheader("Portfolio Backtest")
        backtest_data = yf.download(tickers, start=backtest_start, end=backtest_end)['Close']
        backtest_returns = backtest_data.pct_change().dropna()

        portfolio_returns = (backtest_returns * weights).sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod() * initial_investment
        final_value = portfolio_value.iloc[-1]
        st.subheader("📊 Final Portfolio Value")
        st.metric(label="Final Value (INR)", value=f"₹{final_value:,.2f}")

        fig2, ax2 = plt.subplots()
        portfolio_value.plot(ax=ax2, color="orange")
        ax2.set_title("Portfolio Value Over Time")
        ax2.set_ylabel("Portfolio Value (INR)")
        st.pyplot(fig2)

        # Download option
        st.download_button("Download Weights as CSV", weights_df.to_csv(index=False), "weights.csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
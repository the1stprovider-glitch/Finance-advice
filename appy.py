# ================================================================
# 📈 ADVANCED TRADING ADVICE + SIGNAL SYSTEM (Optimized & Fixed)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Advanced Trading Advisor", layout="wide")
st.title("📊 Advanced Trading Advice & Prediction Engine")

# ---------------- CACHING for Performance ----------------

@st.cache_data
def get_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

@st.cache_data
def compute_indicators(df):
    # Simple Moving Averages
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=50).mean()

    # Exponential Moving Averages
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD and Signal
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (safe 1-D pandas approach)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_models(df):
    X = df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]]
    y = df["Close"].shift(-1).fillna(method="ffill")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    return rf_model, xgb_model

# ---------------- SIGNAL GENERATION ----------------

def generate_signal(current, predicted):
    diff = float(predicted - current)
    pct = (diff / float(current)) * 100

    if pct > 4:
        return "STRONG BUY", pct
    elif pct > 2:
        return "BUY", pct
    elif pct < -4:
        return "STRONG SELL", pct
    elif pct < -2:
        return "SELL", pct
    else:
        return "HOLD", pct

# ---------------- USER INTERFACE ----------------

st.sidebar.header("⚙️ Configurations")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")

if st.sidebar.button("Analyze"):
    with st.spinner("Fetching data and running models..."):

        data = get_stock_data(ticker, start_date, end_date)

        if data.empty:
            st.error("❌ No data found. Try another ticker.")
        else:
            df = compute_indicators(data)
            rf_model, xgb_model = train_models(df)

            latest = df.iloc[-1]
            current_price = latest["Close"]

            features = latest[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]]
            pred_rf = float(rf_model.predict([features])[0])
            pred_xgb = float(xgb_model.predict([features])[0])
            combined = (pred_rf + pred_xgb) / 2

            signal, confidence = generate_signal(current_price, combined)

            # Safe Display of Metrics
            st.write(f"**Current Close Price:** ${current_price:,.2f}")
            st.write(f"**RandomForest Prediction:** ${pred_rf:,.2f}")
            st.write(f"**XGBoost Prediction:** ${pred_xgb:,.2f}")
            st.write(f"**Combined Forecast:** ${combined:,.2f}")
            st.write(f"**Trade Signal:** {signal} ({confidence:.2f}% forecast change)")

            # ==== Simple Backtest Summary ====
            df["Predicted"] = (
                rf_model.predict(df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]]) +
                xgb_model.predict(df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]])
            ) / 2

            df["StrategyReturn"] = np.where(df["Predicted"] > df["Close"],
                                             df["Close"].pct_change(),
                                             -df["Close"].pct_change())

            backtest_summary = {
                "TotalReturn": f"{(df['StrategyReturn'] + 1).prod() - 1:.2%}",
                "AvgDailyReturn": f"{df['StrategyReturn'].mean():.4%}",
                "Volatility": f"{df['StrategyReturn'].std():.4%}"
            }

            st.write("📊 Simple Strategy Backtest Summary:", backtest_summary)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Analysis complete!")

st.caption("*Powered by Yahoo Finance & ensemble ML models — not financial advice.*")

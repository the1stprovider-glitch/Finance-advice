# ================================================================
# 📈 ADVANCED TRADING ADVICE & PREDICTION SYSTEM (ALL-IN-ONE)
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

# ------------ STREAMLIT BASIC CONFIG ------------
st.set_page_config(page_title="Advanced Trading Advisor", layout="wide")
st.title("📊 Advanced Trading Advice & Prediction Engine")

# ------------ TECHNICAL INDICATORS ------------
def compute_indicators(df):
    """
    Compute a set of technical indicators:
    - SMA (20 & 50)
    - EMA (12 & 26)
    - MACD & signal line
    - RSI (correct 1-D implementation)
    """
    # Simple Moving Averages
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=50).mean()

    # Exponential Moving Averages
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI: Uses exponential smoothing approach to avoid 2-D error
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Drop early rows where values are NaN
    df.dropna(inplace=True)
    return df

# ------------ MACHINE LEARNING MODEL TRAINING ------------
def train_models(df):
    """
    Train an ensemble of ML models on engineered features.
    """
    features = df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]]
    labels = df["Close"].shift(-1).fillna(method="ffill")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb_model = XGBRegressor(n_estimators=400, learning_rate=0.05, subsample=0.8, random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model

def predict_next(rf, xgb, feature_row):
    pred_rf = rf.predict(feature_row.values.reshape(1, -1))[0]
    pred_xgb = xgb.predict(feature_row.values.reshape(1, -1))[0]
    combined = (pred_rf + pred_xgb) / 2
    return pred_rf, pred_xgb, combined

# ------------ SIGNAL GENERATION LOGIC ------------
def generate_signal(current, predicted):
    diff = predicted - current
    pct = (diff / current) * 100

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

# ------------ USER INTERFACE ------------
st.sidebar.header("📌 Config Options")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")

if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching market data and running models..."):
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error("❌ No market data found for that ticker.")
        else:
            df = compute_indicators(data)

            # Train models
            rf_model, xgb_model = train_models(df)

            # Latest features
            latest = df.iloc[-1]
            current_price = latest["Close"]
            pred_rf, pred_xgb, combined_pred = predict_next(rf_model, xgb_model, latest[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]])

            signal, confidence = generate_signal(current_price, combined_pred)

            # Display metrics
            st.metric("📊 Current Price", f"${current_price:,.2f}")
            st.metric("📈 RF Prediction", f"${pred_rf:,.2f}")
            st.metric("📈 XGB Prediction", f"${pred_xgb:,.2f}")
            st.metric("📈 Combined Prediction", f"${combined_pred:,.2f}")
            st.metric("💡 Trade Signal", f"{signal} ({confidence:.2f}%)")

            # Basic backtest summary
            df["PredictedNext"] = df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]].apply(
                lambda row: (rf_model.predict(row.values.reshape(1, -1))[0] + 
                             xgb_model.predict(row.values.reshape(1, -1))[0]) / 2, axis=1)

            df["StrategyReturn"] = np.where(df["PredictedNext"] > df["Close"],
                                            df["Close"].pct_change(),
                                            -df["Close"].pct_change())

            backtest_results = {
                "TotalReturn": f"{(df['StrategyReturn'] + 1).prod() - 1:.2%}",
                "AvgDailyReturn": f"{df['StrategyReturn'].mean():.4%}",
                "Volatility": f"{df['StrategyReturn'].std():.4%}"
            }
            st.write("📊 Simple Strategy Backtest Results", backtest_results)

            # Plot price + indicators
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Analysis complete!")

st.caption("📉 Powered by Yahoo Finance and ML models. This is a demo and not financial advice.")

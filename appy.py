# ================================================================
# 📊 SUPER COMPLEX TRADING ADVICE SYSTEM — STREAMLIT DASHBOARD
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Advanced Trading Advisor", layout="wide")
st.title("📈 Advanced Trading Advice & Prediction System")

# ========== Fetch Data ==========

def load_price_data(ticker, period="2y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

# ========== Feature Engineering ==========

def compute_technical_indicators(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

# ========== Models & Training ==========

def train_models(df):
    X = df[["MA20", "MA50", "RSI"]].values
    y = df["Close"].shift(-1).fillna(method="ffill").values

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return rf, xgb, X_test, y_test

# ========== Prediction Logic ==========

def make_prediction(models, latest_features):
    rf, xgb = models
    pred_rf = rf.predict(latest_features.reshape(1, -1))[0]
    pred_xgb = xgb.predict(latest_features.reshape(1, -1))[0]
    combined_pred = (pred_rf + pred_xgb) / 2
    return pred_rf, pred_xgb, combined_pred

# ========== Advice Logic ==========

def generate_signal(current_price, predicted_price):
    diff = predicted_price - current_price
    pct = (diff / current_price) * 100

    if pct > 3:
        return "STRONG BUY", pct
    elif pct > 1:
        return "BUY", pct
    elif pct < -3:
        return "STRONG SELL", pct
    elif pct < -1:
        return "SELL", pct
    else:
        return "HOLD", pct

# ========== Streamlit UI ==========

ticker = st.text_input("📌 Enter Ticker Symbol (e.g., AAPL)", "AAPL")
period = st.selectbox("📆 Data Period", ["1y","2y","5y"])
interval = st.selectbox("⏱ Interval", ["1d","1wk","1mo"])

if st.button("Analyze"):
    with st.spinner("⏳ Fetching and computing..."):
        data = load_price_data(ticker, period, interval)
        if data.empty:
            st.warning("No data available for this ticker!")
        else:
            df = compute_technical_indicators(data)
            models = train_models(df)
            latest_features = df[["MA20", "MA50", "RSI"]].iloc[-1].values

            # Predictions
            pred_rf, pred_xgb, combined = make_prediction(models, latest_features)
            current_price = df["Close"].iloc[-1]

            signal, confidence = generate_signal(current_price, combined)

            # Display Metrics
            st.metric("Current Price", f"${current_price:,.2f}")
            st.metric("RF Predicted Price", f"${pred_rf:,.2f}")
            st.metric("XGB Predicted Price", f"${pred_xgb:,.2f}")
            st.metric("Combined Prediction", f"${combined:,.2f}")
            st.metric("Trade Signal", f"{signal} ({confidence:.2f}% change)")

            # Plot Price + Indicators
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
            st.plotly_chart(fig, use_container_width=True)

            # Confidence Explanation
            st.info(
                f"⚡ The system uses combined predictions from RandomForest and XGBoost "
                f"trained on technical indicators (MA20, MA50, RSI). Predictions are not financial advice."
            )

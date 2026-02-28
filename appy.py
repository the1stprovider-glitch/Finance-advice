# ================================================================
# 📈 ADVANCED TRADING ADVICE + SIGNAL + BACKTEST SYSTEM
# A single all-in-one Streamlit app
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

# ---------------- Technical Indicators ----------------

def compute_indicators(df):
    # Moving Averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df

# ---------------- ML Training ----------------

def build_models(df):
    X = df[["MA20","MA50","EMA12","EMA26","MACD","SignalLine","RSI"]]
    y = df["Close"].shift(-1).fillna(method="ffill")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, subsample=0.8, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return rf, xgb

def predict_next(rf, xgb, features):
    pred_rf = rf.predict(features.values.reshape(1,-1))[0]
    pred_xgb = xgb.predict(features.values.reshape(1,-1))[0]
    combined = (pred_rf + pred_xgb)/2
    return pred_rf, pred_xgb, combined

# ---------------- Signal Logic ----------------

def trade_signal(current, predicted):
    diff = predicted - current
    pct = (diff/current)*100
    if pct > 4:
        return "STRONG BUY", pct
    elif pct > 2:
        return "BUY", pct
    elif pct < -4:
        return "STRONG SELL", pct
    elif pct < -2:
        return "SELL", pct
    return "HOLD", pct

# ---------------- UI Inputs ----------------

st.sidebar.header("📍 Configurations")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", value="AAPL")
start = st.sidebar.date_input("Start Date", value=datetime(2022,1,1))
end = st.sidebar.date_input("End Date", value=datetime.today())
if start >= end:
    st.sidebar.error("Start must be before End")

if st.sidebar.button("Run Analysis"):
    # -- Load Data --
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        st.error("No data found. Check ticker or date range.")
    else:
        df = compute_indicators(df)

        # -- Model Training --
        rf_model, xgb_model = build_models(df)
        latest_features = df.iloc[-1][["MA20","MA50","EMA12","EMA26","MACD","SignalLine","RSI"]]

        # -- Predictions --
        rfp, xgbp, combined = predict_next(rf_model, xgb_model, latest_features)
        current_price = df["Close"].iloc[-1]
        sig, conf = trade_signal(current_price, combined)

        # -- Metrics --
        st.metric("Current Price", f"${current_price:,.2f}")
        st.metric("RandomForest Pred", f"${rfp:,.2f}")
        st.metric("XGBoost Pred", f"${xgbp:,.2f}")
        st.metric("Combined Price", f"${combined:,.2f}")
        st.metric("Signal", f"{sig} ({conf:.2f}%)")

        # -- Backtest Table Snippet --
        df["PredNext"] = df[["MA20","MA50","EMA12","EMA26","MACD","SignalLine","RSI"]].apply(
            lambda row: (rf_model.predict(row.values.reshape(1,-1))[0] + 
                         xgb_model.predict(row.values.reshape(1,-1))[0])/2, axis=1)
        df["StrategyReturn"] = np.where(df["PredNext"] > df["Close"], 
                                        df["Close"].pct_change(), 
                                        -df["Close"].pct_change())
        stats = {
            "TotalReturn": f"{(df['StrategyReturn']+1).prod()-1:.2%}",
            "AvgDailyReturn": f"{df['StrategyReturn'].mean():.4%}",
            "Volatility": f"{df['StrategyReturn'].std():.4%}"
        }
        st.write("📊 Backtest Overview (simple strategy):", stats)

        # -- Price + Indicators Plot --
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
        fig.update_layout(title=f"{ticker} Price + Moving Averages")
        st.plotly_chart(fig, use_container_width=True)

        st.success("Analysis complete!")

# ---------------- Footer ----------------
st.caption("Powered by Yahoo Finance & ML ensemble models. Not financial advice.")

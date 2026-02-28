# ================================================================
# 📊 Trading Advice & Prediction App (Fixed + Faster)
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

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Trading Advisor", layout="wide")
st.title("📈 Trading Advice & Prediction Engine")

# ---------------- Cached Helpers ----------------

@st.cache_data
def fetch_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

@st.cache_data
def compute_features(df):
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

@st.cache_resource
def build_models(df):
    X = df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]]
    y = df["Close"].shift(-1).fillna(method="ffill")
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, shuffle=False)

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return rf, xgb

# ---------------- Signal Logic ----------------

def get_trade_signal(current_price, predicted_price):
    diff = float(predicted_price) - float(current_price)
    pct = (diff / float(current_price)) * 100

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

# ---------------- UI ----------------

st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2022,1,1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

if start_date >= end_date:
    st.sidebar.error("Start must be before End Date")

if st.sidebar.button("Run Analysis"):
    with st.spinner("Analyzing..."):
        data = fetch_data(ticker, start_date, end_date)

        if data.empty:
            st.error("❌ No data. Try a different ticker.")
        else:
            df = compute_features(data)

            # Train models once
            rf_model, xgb_model = build_models(df)

            # Grab last row features
            last_row = df.iloc[-1]
            current_price = float(last_row["Close"])
            feature_vals = last_row[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]].values

            pred_rf = float(rf_model.predict([feature_vals])[0])
            pred_xgb = float(xgb_model.predict([feature_vals])[0])
            combined_pred = (pred_rf + pred_xgb) / 2

            signal, confidence = get_trade_signal(current_price, combined_pred)

            # Display results safely
            st.write(f"**Current Close Price:** ${current_price:.2f}")
            st.write(f"**RandomForest Prediction:** ${pred_rf:.2f}")
            st.write(f"**XGBoost Prediction:** ${pred_xgb:.2f}")
            st.write(f"**Combined Prediction:** ${combined_pred:.2f}")
            st.write(f"**Suggested Trade Signal:** {signal} ({confidence:.2f}%)")

            # Compute simple backtest
            predictions = (rf_model.predict(df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]].values) +
                           xgb_model.predict(df[["SMA20","SMA50","EMA12","EMA26","MACD","SignalLine","RSI"]].values)) / 2

            # Now using numpy arrays directly avoids index alignment errors
            strategy_returns = np.where(predictions > df["Close"].values,
                                       df["Close"].pct_change().values,
                                       -df["Close"].pct_change().values)

            total_return = (np.nan_to_num(strategy_returns) + 1).prod() - 1
            avg_daily = np.nanmean(strategy_returns)
            volatility = np.nanstd(strategy_returns)

            st.write("📊 Simple Strategy Backtest:")
            st.write({
                "Total Return": f"{total_return:.2%}",
                "Avg Daily Return": f"{avg_daily:.4%}",
                "Volatility": f"{volatility:.4%}"
            })

            # Plot price with indicators
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Analysis complete!")

st.caption("*Data via Yahoo Finance; models are illustrative and not financial advice.*")

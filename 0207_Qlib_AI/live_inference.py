
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import requests
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOL = "ETHUSDT"
INTERVAL = "1m" # We fetch 1m data and resample to 10m
LOOKBACK_BARS = 200 # Need enough data for 60-period MA/ROC + Safety buffer
RESAMPLE_FREQ = "10min"
MODEL_PATH = Path(__file__).parent.resolve() / 'lgbm_model_eth_10m.pkl'
THRESHOLD = 0.001 

# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TG_TOKEN = "8052185621:AAFT1gMhEvxZYTixeijsjLA29Q6fpnEc1xs"
TG_CHAT_ID = "6290088209" # ⚠️ 请运行 python get_tg_chat_id.py 获取并填入此处

def send_telegram_alert(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send TG alert: {e}")

def get_binance_klines(symbol, interval, limit=1500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if isinstance(data, dict) and 'code' in data:
            print(f"Error fetching data: {data}")
            return []
        return data # [[open_time, open, high, low, close, vol, ...], ...]
    except Exception as e:
        print(f"Request Error: {e}")
        return []

def fetch_and_prepare_data(symbol=SYMBOL, interval=INTERVAL):
    # Fetch 1m data
    raw_klines = get_binance_klines(symbol, interval, limit=1000)
    if not raw_klines:
        return pd.DataFrame()
        
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'buyer_buy_base', 'buyer_buy_quote', 'ignore']
    df = pd.DataFrame(raw_klines, columns=cols)
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('datetime').sort_index()
    
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in num_cols:
        df[c] = df[c].astype(float)
        
    # Resample to 10min for Model
    # Logic matches prepare_eth_data.py
    df_10m = df.resample(RESAMPLE_FREQ).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_10m

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature Generation (Must match train_lgbm_eth.py)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_features(df):
    df = df.copy()
    
    # 1. Momentum / ROC
    for n in [5, 10, 20, 60]:
        # Ref($close, n) / $close - 1  -> In pandas: shift(n) / close - 1 ??
        # Wait, Qlib Ref(x, n) gets value of x from n steps AGO (history).
        # Qlib: Ref(x, d) = x_{t-d}
        # So Ref($close, 5) is close price 5 bars ago.
        # Formula: Ref($close, n) / $close - 1  =  Close_{t-n} / Close_t - 1
        # This is essentially "Return over last n bars" inverted? 
        # Usually ROC is (Close_t - Close_{t-n}) / Close_{t-n}.
        # Let's check my train definition: "Ref($close, 5) / $close - 1"
        # If I want future return, I use Ref($close, -1). 
        # If I want past feature, I use Ref($close, 5).
        # Wait, usually technical indicators use the PAST.
        # Let's verify Qlib's Ref direction. 
        # Ref(frame, n=1) -> Get data from n periods BEFORE (lag).
        # BUT key thing: Qlib's expression engine matches `Ref` to `shift(n)`.
        # Pandas shift(n) gets the value from n rows BEFORE (if positive n).
        # So `Ref($close, 5)` means value at t-5.
        # `Ref($close, 5) / $close - 1` = Close(t-5) / Close(t) - 1.
        
        # In Pandas: df['close'].shift(n) / df['close'] - 1
        df[f'ROC_{n}'] = df['close'].shift(n) / df['close'] - 1
        
    # 2. Volatility
    for n in [20, 60]:
        # Std($close, n) / Mean($close, n)
        # Pandas rolling std/mean
        rolling_std = df['close'].rolling(n).std()
        rolling_mean = df['close'].rolling(n).mean()
        df[f'VOL_{n}'] = rolling_std / rolling_mean
        
    # 3. MA Divergence
    for n in [5, 10, 20, 60]:
        # $close / Mean($close, n) - 1
        rolling_mean = df['close'].rolling(n).mean()
        df[f'MA_{n}'] = df['close'] / rolling_mean - 1
        
    # 4. Range
    # ($high - $low) / $close
    df['H_L_Ratio'] = (df['high'] - df['low']) / df['close']
    
    # ($close - $open) / $open
    df['C_O_Ratio'] = (df['close'] - df['open']) / df['open']
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Main Loop
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return
        
    print(f"Loading Model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Starting Live Inference for {SYMBOL}...")
    print(f"Resample: {RESAMPLE_FREQ}, Threshold: {THRESHOLD}")
    print("Press Ctrl+C to stop.\n")
    
    last_processed_time = None
    
    try:
        while True:
            # 1. Fetch
            df = fetch_and_prepare_data()
            if df.empty:
                time.sleep(10)
                continue
                
            # Check if new bar arrived
            current_last_time = df.index[-1]
            
            if last_processed_time == current_last_time:
                # Still the same bar, wait a bit
                time.sleep(10)
                continue
                
            last_processed_time = current_last_time
            
            # 2. Generate Features
            # We need to generate for the whole DF to handle rolling windows correctly
            # preparing feature columns
            df_features = generate_features(df)
            
            # Select feature columns in specific order (Must match training!)
            feature_cols = []
            # ROC
            for n in [5, 10, 20, 60]: feature_cols.append(f"ROC_{n}")
            # VOL
            for n in [20, 60]: feature_cols.append(f"VOL_{n}")
            # MA
            for n in [5, 10, 20, 60]: feature_cols.append(f"MA_{n}")
            # Ratios
            feature_cols.append("H_L_Ratio")
            feature_cols.append("C_O_Ratio")
            
            # Get the very last row (Current completed bar)
            # Actually, if we are trading "on close", we trade based on the just-closed bar.
            # df.index[-1] is the open time of the last bar? 
            # Binance kline open time. So if now is 10:11, the last 10m bar is 10:00-10:10.
            # Its open_time is 10:00. 
            # We want to predict for the NEXT bar (10:10-10:20) using data up to 10:10.
            # So taking the last row of df is correct (it represents the most recent completed period).
            
            latest_features = df_features.iloc[[-1]][feature_cols]
            
            # Check for NaNs (e.g. not enough history for rolling 60)
            if latest_features.isna().any().any():
                print(f"[{current_last_time}] Not enough data for features. Waiting...")
                continue
                
            # 3. Predict
            pred_score = model.predict(latest_features)[0]
            
            # 4. Signal Logic
            signal = "HOLD"
            if pred_score > THRESHOLD:
                signal = "BUY 🟢"
            elif pred_score < -THRESHOLD:
                signal = "SELL 🔴"
                
            # Output
            current_price = df['close'].iloc[-1]
            log_msg = f"[{current_last_time}] Price: {current_price:.2f} | Score: {pred_score:.6f} | Signal: {signal}"
            print(log_msg)
            
            # Send Telegram Alert on Signal
            if signal != "HOLD":
                tg_msg = f"🚀 **AI Signal Alert**\n\n" \
                         f"Symbol: `{SYMBOL}`\n" \
                         f"Time: `{current_last_time}`\n" \
                         f"Signal: **{signal}**\n" \
                         f"Price: `{current_price}`\n" \
                         f"Score: `{pred_score:.6f}`"
                send_telegram_alert(tg_msg)
            
            # Sleep to avoid spam
            time.sleep(30) 
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()

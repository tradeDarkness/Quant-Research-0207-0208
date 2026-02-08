
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
    for n in [1, 5, 10, 20, 60]:
        df[f'ROC_{n}'] = df['close'] / df['close'].shift(n) - 1
        
    # 2. Volatility
    for n in [10, 20, 60]:
        rolling_std = df['close'].rolling(n).std()
        rolling_mean = df['close'].rolling(n).mean()
        df[f'VOL_{n}'] = rolling_std / rolling_mean
        
    # 3. MA Divergence
    for n in [5, 10, 20, 60]:
        rolling_mean = df['close'].rolling(n).mean()
        df[f'MA_{n}'] = df['close'] / rolling_mean - 1
        
    # 4. Volume Features
    for n in [5, 10, 20]:
        rolling_v_mean = df['volume'].rolling(n).mean()
        df[f'V_MA_Ratio_{n}'] = df['volume'] / rolling_v_mean

    # 5. K-Line Shapes
    df['K_HIGH_REL'] = df['high'] / df['close'] - 1
    df['K_LOW_REL'] = df['low'] / df['close'] - 1
    df['K_OPEN_REL'] = df['open'] / df['close'] - 1
    df['H_L_Ratio'] = (df['high'] - df['low']) / df['close']
    df['C_O_Ratio'] = (df['close'] - df['open']) / df['open']
    
    # 6. RD-Agent Gen-1 Hypotheses
    # H1: Energy Spike
    df['H1_Spike'] = (df['volume'] / df['volume'].rolling(20).mean()) * (df['close'] / df['close'].shift(1) - 1)
    # H2: Price Quantile Position
    df['H2_Quantile'] = (df['close'] - df['close'].rolling(30).min()) / (df['close'].rolling(30).max() - df['close'].rolling(30).min() + 1e-9)
    # H3: Bias/Volatility
    df['H3_BiasVol'] = (df['close'].rolling(20).mean() / df['close'] - 1) / (df['close'].rolling(20).std() / df['close'].rolling(20).mean() + 1e-9)

    # 7. RD-Agent Gen-2 Hypotheses (Advanced Volatility)
    # H4: VRegime
    vol = (df['high'] - df['low']) / df['close']
    df['H4_VRegime'] = vol / (vol.rolling(60).mean() + 1e-9)
    # H5: MQuality
    df['H5_MQuality'] = (df['close'] / df['close'].shift(5) - 1) / (df['close'].rolling(5).std() / df['close'].rolling(5).mean() + 1e-9)
    # H6: Slope
    df['H6_Slope'] = (df['close'] / df['close'].rolling(20).mean() - 1) * (df['close'] / df['close'].shift(1) - 1)
    
    # 8. RD-Agent Gen-3 Hypotheses
    # H7: Trend Acceleration
    df['H7_TAccel'] = (df['close'] / df['close'].shift(5) - 1) / (df['close'] / df['close'].shift(20) - 1 + 1e-9)
    # H8: Volatility Squeeze
    df['H8_VSqueeze'] = df['close'].rolling(10).std() / (df['close'].rolling(60).std() + 1e-9)
    # H9: Volume-Confirmed Momentum
    df['H9_VMom'] = (df['close'] / df['close'].shift(1) - 1) * (df['volume'] / df['volume'].rolling(20).mean() + 1e-9)

    # 9. RD-Agent Gen-4 Hypotheses (Ultimate Strategy)
    # H10: Price-Volume Interaction
    df['H10_PVInt'] = (df['close'] / df['close'].shift(1) - 1) * (df['volume'] / df['volume'].rolling(10).mean())
    # H11: Dynamic Range Position (KDJ-Ref)
    df['H11_KDJK'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-9)
    # H12: Trend Persistence (ROC Mean)
    df['H12_UpStreak'] = (df['close'] / df['close'].shift(1) > 1).rolling(20).sum() / 20.0

    # 10. RD-Agent Gen-6 Hypotheses (Breakthrough 1000%+)
    # H13: High Breakout Signal
    df['H13_HBreak'] = df['close'] / df['high'].rolling(20).max() - 1
    # H14: Low Breakout Signal
    df['H14_LBreak'] = df['close'] / df['low'].rolling(20).min() - 1
    # H15: Triple MA Spread
    df['H15_TriMA'] = (df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1) + (df['close'].rolling(20).mean() / df['close'].rolling(60).mean() - 1)

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
            for n in [1, 5, 10, 20, 60]: feature_cols.append(f"ROC_{n}")
            # VOL
            for n in [10, 20, 60]: feature_cols.append(f"VOL_{n}")
            # MA
            for n in [5, 10, 20, 60]: feature_cols.append(f"MA_{n}")
            # Volume Features
            for n in [5, 10, 20]: feature_cols.append(f"V_MA_Ratio_{n}")
            # K-Line Shapes
            feature_cols.append("K_HIGH_REL")
            feature_cols.append("K_LOW_REL")
            feature_cols.append("K_OPEN_REL")
            feature_cols.append("H_L_Ratio")
            feature_cols.append("C_O_Ratio")
            # RD-Agent Gen-1 Hypotheses
            feature_cols.append("H1_Spike")
            feature_cols.append("H2_Quantile")
            feature_cols.append("H3_BiasVol")
            # RD-Agent Gen-2 Hypotheses
            feature_cols.append("H4_VRegime")
            feature_cols.append("H5_MQuality")
            feature_cols.append("H6_Slope")
            # RD-Agent Gen-3 Hypotheses
            feature_cols.append("H7_TAccel")
            feature_cols.append("H8_VSqueeze")
            feature_cols.append("H9_VMom")
            # RD-Agent Gen-4 Hypotheses
            feature_cols.append("H10_PVInt")
            feature_cols.append("H11_KDJK")
            feature_cols.append("H12_UpStreak")
            # RD-Agent Gen-6 Hypotheses
            feature_cols.append("H13_HBreak")
            feature_cols.append("H14_LBreak")
            feature_cols.append("H15_TriMA")
            
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

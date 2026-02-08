
import time
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta

# Model Configuration
MODEL_PATH = "lgbm_btc_15m.pkl"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"

def get_latest_klines(symbol, interval, limit=100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"Error fetching klines: {e}")
    return []

def prepare_live_features(klines):
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbb', 'tbq', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in num_cols:
        df[c] = df[c].astype(float)
        
    # Feature Engineering (MUST MATCH prepare_data.py)
    # 1. Price Action
    for i in [1, 2, 4, 8, 12, 24, 96]:
        df[f'ret_{i}'] = df['close'] / df['close'].shift(i) - 1
        df[f'vol_{i}'] = df['close'].rolling(i).std() / df['close']


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bbands(series, period=20, std_dev=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def prepare_live_features(klines):
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbb', 'tbq', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in num_cols:
        df[c] = df[c].astype(float)
        
    # Feature Engineering (MUST MATCH prepare_data.py)
    # 1. Price Action
    for i in [1, 2, 4, 8, 12, 24, 96]:
        df[f'ret_{i}'] = df['close'] / df['close'].shift(i) - 1
        if i > 1:
            df[f'vol_{i}'] = df['close'].rolling(i).std() / df['close']
        else:
            df[f'vol_{i}'] = 0.0

    # 2. Volume
    for i in [1, 4, 12]:
        df[f'v_ret_{i}'] = df['volume'] / df['volume'].shift(i) - 1
        
    # 3. TA (Manual)
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['macd'], df['macdsignal'], df['macdhist'] = calculate_macd(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    upper, middle, lower = calculate_bbands(df['close'])
    df['bb_width'] = (upper - lower) / middle
    df['bb_pos'] = (df['close'] - lower) / (upper - lower)
    
    # 4. Time
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    return df.iloc[[-1]]

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found. Train it first!")
        return

    print("Starting Polymarket BTC 15m Predictor...")
    
    last_processed = None
    
    while True:
        now = datetime.now()
        
        # Run shortly after 15m close (e.g., :00:05, :15:05, :30:05, :45:05)
        # But for live testing, run immediately if new bar available?
        # Let's just poll.
        
        klines = get_latest_klines(SYMBOL, INTERVAL)
        if not klines:
            time.sleep(5)
            continue
            
        latest_open_time = klines[-1][0] # This is current OPEN candle
        # We need the CLOSED candle to predict the NEXT one?
        # Target in training was: Close[t+1] > Close[t]
        # At time t (closed), we predict t+1 (next closed).
        # So we use data up to t (closed).
        # klines[-1] is OPEN (unclosed). klines[-2] is the last CLOSED candle.
        
        # Wait, if we trade Polymarket "15m Up/Down", the event usually corresponds to a specific interval.
        # E.g., "BTC Price at 14:00 > 13:45?"
        # If we are at 13:46, we are predicting the 14:00 close?
        # Training target: Close[t+1] / Close[t] - 1.
        # So features at t (closed) predict t+1 (closed).
        # So we should use klines[-2] features to predict klines[-1] outcome?
        # No, we want to predict FUTURE.
        # We are at time T (now). The current candle is OPEN.
        # We want to predict if this CURRENT candle will close UP?
        # Or predict the NEXT candle?
        
        # Training logic:
        # df['target_return'] = df['close'].shift(-1) / df['close'] - 1
        # at index i, features are from i. target is return from i to i+1.
        # So features[i] predict close[i+1] relative to close[i].
        
        # Realtime:
        # We have closed candle at index -2 (call it T-1).
        # We have open candle at index -1 (call it T).
        # To match training, we take features from T-1 (closed).
        # Prediction will be for T (current open candle, closing soon).
        # Yes.
        
        last_closed_ts = klines[-2][0]
        
        if last_processed != last_closed_ts:
            # New closed candle available
            print(f"Analyzing candle closed at {datetime.fromtimestamp(last_closed_ts/1000)}...")
            
            # Prepare features using history up to -2
            # Need enough history for rolling windows (96)
            # klines has 1000 limit default? function has 100. 100 is enough for 96? barely.
            # 100 bars: 0..99. 96-period rolling needs 96 previous.
            # Change limit to 200.
            
            features_df = prepare_live_features(klines[:-1]) # Exclude current open candle?
            # Wait, `prepare_live_features` calculates rolling on the DF.
            # If we pass klines[:-1], the last row is the one we want to predict from.
            
            # Predict
            # Drop non-feature cols
            drop_cols = ['datetime', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'close_time', 'trades', 'target_return', 'target', 'ignore', 'buyer_buy_base', 'buyer_buy_quote', 'qav', 'tbb', 'tbq']
            # Also features created but not used? 
            # LightGBM selects features by name. We just need to ensure cols match training.
            
            X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
            print(f"Live features ({len(X.columns)}): {list(X.columns)}")
            
            prob = model.predict(X)[0]
            pred = 1 if prob > 0.5 else 0
            
            direction = "UP 🟢" if pred == 1 else "DOWN 🔴"
            confidence = prob if pred == 1 else 1 - prob
            
            current_price = float(klines[-1][4]) # Current live price
            
            print(f"Prediction for 15m candle ending {datetime.fromtimestamp((last_closed_ts+15*60000)/1000)}:")
            print(f"Direction: {direction} | Confidence: {confidence:.2%}")
            print(f"Current Price: {current_price}")
            
            # TODO: Integrate Polymarket API to place bet
            
            last_processed = last_closed_ts
            
        time.sleep(10)

if __name__ == "__main__":
    main()

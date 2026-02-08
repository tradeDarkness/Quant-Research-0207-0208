
import pandas as pd
import numpy as np

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

def prepare_features(input_file="BTCUSDT_15m.csv", output_file="btc_15m_features.csv"):
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Target: 1 if next close > current close, else 0
    df['target_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target'] = (df['target_return'] > 0).astype(int)
    
    # Technical Indicators (Manual Implementation)
    # 1. Price Action
    for i in [1, 2, 4, 8, 12, 24, 96]:
        df[f'ret_{i}'] = df['close'] / df['close'].shift(i) - 1
        if i > 1:
            df[f'vol_{i}'] = df['close'].rolling(i).std() / df['close']
        else:
             df[f'vol_{i}'] = 0.0 # Volatility of 1 bar is 0? Or just skip? Skip is better but keeping column simplifies.
             # Actually, std of 1 point is NaN. Let's just avoid calculating vol_1.

    # 2. Volume
    for i in [1, 4, 12]:
        df[f'v_ret_{i}'] = df['volume'] / df['volume'].shift(i) - 1
        
    # 3. TA
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['macd'], df['macdsignal'], df['macdhist'] = calculate_macd(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    upper, middle, lower = calculate_bbands(df['close'])
    df['bb_width'] = (upper - lower) / middle
    df['bb_pos'] = (df['close'] - lower) / (upper - lower)
    
    # 4. Time Features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # Clean NaN
    print(f"Shape before dropna: {df.shape}")
    print(f"NaN counts:\n{df.isna().sum()}")
    df = df.dropna()
    
    print(f"Features created. Shape: {df.shape}")
    print(f"Target Distribution:\n{df['target'].value_counts(normalize=True)}")
    
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    prepare_features()

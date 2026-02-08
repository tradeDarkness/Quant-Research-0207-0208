
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def get_binance_klines(symbol, interval, end_time=None, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time
        
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=20)
            data = response.json()
            if isinstance(data, dict) and 'code' in data:
                print(f"Error: {data}")
                time.sleep(1)
                continue
            return data
        except Exception as e:
            print(f"Request Error (Attempt {attempt+1}): {e}")
            time.sleep(2)
    return []

def download_binance_history(symbol, interval, days=730):
    # Target: 2 years of 15m data
    all_klines = []
    
    # End time: Now
    end_ts = int(time.time() * 1000)
    start_ts_target = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    current_end = end_ts
    
    print(f"Downloading {symbol} {interval} target start: {pd.to_datetime(start_ts_target, unit='ms')}")
    
    while True:
        klines = get_binance_klines(symbol, interval, end_time=current_end)
        
        if not klines:
            print("No more data/Error.")
            break
            
        first_ts = klines[0][0]
        
        all_klines.extend(klines)
        
        date_str = datetime.fromtimestamp(first_ts/1000).strftime('%Y-%m-%d %H:%M')
        print(f"Fetched {len(klines)} bars... reached {date_str}", end='\r')
        
        # Next request should end before this batch started
        current_end = first_ts - 1
        
        if first_ts < start_ts_target:
            print(f"\nReached target start date: {date_str}")
            break
            
        time.sleep(0.1) # Rate limit
        
    if not all_klines:
        return pd.DataFrame()
        
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'buyer_buy_base', 'buyer_buy_quote', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    
    num_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
    for c in num_cols:
        df[c] = df[c].astype(float)
        
    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    
    # Filter range
    df = df[df['datetime'] >= pd.to_datetime(start_ts_target, unit='ms')]
    
    return df

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "15m"
    days = 730 # 2 years
    
    df = download_binance_history(symbol, interval, days=days)
    
    if not df.empty:
        filename = "BTCUSDT_15m.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} rows to {filename}")
    else:
        print("Download failed.")

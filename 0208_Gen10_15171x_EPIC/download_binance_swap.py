
import requests
import pandas as pd
import time
from datetime import datetime, timedelta


def get_binance_klines(symbol, interval, end_time=None, limit=1500):
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

def download_binance_swap_history(symbol, interval, days=365):
    # Check for existing file to resume
    filename = "ETHUSDT_Swap_1m_1y.csv"
    existing_df = pd.DataFrame()
    end_ts = int(time.time() * 1000)
    
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
            # Resume from the earliest timestamp
            min_ts = existing_df['datetime'].min().value // 10**6 # ns to ms
            end_ts = min_ts - 60000 # 1 min before
            print(f"Resuming from {existing_df['datetime'].min()}...")
        except Exception as e:
            print(f"Read error: {e}, starting fresh.")
            
    all_klines = []
    
    # Start from now (or resume point)
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
        
    # Process
    if not all_klines:
        if not existing_df.empty:
             print("No new data fetched, but have existing data.")
             return existing_df
        return pd.DataFrame()
        
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'buyer_buy_base', 'buyer_buy_quote', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Merge with existing
    if not existing_df.empty:
        # Align columns
        # Existing DF has 'datetime', 'amount' etc.
        # New DF has 'open_time', 'quote_asset_volume'
        # Normalize new DF first
        df = df.rename(columns={
            'open_time': 'datetime',
            'quote_asset_volume': 'amount' if 'amount' in existing_df.columns else 'quote_volume'
        })
        
        # Also ensure existing DF has consistent names
        # Just use pd.concat and verify columns match or are close
        # But existing df might have 'quote_volume' or 'amount'.
        # I should standarize on 'quote_volume' here for raw data.
        pass

    # Standardize New Data
    df = df.rename(columns={'open_time': 'datetime', 'quote_asset_volume': 'quote_volume'})
    num_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    for c in num_cols:
        df[c] = df[c].astype(float)
        
    # Combine
    if not existing_df.empty:
        # Existing df might have 'amount' instead of 'quote_volume' if I renamed it in previous run?
        # NO, prepare_eth_data renames it. The CSV on disk keeps 'quote_volume' unless I messed up.
        # Check existing columns
        if 'amount' in existing_df.columns and 'quote_volume' not in existing_df.columns:
             df = df.rename(columns={'quote_volume': 'amount'})
        
        full_df = pd.concat([existing_df, df])
    else:
        full_df = df

    full_df = full_df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    
    # Filter range
    # full_df = full_df[full_df['datetime'] >= pd.to_datetime(start_ts_target, unit='ms')]
    
    return full_df

if __name__ == "__main__":
    import os
    symbol = "ETHUSDT"
    days = 400 
    
    df = download_binance_swap_history(symbol, "1m", days=days)
    
    if not df.empty:
        filename = "ETHUSDT_Swap_1m_1y.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} rows to {filename}")
        
        # Resample to 10m immediately for Qlib
        print("Resampling to 10m...")
        df.set_index('datetime', inplace=True)
        
        # Ensure 'quote_volume' or 'amount' is used
        vol_col = 'amount' if 'amount' in df.columns else 'quote_volume'
        
        df_10m = df.resample('10min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            vol_col: 'sum' 
        })
        df_10m.dropna(inplace=True)
        
        filename_10m = "ETHUSDT_Swap_10m_1y.csv"
        df_10m.to_csv(filename_10m)
        print(f"Saved {len(df_10m)} rows to {filename_10m}")
        
    else:
        print("Download failed.")


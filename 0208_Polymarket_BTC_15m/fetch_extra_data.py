
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_binance_data(endpoint, params):
    url = f"https://fapi.binance.com{endpoint}"
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                time.sleep(1)
        except Exception as e:
            print(f"Connection Error: {e}")
            time.sleep(2)
    return []

def fetch_historical_oi(symbol, period='15m', limit=500):
    """
    Fetch historical Open Interest.
    Note: Open interest history is usually limited to the last 30 days for 15m.
    """
    print(f"Fetching Open Interest for {symbol} ({period})...")
    endpoint = "/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    data = get_binance_data(endpoint, params)
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Fields: ['symbol', 'sumOpenInterest', 'sumOpenInterestValue', 'timestamp']
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['oi'] = df['sumOpenInterest'].astype(float)
    df['oi_value'] = df['sumOpenInterestValue'].astype(float)
    return df[['datetime', 'oi', 'oi_value']]

def fetch_taker_ls_ratio(symbol, period='15m', limit=500):
    """
    Taker Buy/Sell Volume Ratio.
    """
    print(f"Fetching Taker Buy/Sell Ratio for {symbol} ({period})...")
    endpoint = "/futures/data/takerlongshortRatio"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    data = get_binance_data(endpoint, params)
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Fields: ['buySellRatio', 'buyVol', 'sellVol', 'timestamp']
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['taker_ls_ratio'] = df['buySellRatio'].astype(float)
    df['taker_buy_vol'] = df['buyVol'].astype(float)
    df['taker_sell_vol'] = df['sellVol'].astype(float)
    return df[['datetime', 'taker_ls_ratio', 'taker_buy_vol', 'taker_sell_vol']]

def fetch_top_trader_ls_ratio(symbol, period='15m', limit=500):
    """
    Top Trader Long/Short Ratio (Accounts).
    """
    print(f"Fetching Top Trader Long/Short Ratio for {symbol} ({period})...")
    endpoint = "/futures/data/topLongShortAccountRatio"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    data = get_binance_data(endpoint, params)
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Fields: ['symbol', 'longShortRatio', 'longAccount', 'shortAccount', 'timestamp']
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['top_trader_ls_ratio'] = df['longShortRatio'].astype(float)
    df['top_trader_long_pct'] = df['longAccount'].astype(float)
    return df[['datetime', 'top_trader_ls_ratio', 'top_trader_long_pct']]

if __name__ == "__main__":
    symbol = "BTCUSDT"
    # Fetch as much as possible (limit 500)
    df_oi = fetch_historical_oi(symbol)
    df_taker = fetch_taker_ls_ratio(symbol)
    df_top = fetch_top_trader_ls_ratio(symbol)
    
    # Merge on datetime
    if not df_oi.empty:
        df_final = df_oi.merge(df_taker, on='datetime', how='inner')
        df_final = df_final.merge(df_top, on='datetime', how='inner')
        
        filename = "BTCUSDT_extra_data.csv"
        df_final.to_csv(filename, index=False)
        print(f"Saved {len(df_final)} extra data rows to {filename}")
        print(f"Time range: {df_final['datetime'].min()} to {df_final['datetime'].max()}")
    else:
        print("Failed to fetch extra data.")

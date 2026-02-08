import pandas as pd
import numpy as np
import os

def generate_triple_barrier_labels(csv_path, pt=0.008, sl=0.004, t1=12):
    """
    Generate Fixed Triple Barrier Labels (Alpha-Squeezing):
    - pt: 0.8% Profit Target
    - sl: 0.4% Stop Loss
    - t1: 12 bars (3 hours)
    """
    print(f"Loading {csv_path} for Alpha-Squeezing generation...")
    df = pd.read_csv(csv_path)
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    df = df.sort_values('datetime').reset_index(drop=True)
    prices = df['close'].values
    n = len(prices)
    labels = np.zeros(n)
    
    print(f"Generating labels for {n} rows (PT={pt}, SL={sl}, T1={t1})...")
    
    for i in range(n - t1):
        start_price = prices[i]
        window = prices[i+1 : i+1+t1]
        rets = window / start_price - 1
        
        pt_idx = np.where(rets >= pt)[0]
        sl_idx = np.where(rets <= -sl)[0]
        
        first_pt = pt_idx[0] if len(pt_idx) > 0 else 999
        first_sl = sl_idx[0] if len(sl_idx) > 0 else 999
        
        if first_pt < first_sl and first_pt < t1:
            labels[i] = 1 # Profit Hit
        else:
            labels[i] = 0 # Loss or Time-out
            
    df['lb_tb'] = labels
    output_path = csv_path.replace(".csv", "_tb.csv")
    df.to_csv(output_path, index=False)
    print(f"Labels generated. Positive rate: {labels.mean():.2%}")
    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    path = "/Users/zhangzc/7/20260123/0208_Polymarket_BTC_15m/BTCUSDT_15m.csv"
    # Squeezing high-precision 1-hour alpha
    generate_triple_barrier_labels(path, pt=0.005, sl=0.003, t1=4)

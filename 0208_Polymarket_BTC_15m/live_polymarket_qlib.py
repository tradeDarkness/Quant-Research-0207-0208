
import sys
import os
import pandas as pd
import numpy as np
import requests
import joblib
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lgbm_btc_15m_final.pkl")
CSV_PATH = os.path.join(BASE_DIR, "BTCUSDT_15m.csv")

class LiveModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        # Load Phase 4 Model
        print(f"Loading Phase 4 model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found. Please train it first.")
        self.model = joblib.load(model_path)
        print("Model loaded successfully.")
        
    def fetch_latest_data(self):
        """Fetch latest candles from Binance."""
        print("Fetching latest BTCUSDT data from Binance...")
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 200}
        
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            
            new_rows = []
            for k in data:
                ts = int(k[0])
                dt = datetime.fromtimestamp(ts / 1000)
                new_rows.append({
                    "datetime": dt,
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                    "trades": int(k[8]),
                    "buyer_buy_base": float(k[9]) # Taker Buy Volume
                })
            new_df = pd.DataFrame(new_rows)
            
            if os.path.exists(CSV_PATH):
                existing_df = pd.read_csv(CSV_PATH)
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=['datetime'], keep='last')
                combined = combined.sort_values('datetime').reset_index(drop=True)
                combined.to_csv(CSV_PATH, index=False)
                return combined
            else:
                new_df.to_csv(CSV_PATH, index=False)
                return new_df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_features(self, df):
        """Replicate the high-precision feature engineering from train_qlib_model.py."""
        df = df.copy()
        print("Calculating Phase 4 features (ROC, Vol, L2 Proxy)...")
        # ROC
        for n in [1, 5, 10, 20, 60, 100]: df[f'ROC_{n}'] = df['close'].pct_change(n)
        # Vol
        for n in [10, 20, 60, 100]: df[f'VOL_{n}'] = df['close'].rolling(n).std() / (df['close'].rolling(n).mean() + 1e-9)
        # MA
        for n in [5, 10, 20, 60, 100]: df[f'MA_{n}'] = df['close'] / (df['close'].rolling(n).mean() + 1e-9) - 1
        # L2 Proxy
        df['L2_TakerBuyRatio'] = df['buyer_buy_base'] / (df['volume'] + 1e-9)
        df['L2_Imbalance'] = (df['buyer_buy_base'] - (df['volume'] - df['buyer_buy_base'])) / (df['volume'] + 1e-9)
        df['L2_VolIntensity'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
        df['L2_TradeIntensity'] = df['trades'] / (df['trades'].rolling(20).mean() + 1e-9)
        # Hypotheses
        df['H1_Spike'] = df['L2_VolIntensity'] * df['ROC_1']
        df['H2_Quantile'] = (df['close'] - df['close'].rolling(30).min()) / (df['close'].rolling(30).max() - df['close'].rolling(30).min() + 1e-9)
        df['H21_BBBreak'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-9)
        
        feature_cols = [c for c in df.columns if any(p in c for p in ['ROC_', 'VOL_', 'MA_', 'L2_', 'H1_', 'H2_', 'H21_'])]
        return df, feature_cols

    def predict_next(self):
        """Inference using Phase 4 Meta-tuned thresholds."""
        df = self.fetch_latest_data()
        if df is None: return None
        
        df, feature_cols = self.calculate_features(df)
        latest_row = df.iloc[[-1]]
        
        if latest_row[feature_cols].isna().any().any():
            print("Warning: Latest features contain NaNs. Not enough history?")
            return None
            
        score = self.model.predict(latest_row[feature_cols])[0]
        dt = latest_row['datetime'].iloc[0]
        price = latest_row['close'].iloc[0]
        
        # High-Precision Thresholding (from Ph4 results)
        if score > 0.60:
            signal = "强烈看涨 (STRONG BULLISH)"
        elif score > 0.50:
            signal = "看涨 (BULLISH)"
        elif score < 0.35:
            signal = "看跌 (BEARISH)"
        else:
            signal = "中性 (NEUTRAL)"
            
        print(f"\n>>> Prediction for {dt} <<<")
        print(f"Current Price: {price:.2f}")
        print(f"Confidence Score: {score:.4f} | Signal: {signal}")
        
        return {"score": float(score), "signal": signal, "datetime": str(dt), "price": float(price)}

    def predict_next_dict(self):
        return self.predict_next()

if __name__ == "__main__":
    live = LiveModel()
    live.predict_next()

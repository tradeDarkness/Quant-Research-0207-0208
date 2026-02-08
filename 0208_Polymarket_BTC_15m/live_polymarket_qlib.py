import sys
import os
import pandas as pd
import numpy as np
import requests
import joblib
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.workflow import R
import warnings
import time
from datetime import datetime, timedelta

# Suppress Qlib/Gym warnings
warnings.filterwarnings("ignore")

# Configuration
QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/my_crypto")
MODEL_PATH = "qlib_lgbm_btc_15m.pkl"
CSV_PATH = "BTCUSDT_15m.csv"

# Import data preparation script
# Modify sys.path if needed
sys.path.append(os.getcwd())
import prepare_qlib_btc_15m

class LiveModel:
    def __init__(self, model_path=MODEL_PATH, qlib_dir=QLIB_DIR):
        self.model_path = model_path
        self.qlib_dir = qlib_dir
        
        # Initialize Qlib
        qlib.init(provider_uri=qlib_dir, region="us")
        
        # Load Model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model loaded successfully.")
        
    def fetch_latest_data(self):
        """
        Fetch latest 15m candles from Binance and update CSV.
        """
        print("Fetching latest BTCUSDT data from Binance...")
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "limit": 100 # Fetch last 100 to ensure overlap
        }
        
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            
            # Parse
            # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote, ignore]
            new_rows = []
            for k in data:
                ts = int(k[0])
                dt = datetime.fromtimestamp(ts / 1000)
                row = {
                    "datetime": dt,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_asset_volume": float(k[7]), # amount
                    "number_of_trades": int(k[8]),
                    "taker_buy_base_asset_volume": float(k[9]),
                    "taker_buy_quote_asset_volume": float(k[10])
                }
                new_rows.append(row)
                
            new_df = pd.DataFrame(new_rows)
            
            # Load existing CSV
            if os.path.exists(CSV_PATH):
                existing_df = pd.read_csv(CSV_PATH)
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                
                # Merge and Drop Duplicates
                combined = pd.concat([existing_df, new_df])
                combined = combined.drop_duplicates(subset=['datetime'], keep='last')
                combined = combined.sort_values('datetime').reset_index(drop=True)
                
                # Save
                combined.to_csv(CSV_PATH, index=False)
                print(f"Updated {CSV_PATH}. Latest data: {combined['datetime'].iloc[-1]}")
                return combined['datetime'].iloc[-1]
            else:
                print(f"Error: {CSV_PATH} not found.")
                # If not found, create only new?
                new_df.to_csv(CSV_PATH, index=False)
                return new_df['datetime'].iloc[-1]
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def refresh_qlib_data(self):
        """
        Run the data preparation script to update Qlib binaries.
        """
        print("Refreshing Qlib binary data...")
        # We need to reload the module or just call the function if available
        # But prepare_qlib_btc_15m is a script.
        # We imported it, so we can call convert_to_qlib_format()
        prepare_qlib_btc_15m.convert_to_qlib_format(input_file=CSV_PATH, qlib_dir=self.qlib_dir)
        
        # Reload Qlib to see new data?
        # Qlib caches might persist. Best to clear cache.
        import shutil
        cache_dir = os.path.expanduser(f"{self.qlib_dir}/dataset_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            
        # Features cache
        ft_cache = os.path.expanduser(f"{self.qlib_dir}/features/BTCUSDT")
        if os.path.exists(ft_cache):
            for f in os.listdir(ft_cache):
                if f.endswith(".cache"):
                    os.remove(os.path.join(ft_cache, f))
                
        # Re-init? usually not needed if using D.features with cache cleared.
        print("Qlib data refreshed.")

    def predict_next(self):
        """
        Predict for the latest available timestamp.
        """
        # 1. Update Data
        last_dt = self.fetch_latest_data()
        if not last_dt:
            return
            
        # 2. Refresh Qlib
        self.refresh_qlib_data()
        
        # 3. Define Segment
        # We need to predict for the latest completed candle.
        # Format datetime string
        last_dt_str = last_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Start buffer
        start_dt = last_dt - timedelta(days=5) # ample buffer
        start_dt_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Preparing dataset for segment: {start_dt_str} to {last_dt_str}")
         # ═══════════════════════════════════════════════════════════════════════════════
        # RD-Agent Gen-10 Factor Definition (Ported from ETH Strategy)
        # ═══════════════════════════════════════════════════════════════════════════════
        fields = []
        names = []

        # 1. Momentum / ROC
        for n in [1, 5, 10, 20, 60]:
            fields.append(f"$close / Ref($close, {n}) - 1")
            names.append(f"ROC_{n}")

        # 2. Volatility
        for n in [10, 20, 60]:
            fields.append(f"Std($close, {n}) / Mean($close, {n})")
            names.append(f"VOL_{n}")

        # 3. MA Divergence
        for n in [5, 10, 20, 60]:
            fields.append(f"$close / Mean($close, {n}) - 1")
            names.append(f"MA_{n}")

        # 6. RD-Agent Gen-1 Hypotheses
        fields.append("($volume / Mean($volume, 20)) * ($close / Ref($close, 1) - 1)"); names.append("H1_Spike")
        fields.append("($close - Min($close, 30)) / (Max($close, 30) - Min($close, 30) + 1e-9)"); names.append("H2_Quantile")
        fields.append("(Mean($close, 20) / $close - 1) / (Std($close, 20) / Mean($close, 20) + 1e-9)"); names.append("H3_BiasVol")

        # 7. RD-Agent Gen-2 Hypotheses
        fields.append("(($high-$low)/$close) / (Mean(($high-$low)/$close, 60) + 1e-9)"); names.append("H4_VRegime")
        fields.append("($close / Ref($close, 5) - 1) / (Std($close, 5) / Mean($close, 5) + 1e-9)"); names.append("H5_MQuality")
        fields.append("($close / Mean($close, 20) - 1) * ($close / Ref($close, 1) - 1)"); names.append("H6_Slope")

        # 10. RD-Agent Gen-6 Hypotheses
        fields.append("$close / Max($high, 20) - 1"); names.append("H13_HBreak")
        fields.append("$close / Min($low, 20) - 1"); names.append("H14_LBreak")
        fields.append("(Mean($close, 5) / Mean($close, 20) - 1) + (Mean($close, 20) / Mean($close, 60) - 1)"); names.append("H15_TriMA")
        
        # 11. RD-Agent Gen-7
        fields.append("Abs($close / Ref($close, 1) - 1)"); names.append("H16_ExtRet")
        fields.append("($volume / Mean($volume, 10)) / (Abs($close / Ref($close, 1) - 1) + 1e-9)"); names.append("H17_VPDiv")

        # DataHandler Config
        dh_config = {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "instruments": ["BTCUSDT"],
                "start_time": "2024-02-09 06:45:00",
                "end_time": last_dt_str,
                "infer_processors": [
                    {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True, 'fit_start_time': '2024-02-09 06:45:00', 'fit_end_time': '2025-06-01 00:00:00'}}
                ],
                "learn_processors": [
                    {'class': 'DropnaLabel'},
                    {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True, 'fit_start_time': '2024-02-09 06:45:00', 'fit_end_time': '2025-06-01 00:00:00'}}
                ],
                "process_type": "independent",
                "data_loader": {
                    "class": "QlibDataLoader",
                    "kwargs": {
                        "config": {
                            "feature": (fields, names),
                            "label": (["Ref($close, -1) / $close - 1"], ["LABEL0"])
                        }
                    }
                }
            }
        }

        dataset = DatasetH(
            handler=dh_config,
            segments={
                "test": (start_dt_str, last_dt_str),
            }
        )
        
        try:
            # Prepare DataFrame from DatasetH
            # We need features only
            feature_df = dataset.prepare("test", col_set="feature")
            
            print("Predicting...")
            # self.model is raw LightGBM Booster
            pred = self.model.predict(feature_df)
            
            # Booster returns numpy array or list
            if isinstance(pred, list):
                pred = np.array(pred)
            
            # Get latest
            if len(pred) == 0:
                print("Warning: Prediction returned empty result.")
                return
                
            latest_score = pred[-1] 
            # Index is from feature_df
            latest_idx = feature_df.index[-1]
            if isinstance(latest_idx, tuple):
                 latest_idx = latest_idx[1]
            
            print(f"\n>>> Prediction for {latest_idx} (Target: Next 15m Return Direction) <<<")
            print(f"Score: {latest_score:.4f}")
            print(f"Signal: {'BULLISH' if latest_score > 0.502 else 'BEARISH' if latest_score < 0.498 else 'NEUTRAL'}")
            
            return latest_score
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    live = LiveModel()
    live.predict_next()

import pandas as pd
import numpy as np
import joblib
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import matplotlib.pyplot as plt
import os

# Configuration
QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/my_crypto")
MODEL_PATH = "qlib_lgbm_btc_15m.pkl"

# Initialize Qlib
qlib.init(provider_uri=QLIB_DIR, region="us")

def backtest():
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
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

    # Define Data Handler Config (Must match training!)
    dh_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "instruments": ["BTCUSDT"],
            "start_time": "2024-02-09 06:45:00", # Full history for fit
            "end_time": "2026-02-08 06:30:00",   # Test set end
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
    
    # Create Dataset
    print("Loading Test Data...")
    dataset = DatasetH(
        handler=dh_config,
        segments={
            "test": ("2025-10-01 00:15:00", "2026-02-08 06:30:00"),
        }
    )
    
    # Predict
    print("Predicting...")
    # Prepare DataFrame for raw booster
    feature_df = dataset.prepare("test", col_set="feature")
    # Also get label (Real Return)
    label_df = dataset.prepare("test", col_set="label")
    
    # Predict
    pred_score = model.predict(feature_df)
    
    # Align
    if isinstance(pred_score, list):
        pred_score = np.array(pred_score)
        
    print(f"Score Stats: Min={pred_score.min():.4f}, Max={pred_score.max():.4f}, Mean={pred_score.mean():.4f}, Std={pred_score.std():.4f}")
    print(f"Unique Scores: {len(np.unique(pred_score))}")
    if len(np.unique(pred_score)) < 10:
         print(f"Top 10 Scores: {pred_score[:10]}")
        
    df = pd.DataFrame(index=feature_df.index)
    df['score'] = pred_score
    df['return'] = label_df.iloc[:, 0]
    
    # Dynamic Thresholds?
    # Strategy Logic
    # Thresholds
    # Neutral zone to reduce noise?
    # If scores are super tight, maybe we need to rank them?
    # But for singular asset time-series, rank is meaningless (always 1).
    # We must use absolute threshold or z-score of prediction.
    
    score_mean = pred_score.mean()
    score_std = pred_score.std()
    
    # Use Z-score based thresholds if std > 0
    if score_std > 0:
        buy_threshold = score_mean + 0.1 * score_std
        sell_threshold = score_mean - 0.1 * score_std
    else:
        buy_threshold = 0.502
        sell_threshold = 0.498
        
    print(f"Using Buy Threshold: {buy_threshold:.4f}, Sell Threshold: {sell_threshold:.4f}")
    
    df['signal'] = 0
    df.loc[df['score'] > buy_threshold, 'signal'] = 1
    df.loc[df['score'] < sell_threshold, 'signal'] = -1
    
    # Calculate Strategy Return
    # Signal at T uses data up to T. Return is from T to T+1.
    # So Strategy Return = Signal * Return
    # BUT, wait.
    # Label definition: "Ref($close, -1) / $close - 1"
    # In Qlib, Ref($close, -1) is the price of the NEXT step (weird Qlib notation).
    # Usually Ref(x, d) means x at t-d.
    # Ref($close, -1) means Close at t - (-1) = t + 1.
    # So label at T is (Close_{t+1} / Close_t) - 1.
    # This is exactly the return of holding from T to T+1.
    
    # Transaction Costs? 
    # Let's assume 0.05% per trade (taker).
    cost = 0.0005
    
    # Position change
    df['pos_change'] = df['signal'].diff().abs().fillna(0)
    
    # Strategy Return
    df['strat_ret'] = df['signal'] * df['return'] - df['pos_change'] * cost
    
    # Cumulative
    df['cum_ret'] = (1 + df['strat_ret']).cumprod()
    df['nav'] = df['cum_ret']
    
    # Metrics
    total_ret = df['cum_ret'].iloc[-1] - 1
    win_rate = (df[df['signal'] != 0]['strat_ret'] > 0).mean()
    
    # Max Drawdown
    roll_max = df['nav'].cummax()
    drawdown = (df['nav'] - roll_max) / roll_max
    max_dd = drawdown.min()
    
    print("\n" + "="*40)
    print(f"Backtest Results (Test Set: 2025-10-01 - 2026-02-08)")
    print("="*40)
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Win Rate:     {win_rate*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Trades:       {df['pos_change'].sum():.0f}")
    print("="*40)
    
    # Save results
    df.to_csv("qlib_backtest_results.csv")
    print("Results saved to qlib_backtest_results.csv")

if __name__ == "__main__":
    backtest()

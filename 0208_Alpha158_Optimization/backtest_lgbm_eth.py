
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path

# 1. Load Data & Features
# We reuse the logic from train script to generate features for the whole period
# Or just load the test set using the same DataHandler config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
import qlib

provider_uri = '/Users/zhangzc/.qlib/qlib_data/crypto_10m'
qlib.init(provider_uri=provider_uri)

market = "all"

# Re-define features (Must match training!)
fields = []
names = []
for n in [1, 5, 10, 20, 60]:
    fields.append(f"$close / Ref($close, {n}) - 1"); names.append(f"ROC_{n}")
for n in [10, 20, 60]:
    fields.append(f"Std($close, {n}) / Mean($close, {n})"); names.append(f"VOL_{n}")
for n in [5, 10, 20, 60]:
    fields.append(f"$close / Mean($close, {n}) - 1"); names.append(f"MA_{n}")
for n in [5, 10, 20]:
    fields.append(f"$volume / Mean($volume, {n})"); names.append(f"V_MA_Ratio_{n}")
fields.append("$high / $close - 1"); names.append("K_HIGH_REL")
fields.append("$low / $close - 1"); names.append("K_LOW_REL")
fields.append("$open / $close - 1"); names.append("K_OPEN_REL")
fields.append("($high - $low) / $close"); names.append("H_L_Ratio")
fields.append("($close - $open) / $open"); names.append("C_O_Ratio")

# RD-Agent Gen-1 Hypotheses
fields.append("($volume / Mean($volume, 20)) * ($close / Ref($close, 1) - 1)"); names.append("H1_Spike")
fields.append("($close - Min($close, 30)) / (Max($close, 30) - Min($close, 30) + 1e-9)"); names.append("H2_Quantile")
fields.append("(Mean($close, 20) / $close - 1) / (Std($close, 20) / Mean($close, 20) + 1e-9)"); names.append("H3_BiasVol")

# RD-Agent Gen-2 Hypotheses
fields.append("(($high-$low)/$close) / (Mean(($high-$low)/$close, 60) + 1e-9)"); names.append("H4_VRegime")
fields.append("($close / Ref($close, 5) - 1) / (Std($close, 5) / Mean($close, 5) + 1e-9)"); names.append("H5_MQuality")
fields.append("($close / Mean($close, 20) - 1) * ($close / Ref($close, 1) - 1)"); names.append("H6_Slope")

# RD-Agent Gen-3 Hypotheses
fields.append("($close / Ref($close, 5) - 1) / ($close / Ref($close, 20) - 1 + 1e-9)"); names.append("H7_TAccel")
fields.append("Std($close, 10) / (Std($close, 60) + 1e-9)"); names.append("H8_VSqueeze")
fields.append("($close / Ref($close, 1) - 1) * ($volume / Mean($volume, 20) + 1e-9)"); names.append("H9_VMom")

# RD-Agent Gen-4 Hypotheses
fields.append("($close / Ref($close, 1) - 1) * ($volume / Mean($volume, 10))"); names.append("H10_PVInt")
fields.append("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-9)"); names.append("H11_KDJK")
fields.append("Mean($close / Ref($close, 1) > 1, 20)"); names.append("H12_UpStreak")

# RD-Agent Gen-6 Hypotheses (Fresh Approach)
fields.append("$close / Max($high, 20) - 1"); names.append("H13_HBreak")
fields.append("$close / Min($low, 20) - 1"); names.append("H14_LBreak")
fields.append("(Mean($close, 5) / Mean($close, 20) - 1) + (Mean($close, 20) / Mean($close, 60) - 1)"); names.append("H15_TriMA")

# RD-Agent Gen-7 Hypotheses (Targeting 100x)
fields.append("Abs($close / Ref($close, 1) - 1)"); names.append("H16_ExtRet")
fields.append("($volume / Mean($volume, 10)) / (Abs($close / Ref($close, 1) - 1) + 1e-9)"); names.append("H17_VPDiv")
fields.append("($high - $low) / (Mean($high - $low, 20) + 1e-9)"); names.append("H18_RangeExp")

# RD-Agent Gen-9 Hypotheses (Fresh Approach for 100x)
fields.append("($close / Ref($close, 5) - 1) - (Ref($close, 5) / Ref($close, 10) - 1)"); names.append("H19_MomAccel")
fields.append("Abs($close / Mean($close, 60) - 1)"); names.append("H20_ExtDev")
fields.append("$volume / Max($volume, 20)"); names.append("H21_VolShock")

data_handler_config = {
    "start_time": "2025-12-01", # Test Period
    "end_time": "2026-02-06",
    "instruments": market,
    "infer_processors": [
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
    ],
    "data_loader": {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": (fields, names),
                "label": (["Ref($close, -1) / $close - 1"], ["label"])
            }
        }
    }
}

print("Loading Test Data...")
dh = DataHandlerLP(**data_handler_config)
# Force load to get DataFrame with index
df_test = dh.fetch(slice(None), col_set=["feature", "label"])
print(f"Test Data Shape: {df_test.shape}")

# 2. Load Model

# 2. Load Model
model_path = Path(__file__).parent.resolve() / 'lgbm_model_eth_10m.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 3. Predict
print("Predicting...")
# Extract features and label safely
# Qlib returns MultiIndex columns: (feature, name) and (label, label)
feature_df = df_test["feature"] 
label_series = df_test[("label", "label")] # Access the specific label column

pred = model.predict(feature_df)

# Create a results DataFrame to avoid MultiIndex issues
res_df = pd.DataFrame(index=df_test.index)
res_df["label"] = label_series
res_df["pred"] = pred

# 4. Simple Vectorized Strategy Simulation
# Signal: Top/Bottom Quantile or Threshold
threshold = 0.001 

res_df["signal"] = 0
res_df.loc[res_df["pred"] > threshold, "signal"] = 1
res_df.loc[res_df["pred"] < -threshold, "signal"] = -1

# Calculate Return
res_df["strategy_opt_return"] = res_df["signal"] * res_df["label"]

# Transaction Costs (0.05%)
cost = 0.0005
# Count trades: Signal change
trades = res_df["signal"].diff().abs().fillna(0) > 0
# Approx cost subtraction
res_df["net_return"] = res_df["strategy_opt_return"] - (trades * cost)

# Cumulative Return
res_df["cum_ret"] = (1 + res_df["net_return"]).cumprod()

# Stats
total_ret = res_df["cum_ret"].iloc[-1] - 1
win_rate = (res_df[res_df["signal"]!=0]["net_return"] > 0).mean()
n_trades = res_df["signal"].diff().abs().sum() / 2 

print("\n════════════════════════════════════════════════════════════════════════")
print(f"📊 AI Strategy Results (Test Set: {data_handler_config['start_time']} - {data_handler_config['end_time']})")
print(f"Threshold: {threshold}")
print(f"Total Return: {total_ret*100:.2f}%")
print(f"Win Rate:     {win_rate*100:.2f}%")
print(f"Trades:       {n_trades:.0f}")
print("════════════════════════════════════════════════════════════════════════")

# 5. Export to CSV for detailed analysis
res_csv_path = Path(__file__).parent.resolve() / "ai_backtest_results.csv"
res_df.to_csv(res_csv_path)

# 6. Plotting
try:
    import matplotlib.pyplot as plt
    
    # Handle MultiIndex if present (datetime, instrument)
    plot_df = res_df.copy()
    if isinstance(plot_df.index, pd.MultiIndex):
        plot_df = plot_df.reset_index()
        if 'datetime' in plot_df.columns:
            plot_df = plot_df.set_index('datetime')
        elif 'date' in plot_df.columns:
            plot_df = plot_df.set_index('date')
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df['cum_ret'], label='AI Strategy')
    plt.title(f'AI Strategy Equity Curve (Return: {total_ret*100:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    chart_path = Path(__file__).parent.resolve() / "backtest_chart.png"
    plt.savefig(chart_path)
    print(f"\n📈 Chart saved to: {chart_path}")
    
except ImportError:
    print("\n⚠️ matplotlib not installed. Skipping chart generation.")


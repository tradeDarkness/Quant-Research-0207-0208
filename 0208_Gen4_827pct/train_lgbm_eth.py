
import qlib
from qlib.data import D
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import pickle

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configuration 
# ═══════════════════════════════════════════════════════════════════════════════

provider_uri = '/Users/zhangzc/.qlib/qlib_data/crypto_10m'
qlib.init(provider_uri=provider_uri, region='us')

# Market & Benchmark
market = "all"
benchmark = "ETHUSDT"

# Data Handler Config (Alpha158)
# We use custom handler or standard Alpha158
# Standard Alpha158 might need adjustment for crypto decimals?
# But normalized features should work.

# Custom Feature Factors
# We define a simple set of technical indicators to avoid Alpha158 complexities
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

# 4. Volume Features (Simple)
for n in [5, 10, 20]:
    fields.append(f"$volume / Mean($volume, {n})")
    names.append(f"V_MA_Ratio_{n}")

# 5. K-Line Shapes
fields.append("$high / $close - 1"); names.append("K_HIGH_REL")
fields.append("$low / $close - 1"); names.append("K_LOW_REL")
fields.append("$open / $close - 1"); names.append("K_OPEN_REL")
fields.append("($high - $low) / $close"); names.append("H_L_Ratio")
fields.append("($close - $open) / $open"); names.append("C_O_Ratio")

# 6. RD-Agent Gen-1 Hypotheses
# H1: Energy Spike
fields.append("($volume / Mean($volume, 20)) * ($close / Ref($close, 1) - 1)"); names.append("H1_Spike")
# H2: Price Quantile Position
fields.append("($close - Min($close, 30)) / (Max($close, 30) - Min($close, 30) + 1e-9)"); names.append("H2_Quantile")
# H3: Bias/Volatility
fields.append("(Mean($close, 20) / $close - 1) / (Std($close, 20) / Mean($close, 20) + 1e-9)"); names.append("H3_BiasVol")

# 7. RD-Agent Gen-2 Hypotheses
fields.append("(($high-$low)/$close) / (Mean(($high-$low)/$close, 60) + 1e-9)"); names.append("H4_VRegime")
fields.append("($close / Ref($close, 5) - 1) / (Std($close, 5) / Mean($close, 5) + 1e-9)"); names.append("H5_MQuality")
fields.append("($close / Mean($close, 20) - 1) * ($close / Ref($close, 1) - 1)"); names.append("H6_Slope")

# 8. RD-Agent Gen-3 Hypotheses
fields.append("($close / Ref($close, 5) - 1) / ($close / Ref($close, 20) - 1 + 1e-9)"); names.append("H7_TAccel")
fields.append("Std($close, 10) / (Std($close, 60) + 1e-9)"); names.append("H8_VSqueeze")
fields.append("($close / Ref($close, 1) - 1) * ($volume / Mean($volume, 20) + 1e-9)"); names.append("H9_VMom")

# 9. RD-Agent Gen-4 Hypotheses (Ultimate Strategy)
# H10: Price-Volume Interaction (Proxy for Correlation)
fields.append("($close / Ref($close, 1) - 1) * ($volume / Mean($volume, 10))"); names.append("H10_PVInt")
# H11: Dynamic Range Position (KDJ logic)
fields.append("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-9)"); names.append("H11_KDJK")
# H12: Trend Persistence (ROC Mean)
fields.append("Mean($close / Ref($close, 1) > 1, 20)"); names.append("H12_UpStreak")

# 10. RD-Agent Gen-5 Hypotheses (Targeting 1000%)
# H13: Gap Detection (Open vs Previous Close)
fields.append("($open / Ref($close, 1) - 1)"); names.append("H13_Gap")
# H14: Momentum Reversal Pressure (Close position within bar)
fields.append("($close - $low) / ($high - $low + 1e-9)"); names.append("H14_BarPos")
# H15: Volatility-Normalized ROC (Sharpe-like)
fields.append("Mean($close / Ref($close, 1) - 1, 10) / (Std($close / Ref($close, 1), 10) + 1e-9)"); names.append("H15_SharpeROC")

data_handler_config = {
    "start_time": "2025-01-05",
    "end_time": "2026-02-06",
    # "fit_start_time": "2025-02-10", # Not needed for DataHandlerLP
    # "fit_end_time": "2025-09-30",
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

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset Preparation
# ═══════════════════════════════════════════════════════════════════════════════

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH

print("initializing Dataset (Custom Features)...")
# Use generic DataHandlerLP
dh = DataHandlerLP(**data_handler_config)
ds = DatasetH(dh, segments={
    "train": ("2025-01-05", "2025-09-30"),
    "valid": ("2025-10-01", "2025-11-30"),
    "test":  ("2025-12-01", "2026-02-06"),
})

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Training LightGBM
# ═══════════════════════════════════════════════════════════════════════════════

print("Preparing data for LightGBM...")
# Get DataFrame
df_train = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
df_valid = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

# Split X, y
X_train, y_train = df_train["feature"], df_train["label"]
X_valid, y_valid = df_valid["feature"], df_valid["label"]

print(f"Train Shape: {X_train.shape}")
print(f"Valid Shape: {X_valid.shape}")

print("X_train head:")
print(X_train.head())
print("y_train head:")
print(y_train.head())
print("y_train description:")
print(y_train.describe())

# Check for NaN/Inf
print(f"X_train NaNs: {X_train.isna().sum().sum()}")
print(f"y_train NaNs: {y_train.isna().sum().sum()}")

# Create LGB Dataset
dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

# Params
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['mse', 'l2'],
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print("Training functionality...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dvalid],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Save & Evaluate
# ═══════════════════════════════════════════════════════════════════════════════


model_path = Path(__file__).parent.resolve() / 'lgbm_model_eth_10m.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

# Feature Importance
print("\nTop 10 Features:")
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()
feat_imp = pd.DataFrame({'feature': feature_names, 'gain': importance}).sort_values('gain', ascending=False)
print(feat_imp.head(10))

print("✅ Training Complete.")

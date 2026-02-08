
import qlib
from qlib.constant import REG_CN, REG_US
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import sys
import pandas as pd
import os

# Configuration
provider_uri = os.path.expanduser("~/.qlib/qlib_data/my_crypto")
qlib.init(provider_uri=provider_uri, region=REG_US, mongo=dict(task_url="", db_name=""), kw_conf={"default_freq": "15min"})

def train_qlib_model():
    market = ["BTCUSDT"] # Use explicit list to avoid "all" -> default frequency (day) lookup issues
    benchmark = None
    
    # 1. Data Handler
    # Alpha158 for 15m? Alpha158 is designed for daily.
    # But Qlib factors are just expressions.
    # Ref($close, 1) means 1 step (15m).
    # So Alpha158 logic applies to 15m bars too.
    
    # Label: Next 15m return.
    # Ref($close, -1) / $close - 1
    
    dh_config = {
        "start_time": "2024-02-09 06:45:00",
        "end_time": "2026-02-08 06:30:00",
        "fit_start_time": "2024-02-09 06:45:00",
        "fit_end_time": "2025-06-01 00:00:00",
        "instruments": market,
        "freq": "15min", # Explicitly set frequency
        # "infer_processors": [],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}, # Sectional ZScore? For 1 asset it is just (x-mean)/std ?
            # For 1 asset CSZScore is useless or harmful if it enforces normal dist on time series?
            # CSZScore normalizes Cross Section.
            # With 1 asset, Cross Section is size 1.
            # So x -> (x - x) / 0 = NaN!
            # MUST REMOVE CSZScore for single asset!
        ],
        "label": ["Ref($close, -1) / $close - 1"],
    }
    
    # Use Alpha158. 
    # But standard Alpha158 might use CSZScore in its processor?
    # No, Alpha158 is a DataHandler class.
    # We can inherit or just use expression definitions.
    
    # Let's use custom handler config with Alpha158 expressions
    # To save time, we can instantiate Alpha158 and get fields?
    # Or just use "class": "Alpha158" and override processors?
    
    # Alpha158 defaults:
    # https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
    # It has `infer_processors = [ProcessInf, ZScoreNorm]`.
    # ZScoreNorm is time-series normalization? Or CS?
    # `qlib.data.ops.Ops` ?
    # Usually `RobustZScoreNorm` is robust standardization.
    
    # For single asset, we should use `RobustZScoreNorm(fit_start_time, fit_end_time)` 
    # or just `ZScoreNorm`.
    
    # Actually, let's use a simpler feature set first to verify, then Alpha158?
    # User requested Alpha158.
    
    dh_config["class"] = "Alpha158"
    dh_config["module_path"] = "qlib.contrib.data.handler"
    # DEBUG: Use raw fields to verify data loading
    # dh_config["kwargs"] = { ... }
    # Let's try to override fields to just ["$close"] to see if basic data works
    # But Alpha158 hardcodes fields in __init__.
    
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

    # 8. RD-Agent Gen-3 Hypotheses
    fields.append("($close / Ref($close, 5) - 1) / ($close / Ref($close, 20) - 1 + 1e-9)"); names.append("H7_TAccel")
    fields.append("Std($close, 10) / (Std($close, 60) + 1e-9)"); names.append("H8_VSqueeze")
    fields.append("($close / Ref($close, 1) - 1) * ($volume / Mean($volume, 20) + 1e-9)"); names.append("H9_VMom")

    # 9. RD-Agent Gen-4 Hypotheses
    fields.append("($close / Ref($close, 1) - 1) * ($volume / Mean($volume, 10))"); names.append("H10_PVInt")
    fields.append("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-9)"); names.append("H11_KDJK")
    fields.append("Mean($close / Ref($close, 1) > 1, 20)"); names.append("H12_UpStreak")

    # 10. RD-Agent Gen-6 Hypotheses
    fields.append("$close / Max($high, 20) - 1"); names.append("H13_HBreak")
    fields.append("$close / Min($low, 20) - 1"); names.append("H14_LBreak")
    fields.append("(Mean($close, 5) / Mean($close, 20) - 1) + (Mean($close, 20) / Mean($close, 60) - 1)"); names.append("H15_TriMA")
    
    fields.append("Abs($close / Ref($close, 1) - 1)"); names.append("H16_ExtRet")
    fields.append("($volume / Mean($volume, 10)) / (Abs($close / Ref($close, 1) - 1) + 1e-9)"); names.append("H17_VPDiv")

    # 12. RD-Agent Gen-10 Hypotheses
    fields.append("($close / Ref($close, 3) - 1) + ($close / Ref($close, 5) - 1) + ($close / Ref($close, 10) - 1)"); names.append("H19_MomFusion")
    fields.append("($high - $low) / (Min($high - $low, 20) + 1e-9)"); names.append("H20_VolExplo")
    fields.append("($close - Mean($close, 20)) / (Std($close, 20) + 1e-9)"); names.append("H21_BBBreak")
    fields.append("$volume / Ref($volume, 1) - 1"); names.append("H22_VolMomAcc")
    fields.append("($close - Ref($close, 3)) / 3"); names.append("H23_PriceVel")

    # DataHandler Config
    dh_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "instruments": ["BTCUSDT"],
            "start_time": "2024-02-09 06:45:00",
            "end_time": "2026-02-08 06:30:00",
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
    # 2. Model
    # LightGBM
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "binary",
        }
    }

    model_config["kwargs"].update({
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "verbose": -1
    })

    # 3. Dataset
    dataset = DatasetH(
        handler=dh_config,
        segments={
            "train": ("2024-02-09 06:45:00", "2025-06-01 00:00:00"),
            "valid": ("2025-06-01 00:15:00", "2025-10-01 00:00:00"),
            "test": ("2025-10-01 00:15:00", "2026-02-08 06:30:00"),
        }
    )
    
    print("Preparing Qlib Dataset (Alpha158)... this may take a while to generate factors...")
    # This triggers loading and processing
    # Alpha158 uses multiprocessing.
    
    # Train
    print("Training Model...")
    model = init_instance_by_config(model_config)
    model.fit(dataset)
    
    # Predict
    print("Predicting Test Set...")
    pred = model.predict(dataset, segment="test")
    
    # We can get labels to calc metrics manually if needed, or rely on LGBM logs.
    # Qlib's model.predict returns Score (prob).
    
    # Let's verify AUC
    labels = dataset.prepare("test", col_set="label")
    
    # Align indices
    # pred = pred.reindex(labels.index) # prediction might be Series
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Label is return, we need binary for AUC
    y_return = labels.iloc[:, 0]
    y_true = (y_return > 0).astype(int)
    
    if isinstance(pred, pd.DataFrame):
        y_pred = pred.iloc[:, 0]
    else:
        y_pred = pred
        
    y_pred = y_pred.reindex(y_true.index).fillna(0.5)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"AUC Error: {e}")
        auc = 0.5
        
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    
    print(f"\nQlib Alpha158 + LGBM Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    # Save model
    import joblib
    joblib.dump(model.model, "qlib_lgbm_btc_15m.pkl") # Save internal boiler/LGBM booster
    print("Model saved to qlib_lgbm_btc_15m.pkl")

if __name__ == "__main__":
    train_qlib_model()

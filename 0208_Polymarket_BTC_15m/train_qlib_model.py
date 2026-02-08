
import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

def train_qlib_model():
    csv_path = "/Users/zhangzc/7/20260123/0208_Polymarket_BTC_15m/BTCUSDT_15m_tb.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['symbol'] = 'btcusdt'
    df = df.set_index(['datetime', 'symbol']).sort_index()
    
    if 'trades' not in df.columns: df['trades'] = df['number_of_trades']
    if 'buyer_buy_base' not in df.columns: df['buyer_buy_base'] = df['taker_buy_base_asset_volume']
        
    print("Calculating full features for 70% target...")
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
    
    df['label'] = df['lb_tb']
    feature_cols = [c for c in df.columns if any(p in c for p in ['ROC_', 'VOL_', 'MA_', 'L2_', 'H1_', 'H2_', 'H21_'])]
    df = df.dropna(subset=feature_cols + ['label'])
    
    # Split
    ts = df.index.get_level_values('datetime')
    train_df = df[ts < pd.Timestamp('2025-06-01')]
    valid_df = df[(ts >= pd.Timestamp('2025-06-01')) & (ts < pd.Timestamp('2025-10-01'))]
    test_df = df[ts >= pd.Timestamp('2025-10-01')]

    print(f"Final Count: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    # Stage 1 + Stage 2 (Combined Features)
    train_data = lgb.Dataset(train_df[feature_cols], label=train_df['label'])
    valid_data = lgb.Dataset(valid_df[feature_cols], label=valid_df['label'], reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42
    }

    print("Training High-Alpha Model...")
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[valid_data], 
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    # Evaluate Top-K Precision
    scores = model.predict(test_df[feature_cols])
    test_df = test_df.copy()
    test_df['score'] = scores
    test_df = test_df.sort_values('score', ascending=False)
    
    y_true = test_df['label'].values
    print("\n--- Top-K Precision Analysis (Squeezing 70% Accuracy) ---")
    results = []
    for k in [10, 20, 50, 100, 200, 500]:
        if len(y_true) < k: break
        top_y = y_true[:k]
        prec = np.mean(top_y)
        results.append({"Top-K": k, "Precision": f"{prec:.2%}", "MinScore": f"{test_df['score'].iloc[k-1]:.4f}"})
    
    print(pd.DataFrame(results).to_string(index=False))

    auc = roc_auc_score(y_true, scores)
    print(f"\nTest AUC: {auc:.4f}")
    
    joblib.dump(model, "lgbm_btc_15m_final.pkl")
    print("Final Model saved.")

if __name__ == "__main__":
    train_qlib_model()

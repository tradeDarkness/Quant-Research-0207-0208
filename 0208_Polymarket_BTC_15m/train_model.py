
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

def train_model(input_file="btc_15m_features.csv", model_file="lgbm_btc_15m.pkl"):
    print("Loading data...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Run prepare_data.py first.")
        return

    # Drop non-feature columns
    drop_cols = ['datetime', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'close_time', 'trades', 'target_return', 'target', 'ignore', 'buyer_buy_base', 'buyer_buy_quote']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # Split
    # Time Analysis Split: Train (80%), Valid (10%), Test (10%)
    train_size = int(len(df) * 0.8)
    valid_size = int(len(df) * 0.1)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_valid = X.iloc[train_size:train_size+valid_size]
    y_valid = y.iloc[train_size:train_size+valid_size]
    
    X_test = X.iloc[train_size+valid_size:]
    y_test = y.iloc[train_size+valid_size:]
    
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    # Train
    print("Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Eval
    y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"\nModel Evaluation (Test Set):")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance.head(10))
    
    # Save
    joblib.dump(model, model_file)
    print(f"\nModel saved to {model_file}")

if __name__ == "__main__":
    train_model()

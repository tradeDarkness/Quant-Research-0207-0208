
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

# 北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOL = "ETHUSDT"
INTERVAL = "1m" # We fetch 1m data and resample to 10m
LOOKBACK_BARS = 200 # Need enough data for 60-period MA/ROC + Safety buffer
RESAMPLE_FREQ = "10min"
MODEL_PATH = Path(__file__).parent.resolve() / 'lgbm_model_eth_10m.pkl'
THRESHOLD = 0.00025  # Gen-7 3791% 阈值
STRATEGY_NAME = "Gen-7-3791pct"

# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TG_TOKEN = "8052185621:AAFT1gMhEvxZYTixeijsjLA29Q6fpnEc1xs"
TG_CHAT_ID = "6290088209"

def send_telegram_alert(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"Failed to send TG alert: {e}")

def get_binance_klines(symbol, interval, limit=1500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if isinstance(data, dict) and 'code' in data:
            print(f"Error fetching data: {data}")
            return []
        return data
    except Exception as e:
        print(f"Request Error: {e}")
        return []

def fetch_and_prepare_data(symbol=SYMBOL, interval=INTERVAL):
    raw_klines = get_binance_klines(symbol, interval, limit=1000)
    if not raw_klines:
        return pd.DataFrame()
        
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'buyer_buy_base', 'buyer_buy_quote', 'ignore']
    df = pd.DataFrame(raw_klines, columns=cols)
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('datetime').sort_index()
    
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in num_cols:
        df[c] = df[c].astype(float)
        
    df_10m = df.resample(RESAMPLE_FREQ).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    

    # 11. RD-Agent Gen-7 Hypotheses (Targeting 100x)
    # H16: Extreme Return Detection
    df['H16_ExtRet'] = (df['close'] / df['close'].shift(1) - 1).abs()
    # H17: Volume-Price Divergence
    df['H17_VPDiv'] = (df['volume'] / df['volume'].rolling(10).mean()) / ((df['close'] / df['close'].shift(1) - 1).abs() + 1e-9)
    # H18: Range Expansion Ratio
    df['H18_RangeExp'] = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-9)

    return df_10m

def generate_features(df):
    df = df.copy()
    
    for n in [1, 5, 10, 20, 60]:
        df[f'ROC_{n}'] = df['close'] / df['close'].shift(n) - 1
        
    for n in [10, 20, 60]:
        rolling_std = df['close'].rolling(n).std()
        rolling_mean = df['close'].rolling(n).mean()
        df[f'VOL_{n}'] = rolling_std / rolling_mean
        
    for n in [5, 10, 20, 60]:
        rolling_mean = df['close'].rolling(n).mean()
        df[f'MA_{n}'] = df['close'] / rolling_mean - 1
        
    for n in [5, 10, 20]:
        rolling_v_mean = df['volume'].rolling(n).mean()
        df[f'V_MA_Ratio_{n}'] = df['volume'] / rolling_v_mean

    df['K_HIGH_REL'] = df['high'] / df['close'] - 1
    df['K_LOW_REL'] = df['low'] / df['close'] - 1
    df['K_OPEN_REL'] = df['open'] / df['close'] - 1
    df['H_L_Ratio'] = (df['high'] - df['low']) / df['close']
    df['C_O_Ratio'] = (df['close'] - df['open']) / df['open']
    
    df['H1_Spike'] = (df['volume'] / df['volume'].rolling(20).mean()) * (df['close'] / df['close'].shift(1) - 1)
    df['H2_Quantile'] = (df['close'] - df['close'].rolling(30).min()) / (df['close'].rolling(30).max() - df['close'].rolling(30).min() + 1e-9)
    df['H3_BiasVol'] = (df['close'].rolling(20).mean() / df['close'] - 1) / (df['close'].rolling(20).std() / df['close'].rolling(20).mean() + 1e-9)

    vol = (df['high'] - df['low']) / df['close']
    df['H4_VRegime'] = vol / (vol.rolling(60).mean() + 1e-9)
    df['H5_MQuality'] = (df['close'] / df['close'].shift(5) - 1) / (df['close'].rolling(5).std() / df['close'].rolling(5).mean() + 1e-9)
    df['H6_Slope'] = (df['close'] / df['close'].rolling(20).mean() - 1) * (df['close'] / df['close'].shift(1) - 1)
    
    df['H7_TAccel'] = (df['close'] / df['close'].shift(5) - 1) / (df['close'] / df['close'].shift(20) - 1 + 1e-9)
    df['H8_VSqueeze'] = df['close'].rolling(10).std() / (df['close'].rolling(60).std() + 1e-9)
    df['H9_VMom'] = (df['close'] / df['close'].shift(1) - 1) * (df['volume'] / df['volume'].rolling(20).mean() + 1e-9)

    df['H10_PVInt'] = (df['close'] / df['close'].shift(1) - 1) * (df['volume'] / df['volume'].rolling(10).mean())
    df['H11_KDJK'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-9)
    df['H12_UpStreak'] = (df['close'] / df['close'].shift(1) > 1).rolling(20).sum() / 20.0

    df['H13_HBreak'] = df['close'] / df['high'].rolling(20).max() - 1
    df['H14_LBreak'] = df['close'] / df['low'].rolling(20).min() - 1
    df['H15_TriMA'] = (df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1) + (df['close'].rolling(20).mean() / df['close'].rolling(60).mean() - 1)
    

    # 11. RD-Agent Gen-7 Hypotheses (Targeting 100x)
    # H16: Extreme Return Detection
    df['H16_ExtRet'] = (df['close'] / df['close'].shift(1) - 1).abs()
    # H17: Volume-Price Divergence
    df['H17_VPDiv'] = (df['volume'] / df['volume'].rolling(10).mean()) / ((df['close'] / df['close'].shift(1) - 1).abs() + 1e-9)
    # H18: Range Expansion Ratio
    df['H18_RangeExp'] = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-9)

    return df

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return
        
    print(f"Loading Model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Starting Live Inference for {SYMBOL}...")
    print(f"Resample: {RESAMPLE_FREQ}, Threshold: {THRESHOLD}")
    print("Press Ctrl+C to stop.\n")
    
    last_processed_time = None
    last_heartbeat_time = None
    
    # 持仓状态跟踪
    position = None  # None表示无持仓, "LONG"表示多单, "SHORT"表示空单
    entry_price = 0.0
    entry_time = None
    take_profit = 0.0
    stop_loss = 0.0
    trade_count = 0
    total_pnl = 0.0
    win_count = 0
    
    try:
        while True:
            df = fetch_and_prepare_data()
            if df.empty:
                time.sleep(10)
                continue
                
            current_last_time_utc = df.index[-1]
            current_last_time = current_last_time_utc + timedelta(hours=8)
            
            if last_processed_time == current_last_time:
                time.sleep(10)
                continue
                
            last_processed_time = current_last_time
            
            df_features = generate_features(df)
            
            feature_cols = []
            for n in [1, 5, 10, 20, 60]: feature_cols.append(f"ROC_{n}")
            for n in [10, 20, 60]: feature_cols.append(f"VOL_{n}")
            for n in [5, 10, 20, 60]: feature_cols.append(f"MA_{n}")
            for n in [5, 10, 20]: feature_cols.append(f"V_MA_Ratio_{n}")
            feature_cols.extend(["K_HIGH_REL", "K_LOW_REL", "K_OPEN_REL", "H_L_Ratio", "C_O_Ratio"])
            feature_cols.extend(["H1_Spike", "H2_Quantile", "H3_BiasVol"])
            feature_cols.extend(["H4_VRegime", "H5_MQuality", "H6_Slope"])
            feature_cols.extend(["H7_TAccel", "H8_VSqueeze", "H9_VMom"])
            feature_cols.extend(["H10_PVInt", "H11_KDJK", "H12_UpStreak"])
            feature_cols.extend(["H13_HBreak", "H14_LBreak", "H15_TriMA"])
            feature_cols.extend(["H16_ExtRet", "H17_VPDiv", "H18_RangeExp"])
            
            latest_features = df_features.iloc[[-1]][feature_cols]
            
            if latest_features.isna().any().any():
                print(f"[{current_last_time}] Not enough data for features. Waiting...")
                continue
                
            pred_score = model.predict(latest_features)[0]
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            atr_20 = (df['high'] - df['low']).rolling(20).mean().iloc[-1]

            # === 4.1 Check Exit ===
            if position is not None:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price
                
                if position == "LONG":
                    if current_high >= take_profit:
                        exit_signal = True
                        exit_reason = "🎯 触发止盈"
                        exit_price = take_profit
                    elif current_low <= stop_loss:
                        exit_signal = True
                        exit_reason = "🛡️ 触发止损"
                        exit_price = stop_loss
                    elif pred_score < -THRESHOLD:
                        exit_signal = True
                        exit_reason = "🔄 反向信号平仓"
                        exit_price = current_price
                elif position == "SHORT":
                    if current_low <= take_profit:
                        exit_signal = True
                        exit_reason = "🎯 触发止盈"
                        exit_price = take_profit
                    elif current_high >= stop_loss:
                        exit_signal = True
                        exit_reason = "🛡️ 触发止损"
                        exit_price = stop_loss
                    elif pred_score > THRESHOLD:
                        exit_signal = True
                        exit_reason = "🔄 反向信号平仓"
                        exit_price = current_price
                
                if exit_signal:
                    if position == "LONG":
                        pnl = exit_price - entry_price
                        pnl_pct = (exit_price / entry_price - 1) * 100
                    else:
                        pnl = entry_price - exit_price
                        pnl_pct = (entry_price / exit_price - 1) * 100
                    
                    trade_count += 1
                    total_pnl += pnl
                    if pnl > 0: win_count += 1
                    
                    emoji = "💰" if pnl > 0 else "💸"
                    print(f"\n{'═'*60}\n{emoji} 【平仓信号】{SYMBOL} - {STRATEGY_NAME}\n{'═'*60}")
                    print(f"⏰ 时间：{current_last_time}\n📊 方向：平{'多' if position=='LONG' else '空'}")
                    print(f"💰 开仓：{entry_price:.2f} | 💵 平仓：{exit_price:.2f}")
                    print(f"📈 收益：{pnl:+.2f} ({pnl_pct:+.2f}%) | 📝 原因：{exit_reason}")
                    print(f"{'═'*60}\n")
                    
                    import json
                    exit_json = {
                        "strategy": STRATEGY_NAME,
                        "time": str(current_last_time),
                        "type": "EXIT",
                        "direction": f"平{position}",
                        "entry": round(entry_price, 2),
                        "exit": round(exit_price, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "reason": exit_reason
                    }
                    print(f"SIGNAL_JSON:{json.dumps(exit_json)}")
                    
                    tg_msg = f"{emoji} **【平仓信号】** {STRATEGY_NAME}\n" \
                             f"收益: `{pnl:+.2f}` (`{pnl_pct:+.2f}%`)\n" \
                             f"原因: {exit_reason}"
                    send_telegram_alert(tg_msg)
                    
                    position = None
                    entry_price = 0.0

            # === 4.2 Check Entry ===
            signal = "观望"
            direction = None
            if pred_score > THRESHOLD:
                signal = "做多 🟢"
                direction = "LONG"
            elif pred_score < -THRESHOLD:
                signal = "做空 🔴"
                direction = "SHORT"
            
            if direction and position is None:
                if direction == "LONG":
                    new_entry = current_price
                    sl = current_price - atr_20 * 1.5
                    tp = current_price + atr_20 * 2.0
                else:
                    new_entry = current_price
                    sl = current_price + atr_20 * 1.5
                    tp = current_price - atr_20 * 2.0
                
                position = direction
                entry_price = new_entry
                stop_loss = sl
                take_profit = tp
                
                # Try to generate detailed reasons if columns exist
                reasons = []
                try:
                    if 'H14_LBreak' in df_features.columns:
                        h14 = df_features['H14_LBreak'].iloc[-1]
                        if direction == "LONG" and h14 > 0.02: reasons.append(f"脱离低点 +{h14*100:.1f}%")
                        if direction == "SHORT" and h14 < 0.01: reasons.append(f"接近低点 {h14*100:.1f}%")
                except: pass
                
                if not reasons: reasons.append(f"置信度 {abs(pred_score)*10000:.1f}bp")
                reason_str = "、".join(reasons)
                
                print(f"\n{'═'*60}\n📊 【开仓信号】{SYMBOL} - {STRATEGY_NAME}\n{'═'*60}")
                print(f"⏰ 时间：{current_last_time}\n📈 方向：{signal}")
                print(f"💰 开仓：{entry_price:.2f} | 🎯 止盈：{take_profit:.2f} | 🛡️ 止损：{stop_loss:.2f}")
                print(f"📝 理由：{reason_str}")
                print(f"{'═'*60}\n")
                
                import json
                sig_json = {
                    "strategy": STRATEGY_NAME,
                    "time": str(current_last_time),
                    "direction": direction,
                    "entry": round(entry_price, 2),
                    "tp": round(take_profit, 2),
                    "sl": round(stop_loss, 2),
                    "score": round(pred_score, 6),
                    "reason": reason_str
                }
                print(f"SIGNAL_JSON:{json.dumps(sig_json)}")
                
                tg_msg = f"📊 **【开仓信号】** {STRATEGY_NAME}\n" \
                         f"方向: **{signal}**\n" \
                         f"价格: `{entry_price:.2f}`\n" \
                         f"理由: {reason_str}"
                send_telegram_alert(tg_msg)
                
            else:
                 # Status or Hold output
                 if position:
                     pnl = (current_price - entry_price) if position == "LONG" else (entry_price - current_price)
                     pct = pnl / entry_price * 100
                     print(f"[{current_last_time}] 持仓: {position} | 浮盈: {pnl:+.2f} ({pct:+.2f}%) | 得分: {pred_score:.6f}")
                 else:
                     print(f"[{current_last_time}] 观望 | 价格: {current_price:.2f} | 得分: {pred_score:.6f}")

            # === Heartbeat ===
            import json
            fpnl = 0.0
            fpct = 0.0
            if position:
                fpnl = (current_price - entry_price) if position == "LONG" else (entry_price - current_price)
                fpct = fpnl / entry_price * 100
                
            status_data = {
                "type": "STATUS",
                "strategy": STRATEGY_NAME,
                "time": str(current_last_time),
                "price": round(current_price, 2),
                "score": round(pred_score, 6),
                "position": position,
                "entry": round(entry_price, 2) if position else 0,
                "pnl": round(fpnl, 2),
                "pnl_pct": round(fpct, 2)
            }
            print(f"SIGNAL_JSON:{json.dumps(status_data)}")

            if last_heartbeat_time != current_last_time.hour:
                 msg = f"💓 **【策略心跳】** {STRATEGY_NAME}\n" \
                       f"价格: `{current_price:.2f}`\n" \
                       f"得分: `{pred_score:.6f}`\n"
                 if position:
                     msg += f"持仓: **{position}**\n浮盈: `{fpct:+.2f}%`"
                 else:
                     msg += f"持仓: 空仓"
                 send_telegram_alert(msg)
                 last_heartbeat_time = current_last_time.hour

            time.sleep(30) 
            
    except KeyboardInterrupt:
        print("\n已停止监控。")

if __name__ == "__main__":
    main()

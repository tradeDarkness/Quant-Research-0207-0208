
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import requests
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOL = "ETHUSDT"
INTERVAL = "1m" # We fetch 1m data and resample to 10m
LOOKBACK_BARS = 200 # Need enough data for 60-period MA/ROC + Safety buffer
RESAMPLE_FREQ = "10min"
MODEL_PATH = Path(__file__).parent.resolve() / 'lgbm_model_eth_10m.pkl'
THRESHOLD = 0.0002 

# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TG_TOKEN = "8052185621:AAFT1gMhEvxZYTixeijsjLA29Q6fpnEc1xs"
TG_CHAT_ID = "6290088209" # ⚠️ 请运行 python get_tg_chat_id.py 获取并填入此处

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
        return data # [[open_time, open, high, low, close, vol, ...], ...]
    except Exception as e:
        print(f"Request Error: {e}")
        return []

def fetch_and_prepare_data(symbol=SYMBOL, interval=INTERVAL):
    # Fetch 1m data
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
        
    # Resample to 10min for Model
    # Logic matches prepare_eth_data.py
    df_10m = df.resample(RESAMPLE_FREQ).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_10m

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature Generation (Must match train_lgbm_eth.py)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_features(df):
    df = df.copy()
    
    # 1. Momentum / ROC
    for n in [1, 5, 10, 20, 60]:
        df[f'ROC_{n}'] = df['close'] / df['close'].shift(n) - 1
        
    # 2. Volatility
    for n in [10, 20, 60]:
        rolling_std = df['close'].rolling(n).std()
        rolling_mean = df['close'].rolling(n).mean()
        df[f'VOL_{n}'] = rolling_std / rolling_mean
        
    # 3. MA Divergence
    for n in [5, 10, 20, 60]:
        rolling_mean = df['close'].rolling(n).mean()
        df[f'MA_{n}'] = df['close'] / rolling_mean - 1
        
    # 4. Volume Features
    for n in [5, 10, 20]:
        rolling_v_mean = df['volume'].rolling(n).mean()
        df[f'V_MA_Ratio_{n}'] = df['volume'] / rolling_v_mean

    # 5. K-Line Shapes
    df['K_HIGH_REL'] = df['high'] / df['close'] - 1
    df['K_LOW_REL'] = df['low'] / df['close'] - 1
    df['K_OPEN_REL'] = df['open'] / df['close'] - 1
    df['H_L_Ratio'] = (df['high'] - df['low']) / df['close']
    df['C_O_Ratio'] = (df['close'] - df['open']) / df['open']
    
    # 6. RD-Agent Gen-1 Hypotheses
    # H1: Energy Spike
    df['H1_Spike'] = (df['volume'] / df['volume'].rolling(20).mean()) * (df['close'] / df['close'].shift(1) - 1)
    # H2: Price Quantile Position
    df['H2_Quantile'] = (df['close'] - df['close'].rolling(30).min()) / (df['close'].rolling(30).max() - df['close'].rolling(30).min() + 1e-9)
    # H3: Bias/Volatility
    df['H3_BiasVol'] = (df['close'].rolling(20).mean() / df['close'] - 1) / (df['close'].rolling(20).std() / df['close'].rolling(20).mean() + 1e-9)

    # 7. RD-Agent Gen-2 Hypotheses (Advanced Volatility)
    # H4: VRegime
    vol = (df['high'] - df['low']) / df['close']
    df['H4_VRegime'] = vol / (vol.rolling(60).mean() + 1e-9)
    # H5: MQuality
    df['H5_MQuality'] = (df['close'] / df['close'].shift(5) - 1) / (df['close'].rolling(5).std() / df['close'].rolling(5).mean() + 1e-9)
    # H6: Slope
    df['H6_Slope'] = (df['close'] / df['close'].rolling(20).mean() - 1) * (df['close'] / df['close'].shift(1) - 1)
    
    # 8. RD-Agent Gen-3 Hypotheses
    # H7: Trend Acceleration
    df['H7_TAccel'] = (df['close'] / df['close'].shift(5) - 1) / (df['close'] / df['close'].shift(20) - 1 + 1e-9)
    # H8: Volatility Squeeze
    df['H8_VSqueeze'] = df['close'].rolling(10).std() / (df['close'].rolling(60).std() + 1e-9)
    # H9: Volume-Confirmed Momentum
    df['H9_VMom'] = (df['close'] / df['close'].shift(1) - 1) * (df['volume'] / df['volume'].rolling(20).mean() + 1e-9)

    # 9. RD-Agent Gen-4 Hypotheses (Ultimate Strategy)
    # H10: Price-Volume Interaction
    df['H10_PVInt'] = (df['close'] / df['close'].shift(1) - 1) * (df['volume'] / df['volume'].rolling(10).mean())
    # H11: Dynamic Range Position (KDJ-Ref)
    df['H11_KDJK'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-9)
    # H12: Trend Persistence (ROC Mean)
    df['H12_UpStreak'] = (df['close'] / df['close'].shift(1) > 1).rolling(20).sum() / 20.0

    # 10. RD-Agent Gen-6 Hypotheses (Breakthrough 1000%+)
    # H13: High Breakout Signal
    df['H13_HBreak'] = df['close'] / df['high'].rolling(20).max() - 1
    # H14: Low Breakout Signal
    df['H14_LBreak'] = df['close'] / df['low'].rolling(20).min() - 1
    # H15: Triple MA Spread
    df['H15_TriMA'] = (df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1) + (df['close'].rolling(20).mean() / df['close'].rolling(60).mean() - 1)

    # 11. RD-Agent Gen-7 Hypotheses (Targeting 100x)
    # H16: Extreme Return Detection
    df['H16_ExtRet'] = (df['close'] / df['close'].shift(1) - 1).abs()
    # H17: Volume-Price Divergence
    df['H17_VPDiv'] = (df['volume'] / df['volume'].rolling(10).mean()) / ((df['close'] / df['close'].shift(1) - 1).abs() + 1e-9)
    # H18: Range Expansion Ratio
    df['H18_RangeExp'] = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-9)
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Main Loop
# ═══════════════════════════════════════════════════════════════════════════════
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
            # 1. Fetch
            df = fetch_and_prepare_data()
            if df.empty:
                time.sleep(10)
                continue
                
            # Check if new bar arrived
            current_last_time = df.index[-1]
            
            if last_processed_time == current_last_time:
                # Still the same bar, wait a bit
                time.sleep(10)
                continue
                
            last_processed_time = current_last_time
            
            # 2. Generate Features
            # We need to generate for the whole DF to handle rolling windows correctly
            # preparing feature columns
            df_features = generate_features(df)
            
            # Select feature columns in specific order (Must match training!)
            feature_cols = []
            # ROC
            for n in [1, 5, 10, 20, 60]: feature_cols.append(f"ROC_{n}")
            # VOL
            for n in [10, 20, 60]: feature_cols.append(f"VOL_{n}")
            # MA
            for n in [5, 10, 20, 60]: feature_cols.append(f"MA_{n}")
            # Volume Features
            for n in [5, 10, 20]: feature_cols.append(f"V_MA_Ratio_{n}")
            # K-Line Shapes
            feature_cols.append("K_HIGH_REL")
            feature_cols.append("K_LOW_REL")
            feature_cols.append("K_OPEN_REL")
            feature_cols.append("H_L_Ratio")
            feature_cols.append("C_O_Ratio")
            # RD-Agent Gen-1 Hypotheses
            feature_cols.append("H1_Spike")
            feature_cols.append("H2_Quantile")
            feature_cols.append("H3_BiasVol")
            # RD-Agent Gen-2 Hypotheses
            feature_cols.append("H4_VRegime")
            feature_cols.append("H5_MQuality")
            feature_cols.append("H6_Slope")
            # RD-Agent Gen-3 Hypotheses
            feature_cols.append("H7_TAccel")
            feature_cols.append("H8_VSqueeze")
            feature_cols.append("H9_VMom")
            # RD-Agent Gen-4 Hypotheses
            feature_cols.append("H10_PVInt")
            feature_cols.append("H11_KDJK")
            feature_cols.append("H12_UpStreak")
            # RD-Agent Gen-6 Hypotheses
            feature_cols.append("H13_HBreak")
            feature_cols.append("H14_LBreak")
            feature_cols.append("H15_TriMA")
            # RD-Agent Gen-7 Hypotheses
            feature_cols.append("H16_ExtRet")
            feature_cols.append("H17_VPDiv")
            feature_cols.append("H18_RangeExp")
            
            # Get the very last row (Current completed bar)
            # Actually, if we are trading "on close", we trade based on the just-closed bar.
            # df.index[-1] is the open time of the last bar? 
            # Binance kline open time. So if now is 10:11, the last 10m bar is 10:00-10:10.
            # Its open_time is 10:00. 
            # We want to predict for the NEXT bar (10:10-10:20) using data up to 10:10.
            # So taking the last row of df is correct (it represents the most recent completed period).
            
            latest_features = df_features.iloc[[-1]][feature_cols]
            
            # Check for NaNs (e.g. not enough history for rolling 60)
            if latest_features.isna().any().any():
                print(f"[{current_last_time}] Not enough data for features. Waiting...")
                continue
                
            # 3. Predict
            pred_score = model.predict(latest_features)[0]
            
            # 4. Signal Logic with detailed info
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            atr_20 = (df['high'] - df['low']).rolling(20).mean().iloc[-1]  # ATR for SL/TP
            
            # === 4.1 检查是否需要平仓 (Stateful Logic) ===
            if position is not None:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price
                
                if position == "LONG":
                    # 检查止盈
                    if current_high >= take_profit:
                        exit_signal = True
                        exit_reason = "🎯 触发止盈"
                        exit_price = take_profit
                    # 检查止损
                    elif current_low <= stop_loss:
                        exit_signal = True
                        exit_reason = "🛡️ 触发止损"
                        exit_price = stop_loss
                    # 检查反向信号
                    elif pred_score < -THRESHOLD:
                        exit_signal = True
                        exit_reason = "🔄 反向信号平仓"
                        exit_price = current_price
                        
                elif position == "SHORT":
                    # 检查止盈
                    if current_low <= take_profit:
                        exit_signal = True
                        exit_reason = "🎯 触发止盈"
                        exit_price = take_profit
                    # 检查止损
                    elif current_high >= stop_loss:
                        exit_signal = True
                        exit_reason = "🛡️ 触发止损"
                        exit_price = stop_loss
                    # 检查反向信号
                    elif pred_score > THRESHOLD:
                        exit_signal = True
                        exit_reason = "🔄 反向信号平仓"
                        exit_price = current_price
                
                # 执行平仓
                if exit_signal:
                    # 计算收益
                    if position == "LONG":
                        pnl = exit_price - entry_price
                        pnl_pct = (exit_price / entry_price - 1) * 100
                    else:  # SHORT
                        pnl = entry_price - exit_price
                        pnl_pct = (entry_price / exit_price - 1) * 100
                    
                    trade_count += 1
                    total_pnl += pnl
                    if pnl > 0:
                        win_count += 1
                    
                    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
                    
                    # 平仓日志
                    emoji = "💰" if pnl > 0 else "💸"
                    print("\n" + "═" * 60)
                    print(f"{emoji} 【平仓信号】{SYMBOL}")
                    print("═" * 60)
                    print(f"⏰ 时间：{current_last_time}")
                    print(f"📊 方向：平{'多' if position == 'LONG' else '空'}")
                    print(f"💰 开仓价：{entry_price:.2f} USDT")
                    print(f"💵 平仓价：{exit_price:.2f} USDT")
                    print(f"📈 收益：{pnl:+.2f} 点 ({pnl_pct:+.2f}%)")
                    print(f"📝 平仓原因：{exit_reason}")
                    print(f"📊 累计交易：{trade_count} 笔 | 胜率：{win_rate:.1f}% | 总收益：{total_pnl:+.2f}")
                    print("═" * 60 + "\n")
                    
                    # JSON for Dashboard
                    import json
                    exit_json = {
                        "strategy": "Gen-7-117x",
                        "time": str(current_last_time),
                        "type": "EXIT",
                        "direction": f"平{position}",
                        "entry": round(entry_price, 2),
                        "exit": round(exit_price, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "reason": exit_reason,
                        "trade_count": trade_count,
                        "win_rate": round(win_rate, 1)
                    }
                    print(f"SIGNAL_JSON:{json.dumps(exit_json)}")
                    
                    # Telegram 平仓通知
                    tg_exit = f"{emoji} **【平仓信号】**\n\n" \
                              f"币对：`{SYMBOL}`\n" \
                              f"时间：`{current_last_time}`\n" \
                              f"方向：**平{'多' if position == 'LONG' else '空'}**\n\n" \
                              f"💰 开仓价：`{entry_price:.2f}` USDT\n" \
                              f"💵 平仓价：`{exit_price:.2f}` USDT\n" \
                              f"📈 收益：`{pnl:+.2f}` 点 (`{pnl_pct:+.2f}%`)\n\n" \
                              f"📝 原因：{exit_reason}\n" \
                              f"📊 累计：{trade_count} 笔 | 胜率：{win_rate:.1f}%"
                    send_telegram_alert(tg_exit)
                    
                    # 清空持仓状态
                    position = None
                    entry_price = 0.0
                    entry_time = None

            # === 4.2 检查是否需要开仓 ===
            signal = "观望"
            direction = None
            if pred_score > THRESHOLD:
                signal = "做多 🟢"
                direction = "LONG"
            elif pred_score < -THRESHOLD:
                signal = "做空 🔴"
                direction = "SHORT"
            
            # 只有无持仓时才开新仓
            if direction and position is None:
                if direction == "LONG":
                    new_entry_price = current_price
                    new_stop_loss = current_price - atr_20 * 1.5
                    new_take_profit = current_price + atr_20 * 2.0
                    expected_profit = atr_20 * 2.0
                else:  # SHORT
                    new_entry_price = current_price
                    new_stop_loss = current_price + atr_20 * 1.5
                    new_take_profit = current_price - atr_20 * 2.0
                    expected_profit = atr_20 * 2.0
                
                # 更新持仓状态
                position = direction
                entry_price = new_entry_price
                entry_time = current_last_time
                take_profit = new_take_profit
                stop_loss = new_stop_loss
                
                # Generate entry reason based on top factors
                reasons = []
                h14_val = df_features['H14_LBreak'].iloc[-1]
                h11_val = df_features['H11_KDJK'].iloc[-1]
                h6_val = df_features['H6_Slope'].iloc[-1]
                
                if direction == "LONG":
                    if h14_val > 0.02:
                        reasons.append(f"脱离近期低点 +{h14_val*100:.1f}%")
                    if h11_val > 0.7:
                        reasons.append(f"强势区间位置 {h11_val:.2f}")
                    if h6_val > 0:
                        reasons.append("正向斜率确认")
                else:
                    if h14_val < 0.01:
                        reasons.append(f"接近近期低点 {h14_val*100:.1f}%")
                    if h11_val < 0.3:
                        reasons.append(f"弱势区间位置 {h11_val:.2f}")
                    if h6_val < 0:
                        reasons.append("负向斜率确认")
                
                if not reasons:
                    reasons.append(f"模型置信度 {abs(pred_score)*10000:.1f}bp")
                
                reason_str = "、".join(reasons)
                
                # Detailed Chinese output
                print("\n" + "═" * 60)
                print(f"📊 【AI 交易信号】{SYMBOL}")
                print("═" * 60)
                print(f"⏰ 时间：{current_last_time}")
                print(f"📈 方向：{signal}")
                print(f"💰 开仓点位：{entry_price:.2f} USDT")
                print(f"🎯 止盈点位：{take_profit:.2f} USDT")
                print(f"🛡️ 止损点位：{stop_loss:.2f} USDT")
                print(f"💎 预期收益：{expected_profit:.2f} 点 (约 {expected_profit/current_price*100:.2f}%)")
                print(f"🔍 模型得分：{pred_score:.6f}")
                print(f"📝 开仓理由：{reason_str}")
                print("═" * 60 + "\n")
                
                # JSON output for Dashboard backend to capture
                import json
                signal_json = {
                    "strategy": "Gen-7-117x",
                    "time": str(current_last_time),
                    "direction": direction,
                    "entry": round(entry_price, 2),
                    "tp": round(take_profit, 2),
                    "sl": round(stop_loss, 2),
                    "score": round(pred_score, 6),
                    "reason": reason_str
                }
                print(f"SIGNAL_JSON:{json.dumps(signal_json)}")
                
                # Telegram Alert with Chinese
                tg_msg = f"📊 **【AI 交易信号】**\n\n" \
                         f"币对：`{SYMBOL}`\n" \
                         f"时间：`{current_last_time}`\n" \
                         f"方向：**{signal}**\n\n" \
                         f"💰 开仓：`{entry_price:.2f}` USDT\n" \
                         f"🎯 止盈：`{take_profit:.2f}` USDT\n" \
                         f"🛡️ 止损：`{stop_loss:.2f}` USDT\n" \
                         f"💎 预期收益：`{expected_profit:.2f}` 点\n\n" \
                         f"📝 理由：{reason_str}\n" \
                         f"🔍 得分：`{pred_score:.6f}`"
                send_telegram_alert(tg_msg)
            else:
                # 显示当前状态
                if position:
                    # 计算浮动盈亏
                    if position == "LONG":
                        floating_pnl = current_price - entry_price
                        floating_pct = (current_price / entry_price - 1) * 100
                    else:
                        floating_pnl = entry_price - current_price
                        floating_pct = (entry_price / current_price - 1) * 100
                    print(f"[{current_last_time}] 价格: {current_price:.2f} | 持仓: {position} | 浮动: {floating_pnl:+.2f} ({floating_pct:+.2f}%) | 得分: {pred_score:.6f}")
                else:
                    print(f"[{current_last_time}] 价格: {current_price:.2f} | 得分: {pred_score:.6f} | 信号: {signal}")
            
            # === Heartbeat & Status Update ===
            floating_pnl = 0.0
            floating_pct = 0.0
            if position:
                if position == "LONG":
                    floating_pnl = current_price - entry_price
                    floating_pct = (current_price / entry_price - 1) * 100
                else:
                    floating_pnl = entry_price - current_price
                    floating_pct = (entry_price / current_price - 1) * 100

            import json
            status_data = {
                "type": "STATUS",
                "strategy": "Gen-7-117x",
                "time": str(current_last_time),
                "price": round(current_price, 2),
                "score": round(pred_score, 6),
                "position": position,
                "entry": round(entry_price, 2) if position else 0,
                "pnl": round(floating_pnl, 2),
                "pnl_pct": round(floating_pct, 2)
            }
            print(f"SIGNAL_JSON:{json.dumps(status_data)}")
            
            # Telegram Heartbeat (Every hour)
            if last_heartbeat_time != current_last_time.hour:
                 msg = f"💓 **【策略心跳】** Gen-7-117x\n" \
                       f"价格: `{current_price:.2f}`\n" \
                       f"得分: `{pred_score:.6f}`\n"
                 if position:
                     msg += f"持仓: **{position}**\n浮盈: `{floating_pct:+.2f}%`"
                 else:
                     msg += f"持仓: 空仓"
                 
                 send_telegram_alert(msg)
                 last_heartbeat_time = current_last_time.hour
            
            # Sleep to avoid spam
            time.sleep(30) 
            
    except KeyboardInterrupt:
        print("\n已停止监控。")

if __name__ == "__main__":
    main()

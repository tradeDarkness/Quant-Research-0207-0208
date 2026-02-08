#!/usr/bin/env python3
"""
Telegram Bot 对接：接收 /predict 等命令，运行 BTC 15m 预测并回复结果。
Token 从环境变量 BOT_TOKEN 或 .env 读取，请勿提交到仓库。
"""
import os
import sys
import time
import json
import requests

# 加载 .env（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BOT_TOKEN = os.environ.get("BOT_TOKEN")
TELEGRAM_API = "https://api.telegram.org/bot{token}"

def api_url(method):
    return TELEGRAM_API.format(token=BOT_TOKEN) + "/" + method

def send_message(chat_id, text):
    r = requests.post(api_url("sendMessage"), json={"chat_id": chat_id, "text": text}, timeout=30)
    return r.json() if r.ok else None

def get_updates(offset=None):
    params = {"timeout": 60}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(api_url("getUpdates"), params=params, timeout=65)
    if not r.ok:
        return []
    data = r.json()
    return data.get("result") or []

def run_prediction():
    """运行一次预测，返回可发送给用户的文本；失败返回错误信息。"""
    try:
        # 确保当前目录在项目根，以便 import 和找 CSV/模型
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import live_polymarket_qlib as live_module
        model = live_module.LiveModel()
        out = model.predict_next_dict()
        if not out:
            return "❌ 预测失败（拉取数据或模型出错）"
        s = out.get("signal", "中性 (NEUTRAL)")
        sc = out.get("score", 0)
        dt = out.get("datetime", "")
        pr = out.get("price", 0)
        
        # Emoji mapping
        if "强烈看涨" in s:
            emoji = "🚀🔥 [强势信号]" 
        elif "看涨" in s:
            emoji = "🟢 [建议关注]"
        elif "看跌" in s:
            emoji = "🔴 [风险警告]"
        else:
            emoji = "⚪ [震荡观望]"
            
        return (
            f"{emoji}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏷 标的: BTC 15m\n"
            f"⏰ 时间: {dt}\n"
            f"💰 现价: ${pr:,.2f}\n"
            f"📈 信号: {s}\n"
            f"🎯 置信度: {sc:.4f}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💡 基于 Phase 4 高精准 Alpha 模型"
        )
    except Exception as e:
        return f"❌ 预测异常: {e}"

def main():
    if not BOT_TOKEN:
        print("请设置环境变量 BOT_TOKEN，或在项目目录下创建 .env 并写入 BOT_TOKEN=你的token")
        sys.exit(1)
    print("Bot 已启动，等待命令 (/start /help /predict)...")
    offset = None
    while True:
        try:
            updates = get_updates(offset)
            for u in updates:
                offset = u.get("update_id", 0) + 1
                msg = u.get("message") or {}
                chat_id = msg.get("chat", {}).get("id")
                text = (msg.get("text") or "").strip()
                if not chat_id or not text:
                    continue
                if text == "/start":
                    send_message(chat_id, "你好，我是小叮当 BTC 15m 预测机器人。\n发送 /predict 获取最新预测，/help 查看帮助。")
                elif text == "/help":
                    send_message(chat_id, "/predict - 拉取最新 K 线并输出下一根 15m 涨跌预测\n/help - 本帮助\n/start - 欢迎语")
                elif text == "/predict":
                    send_message(chat_id, "正在拉取数据并计算预测，请稍候…")
                    reply = run_prediction()
                    send_message(chat_id, reply)
                else:
                    send_message(chat_id, "未知命令。发送 /help 查看可用命令。")
        except KeyboardInterrupt:
            print("已退出")
            break
        except Exception as e:
            print(f"轮询异常: {e}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()

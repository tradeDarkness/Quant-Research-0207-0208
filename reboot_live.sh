#!/bin/bash

# ==========================================================
# ETH Strategy & Dashboard 统一管理脚本
# ==========================================================

# 基础路径
BASE_DIR="/Users/zhangzc/7/20260123"
STRATEGY_DIR="$BASE_DIR/0210_ETH_Optimization"
DASHBOARD_DIR="$BASE_DIR/0210_ETH_Dashboard"
VENV_PYTHON="$BASE_DIR/.venv/bin/python"
LOG_DIR="$DASHBOARD_DIR/backend/logs"

echo "🚀 开始重启 ETH 实盘系统..."

# 1. 停止旧进程
echo "🛑 正在停止旧进程..."
pkill -f "live_inference.py"
pkill -f "live_inference_ws.py"
pkill -f "main.py"
pkill -f "ngrok"
sleep 2

# 2. 启动后端仪表盘
echo "🖥️ 启动后端仪表盘 (Port 8000)..."
cd "$DASHBOARD_DIR/backend"
nohup "$VENV_PYTHON" main.py > backend_restart.log 2>&1 &

# 3. 启动实盘引擎
echo "🤖 启动 AI 策略引擎 (WebSocket HighPerf)..."
cd "$STRATEGY_DIR"
nohup "$VENV_PYTHON" -u live_inference_ws.py >> "$LOG_DIR/gen10_eth.log" 2>&1 &

# 4. 重新开启内网穿透 (可选)
echo "🌐 启动 ngrok 隧道..."
nohup ngrok http 3000 --host-header=localhost --log=stdout > /tmp/ngrok.log 2>&1 &

echo "----------------------------------------------------"
echo "✅ 所有系统已重启！"
echo "📊 后端进程: $(pgrep -f 'main.py' || echo '失败')"
echo "📈 策略进程: $(pgrep -f 'live_inference_ws.py' || echo '失败')"
echo "🌐 ngrok 进程: $(pgrep -f 'ngrok' || echo '失败')"
echo "----------------------------------------------------"
echo "💡 查看日志: tail -f $LOG_DIR/gen10_eth.log"

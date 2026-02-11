#!/bin/bash

# 定义基础路径
BASE_DIR="/Users/zhangzc/7/20260123"
BACKEND_DIR="$BASE_DIR/0210_ETH_Dashboard/backend"
STRATEGY_DIR="$BASE_DIR/0210_ETH_Optimization"
PYTHON_EXEC="$BASE_DIR/.venv/bin/python"

echo "🛑 正在精准停止旧进程..."
# 仅根据特定的脚本文件名杀死进程，避免误伤同样使用 Python 的软件 (如 VPN 客户端)
pkill -f "0210_ETH_Dashboard/backend/main.py" 2>/dev/null
pkill -f "0210_ETH_Optimization/live_inference_ws.py" 2>/dev/null
# 释放端口并停止可能的残留
# 释放端口并停止可能的残留 (仅在 pkill 失败时手动检查，避免误杀 VPN)
# lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "🚀 启动系统 (独立终端窗口)..."

# 1. 启动 Backend (在新窗口)
cat <<EOF > /tmp/start_backend.sh
#!/bin/bash
echo "🖥️  Dashboard Backend Starting..."
cd "$BACKEND_DIR"
"$PYTHON_EXEC" main.py
exec $SHELL
EOF
chmod +x /tmp/start_backend.sh
open -a Terminal /tmp/start_backend.sh

# 2. 策略控制台 (监视由 Backend 管理的策略输出)
cat <<EOF > /tmp/start_strategy.sh
#!/bin/bash
STRATEGY_LOG="$BACKEND_DIR/logs/gen10_eth.log"
echo "🤖 等待策略引擎启动并生成日志..."
sleep 3
if [ ! -f "\$STRATEGY_LOG" ]; then
    mkdir -p "\$(dirname "\$STRATEGY_LOG")"
    touch "\$STRATEGY_LOG"
fi
echo "✅ 正在实时追踪策略输出 (Gen-10 EPIC):"
echo "------------------------------------------------"
tail -f "\$STRATEGY_LOG"
EOF
chmod +x /tmp/start_strategy.sh
open -a Terminal /tmp/start_strategy.sh

echo "✅ commands sent. Check new terminal windows."

#!/bin/bash
# AI 策略监控仪表盘 - 启动脚本

echo "🚀 正在启动 AI 策略监控仪表盘..."
echo ""

# 项目路径
DASHBOARD_DIR="/Users/zhangzc/7/20260123/0208_Dashboard"
VENV_PATH="/Users/zhangzc/7/20260123/.venv/bin/python"

# 1. 安装后端依赖
echo "📦 检查后端依赖..."
$VENV_PATH -m pip install fastapi uvicorn websockets pydantic --quiet

# 2. 初始化数据库
echo "🗄️ 初始化数据库..."
cd "$DASHBOARD_DIR/backend"
$VENV_PATH database.py

# 3. 启动后端
echo "🔧 启动后端服务 (http://localhost:8000)..."
$VENV_PATH -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

# 4. 安装前端依赖并启动
echo "📦 安装前端依赖..."
cd "$DASHBOARD_DIR/frontend"
npm install --silent

echo "🌐 启动前端服务 (http://localhost:3000)..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ AI 策略监控仪表盘启动成功！"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 前端地址：http://localhost:3000"
echo "🔧 后端地址：http://localhost:8000"
echo "📡 WebSocket：ws://localhost:8000/ws"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo ""

# 捕获退出信号
trap "echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# 保持前台运行
wait

#!/bin/bash

# RD-Agent Start Script for Gemini Integration
# Usage: ./start_rdagent.sh

LOG_FILE="rdagent.log"

echo "==========================================="
echo "   RD-Agent Auto-Launcher (Gemini Mode)    "
echo "==========================================="

# 1. Clean up previous instances
echo "[1/4] Checking for existing processes..."
PID=$(pgrep -f "rdagent fin_factor")
if [ -n "$PID" ]; then
    echo "Stopping existing rdagent process (PID: $PID)..."
    kill $PID
    sleep 2
else
    echo "No running rdagent process found."
fi

# 2. Check and Apply Patches
echo "[2/4] Verifying library patches..."
python3 patch_rdagent.py
if [ $? -ne 0 ]; then
    echo "❌ Patching failed! Please check patch_rdagent.py."
    exit 1
fi

# 3. Setup Environment
echo "[3/4] Loading environment variables..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ .env file not found!"
    exit 1
fi

# Ensure conda mock is executable
chmod +x .venv_rdagent/bin/conda

# 4. Start RD-Agent
echo "[4/4] Starting RD-Agent..."
nohup .venv_rdagent/bin/rdagent fin_factor > "$LOG_FILE" 2>&1 &
NEW_PID=$!

echo "✅ RD-Agent started successfully with PID: $NEW_PID"
echo "   Output is being redirected to $LOG_FILE"
echo "   Tailing log now (Ctrl+C to exit log view, process will keep running)..."
echo "==========================================="
sleep 2
tail -f "$LOG_FILE"

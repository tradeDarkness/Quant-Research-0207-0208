"""
AI 策略监控仪表盘 - FastAPI 后端
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import asyncio
from pathlib import Path
from datetime import datetime

from database import init_db, add_trade, get_trades, add_equity_point, get_equity_curve
from database import update_strategy_status, get_strategy_status, get_stats, close_latest_trade
from process_manager import ProcessManager

# 初始化
app = FastAPI(title="AI Strategy Dashboard", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
init_db()

# WebSocket 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# 主事件循环引用（用于线程安全的异步调用）
main_loop = None

@app.on_event("startup")
async def startup_event():
    """启动时保存主事件循环引用"""
    global main_loop
    main_loop = asyncio.get_event_loop()

# 全局策略状态缓存
strategy_states = {}

# 进程管理器
def on_signal_received(signal_data: dict):
    """处理接收到的交易信号（开仓、平仓或状态更新）"""
    signal_type = signal_data.get('type', 'ENTRY')
    strategy_id = signal_data.get('strategy_id', 'unknown')
    
    if signal_type == 'STATUS':
        # 更新实盘状态缓存
        strategy_states[strategy_id] = signal_data
        # 不写入数据库，仅广播
    elif signal_type == 'EXIT':
        # 处理平仓信号
        exit_price = signal_data.get('exit', 0)
        exit_time = signal_data.get('time', datetime.now().isoformat())
        pnl = signal_data.get('pnl', 0)
        
        success = close_latest_trade(strategy_id, exit_price, exit_time, pnl)
        if success:
            print(f"[DB] Closed trade for {strategy_id}, PnL: {pnl}")
        else:
            print(f"[DB] No open trade found for {strategy_id}")
    else:
        # 处理开仓信号
        trade_id = add_trade(
            strategy_id=strategy_id,
            timestamp=signal_data.get('time', datetime.now().isoformat()),
            direction=signal_data.get('direction', 'UNKNOWN'),
            entry_price=signal_data.get('entry', 0),
            take_profit=signal_data.get('tp'),
            stop_loss=signal_data.get('sl'),
            score=signal_data.get('score'),
            reason=signal_data.get('reason', '')
        )
        signal_data['trade_id'] = trade_id
        print(f"[DB] Saved trade {trade_id} for {strategy_id}")
    
    # 广播到所有 WebSocket 客户端
    if main_loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "signal", "data": signal_data}),
            main_loop
        )

process_manager = ProcessManager(on_signal=on_signal_received)
process_manager.load_strategies()

# ═══════════════════════════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "AI Strategy Dashboard API", "version": "1.0.0"}

# ───────────────────────────────────────────────────────────────────────────────
# 策略管理
# ───────────────────────────────────────────────────────────────────────────────

@app.get("/api/strategies")
async def list_strategies():
    """获取所有策略列表"""
    config_path = Path(__file__).parent.parent / "strategies.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 添加运行状态
    status = process_manager.get_status()
    for strategy in config['strategies']:
        strategy['running'] = status.get(strategy['id'], {}).get('running', False)
        strategy['pid'] = status.get(strategy['id'], {}).get('pid')
        # 添加实时状态信息
        strategy['realtime'] = strategy_states.get(strategy['id'], {})
    
    return config['strategies']

@app.post("/api/strategies/{strategy_id}/start")
async def start_strategy(strategy_id: str):
    """启动指定策略"""
    success = process_manager.start_strategy(strategy_id)
    if success:
        update_strategy_status(strategy_id, status='RUNNING', pid=process_manager.strategies[strategy_id].process.pid)
        await manager.broadcast({"type": "status_change", "strategy_id": strategy_id, "status": "RUNNING"})
        return {"success": True, "message": f"Strategy {strategy_id} started"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to start strategy {strategy_id}")

@app.post("/api/strategies/{strategy_id}/stop")
async def stop_strategy(strategy_id: str):
    """停止指定策略"""
    success = process_manager.stop_strategy(strategy_id)
    if success:
        update_strategy_status(strategy_id, status='STOPPED')
        await manager.broadcast({"type": "status_change", "strategy_id": strategy_id, "status": "STOPPED"})
        return {"success": True, "message": f"Strategy {strategy_id} stopped"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to stop strategy {strategy_id}")

@app.post("/api/strategies/start-all")
async def start_all_strategies():
    """启动所有策略"""
    results = process_manager.start_all()
    for strategy_id, success in results.items():
        if success:
            update_strategy_status(strategy_id, status='RUNNING')
    return {"success": True, "results": results}

@app.post("/api/strategies/stop-all")
async def stop_all_strategies():
    """停止所有策略"""
    results = process_manager.stop_all()
    for strategy_id, success in results.items():
        if success:
            update_strategy_status(strategy_id, status='STOPPED')
    return {"success": True, "results": results}

@app.get("/api/strategies/{strategy_id}/status")
async def get_single_strategy_status(strategy_id: str):
    """获取单个策略状态"""
    status = process_manager.get_status()
    if strategy_id not in status:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return status[strategy_id]

# ───────────────────────────────────────────────────────────────────────────────
# 交易记录
# ───────────────────────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def list_trades(strategy_id: Optional[str] = None, limit: int = 100):
    """获取交易记录"""
    trades = get_trades(strategy_id, limit)
    return trades

@app.get("/api/trades/{strategy_id}/stats")
async def get_trade_stats(strategy_id: str):
    """获取策略统计数据"""
    stats = get_stats(strategy_id)
    return stats

# ───────────────────────────────────────────────────────────────────────────────
# 收益曲线
# ───────────────────────────────────────────────────────────────────────────────

@app.get("/api/equity/{strategy_id}")
async def get_equity(strategy_id: str, hours: int = 24):
    """获取收益曲线数据"""
    curve = get_equity_curve(strategy_id, hours)
    return curve

# ───────────────────────────────────────────────────────────────────────────────
# WebSocket
# ───────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接，用于实时推送信号"""
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接，接收客户端消息（如心跳）
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ═══════════════════════════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
数据库模型和操作
SQLite 存储交易信号和收益记录
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

DB_PATH = Path(__file__).parent / "dashboard.db"

def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 交易信号表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            take_profit REAL,
            stop_loss REAL,
            score REAL,
            reason TEXT,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_time TEXT,
            pnl REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 收益曲线表（每分钟记录）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equity_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            equity REAL NOT NULL,
            drawdown REAL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 策略状态表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_status (
            strategy_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'STOPPED',
            pid INTEGER,
            start_time TEXT,
            total_trades INTEGER DEFAULT 0,
            win_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def add_trade(
    strategy_id: str,
    timestamp: str,
    direction: str,
    entry_price: float,
    take_profit: float = None,
    stop_loss: float = None,
    score: float = None,
    reason: str = None
) -> int:
    """添加新交易记录"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO trades (strategy_id, timestamp, direction, entry_price, take_profit, stop_loss, score, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (strategy_id, timestamp, direction, entry_price, take_profit, stop_loss, score, reason))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def close_trade(trade_id: int, exit_price: float, exit_time: str, pnl: float):
    """关闭交易（记录平仓）"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE trades 
        SET status = 'CLOSED', exit_price = ?, exit_time = ?, pnl = ?
        WHERE id = ?
    """, (exit_price, exit_time, pnl, trade_id))
    
    conn.commit()
    conn.close()

def close_latest_trade(strategy_id: str, exit_price: float, exit_time: str, pnl: float) -> bool:
    """关闭策略的最新开仓交易"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 找到该策略最新的 OPEN 状态交易
    cursor.execute("""
        SELECT id FROM trades 
        WHERE strategy_id = ? AND status = 'OPEN' 
        ORDER BY timestamp DESC LIMIT 1
    """, (strategy_id,))
    
    row = cursor.fetchone()
    if row:
        trade_id = row['id']
        cursor.execute("""
            UPDATE trades 
            SET status = 'CLOSED', exit_price = ?, exit_time = ?, pnl = ?
            WHERE id = ?
        """, (exit_price, exit_time, pnl, trade_id))
        conn.commit()
        conn.close()
        print(f"[DB] Closed trade {trade_id} for {strategy_id}, PnL: {pnl}")
        return True
    
    conn.close()
    return False

def get_trades(strategy_id: str = None, limit: int = 100) -> List[Dict]:
    """获取交易记录"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if strategy_id:
        cursor.execute("""
            SELECT * FROM trades WHERE strategy_id = ? ORDER BY timestamp DESC LIMIT ?
        """, (strategy_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_equity_point(strategy_id: str, timestamp: str, equity: float, drawdown: float = 0):
    """添加收益曲线数据点"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO equity_curve (strategy_id, timestamp, equity, drawdown)
        VALUES (?, ?, ?, ?)
    """, (strategy_id, timestamp, equity, drawdown))
    
    conn.commit()
    conn.close()

def get_equity_curve(strategy_id: str, hours: int = 24) -> List[Dict]:
    """获取收益曲线数据"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, equity, drawdown 
        FROM equity_curve 
        WHERE strategy_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (strategy_id, hours * 6))  # 10分钟间隔
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in reversed(rows)]

def update_strategy_status(
    strategy_id: str,
    status: str = None,
    pid: int = None,
    total_trades: int = None,
    win_trades: int = None,
    total_pnl: float = None
):
    """更新策略状态"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 先检查是否存在
    cursor.execute("SELECT * FROM strategy_status WHERE strategy_id = ?", (strategy_id,))
    exists = cursor.fetchone() is not None
    
    if exists:
        updates = []
        values = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if pid is not None:
            updates.append("pid = ?")
            values.append(pid)
        if total_trades is not None:
            updates.append("total_trades = ?")
            values.append(total_trades)
        if win_trades is not None:
            updates.append("win_trades = ?")
            values.append(win_trades)
        if total_pnl is not None:
            updates.append("total_pnl = ?")
            values.append(total_pnl)
        
        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(strategy_id)
        
        cursor.execute(f"""
            UPDATE strategy_status SET {', '.join(updates)} WHERE strategy_id = ?
        """, values)
    else:
        cursor.execute("""
            INSERT INTO strategy_status (strategy_id, status, pid, start_time)
            VALUES (?, ?, ?, ?)
        """, (strategy_id, status or 'STOPPED', pid, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

def get_strategy_status(strategy_id: str = None) -> List[Dict]:
    """获取策略状态"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if strategy_id:
        cursor.execute("SELECT * FROM strategy_status WHERE strategy_id = ?", (strategy_id,))
    else:
        cursor.execute("SELECT * FROM strategy_status")
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_stats(strategy_id: str) -> Dict:
    """获取策略统计数据"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 总交易数
    cursor.execute("SELECT COUNT(*) as total FROM trades WHERE strategy_id = ?", (strategy_id,))
    total = cursor.fetchone()['total']
    
    # 盈利交易数
    cursor.execute("SELECT COUNT(*) as wins FROM trades WHERE strategy_id = ? AND pnl > 0", (strategy_id,))
    wins = cursor.fetchone()['wins']
    
    # 总盈亏
    cursor.execute("SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM trades WHERE strategy_id = ?", (strategy_id,))
    total_pnl = cursor.fetchone()['total_pnl']
    
    conn.close()
    
    return {
        "total_trades": total,
        "win_trades": wins,
        "win_rate": wins / total * 100 if total > 0 else 0,
        "total_pnl": total_pnl
    }

if __name__ == "__main__":
    init_db()
    print("Database setup complete!")

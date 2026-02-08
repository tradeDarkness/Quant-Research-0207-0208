# AI 策略监控仪表盘 (Strategy Dashboard)

> 实时监控多策略交易信号与收益表现

## 快速启动

### 方式一：一键启动

```bash
cd /Users/zhangzc/7/20260123/0208_Dashboard
chmod +x start.sh
./start.sh
```

### 方式二：分别启动

```bash
# 1. 启动后端
cd /Users/zhangzc/7/20260123/0208_Dashboard/backend
/Users/zhangzc/7/20260123/.venv/bin/python -m uvicorn main:app --port 8000

# 2. 启动前端（新终端）
cd /Users/zhangzc/7/20260123/0208_Dashboard/frontend
npm install
npm run dev
```

## 访问地址

| 服务     | 地址                       |
| :------- | :------------------------- |
| 前端界面 | http://localhost:3000      |
| 后端 API | http://localhost:8000      |
| API 文档 | http://localhost:8000/docs |

## 功能说明

### 🎛️ 策略控制面板
- 启动/停止单个策略
- 全部启动/全部停止
- 查看策略运行状态

### 📡 实时信号流
- 实时显示交易信号
- 显示开仓点位、止盈止损、开仓理由
- 支持 WebSocket 推送

### 📈 收益曲线对比
- 各策略累计收益曲线
- 多策略叠加对比

### 📊 统计数据
- 总交易次数
- 胜率
- 总盈亏

## 项目结构

```
0208_Dashboard/
├── backend/
│   ├── main.py              # FastAPI 主应用
│   ├── process_manager.py   # 进程管理器
│   ├── database.py          # SQLite 数据库
│   └── requirements.txt     # Python 依赖
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # 主应用组件
│   │   └── index.css        # 样式
│   ├── package.json
│   └── vite.config.js
├── strategies.json          # 策略配置
├── start.sh                 # 一键启动脚本
└── README.md
```

## API 文档

### 策略管理

| 方法 | 路径                         | 说明         |
| :--- | :--------------------------- | :----------- |
| GET  | `/api/strategies`            | 获取策略列表 |
| POST | `/api/strategies/{id}/start` | 启动策略     |
| POST | `/api/strategies/{id}/stop`  | 停止策略     |
| POST | `/api/strategies/start-all`  | 启动所有     |
| POST | `/api/strategies/stop-all`   | 停止所有     |

### 交易记录

| 方法 | 路径                     | 说明         |
| :--- | :----------------------- | :----------- |
| GET  | `/api/trades`            | 获取交易记录 |
| GET  | `/api/trades/{id}/stats` | 获取统计数据 |

### WebSocket

连接 `ws://localhost:8000/ws` 接收实时信号推送。

---

> **注意**：确保策略脚本已添加 JSON 输出支持，否则信号无法被后端捕获。

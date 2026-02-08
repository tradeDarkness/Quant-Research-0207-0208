import { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts'

function App() {
    const [strategies, setStrategies] = useState([])
    const [trades, setTrades] = useState([])
    const [selectedStrategy, setSelectedStrategy] = useState(null)
    const [strategyTrades, setStrategyTrades] = useState([])
    const [view, setView] = useState('dashboard') // 'dashboard' or 'detail'
    const [btcPrediction, setBtcPrediction] = useState(null)
    const [loadingBtc, setLoadingBtc] = useState(false)
    const wsRef = useRef(null)

    // 获取策略列表
    const fetchStrategies = async () => {
        try {
            const res = await fetch('/api/strategies')
            const data = await res.json()
            setStrategies(data)
        } catch (err) {
            console.error('Failed to fetch strategies:', err)
        }
    }

    // 获取全部交易记录
    const fetchTrades = async () => {
        try {
            const res = await fetch('/api/trades?limit=100')
            const data = await res.json()
            setTrades(data)
        } catch (err) {
            console.error('Failed to fetch trades:', err)
        }
    }

    // 获取单个策略的交易记录
    const fetchStrategyTrades = async (strategyId) => {
        try {
            const res = await fetch(`/api/trades?strategy_id=${strategyId}&limit=200`)
            const data = await res.json()
            setStrategyTrades(data)
        } catch (err) {
            console.error('Failed to fetch strategy trades:', err)
        }
    }

    // 启动策略
    const startStrategy = async (strategyId) => {
        try {
            await fetch(`/api/strategies/${strategyId}/start`, { method: 'POST' })
            fetchStrategies()
        } catch (err) {
            console.error('Failed to start strategy:', err)
        }
    }

    // 停止策略
    const stopStrategy = async (strategyId) => {
        try {
            await fetch(`/api/strategies/${strategyId}/stop`, { method: 'POST' })
            fetchStrategies()
        } catch (err) {
            console.error('Failed to stop strategy:', err)
        }
    }

    // 启动所有策略
    const startAll = async () => {
        try {
            await fetch('/api/strategies/start-all', { method: 'POST' })
            fetchStrategies()
        } catch (err) {
            console.error('Failed to start all:', err)
        }
    }

    // 停止所有策略
    const stopAll = async () => {
        try {
            await fetch('/api/strategies/stop-all', { method: 'POST' })
            fetchStrategies()
        } catch (err) {
            console.error('Failed to stop all:', err)
        }
    }

    // 查看策略详情
    const viewStrategyDetail = (strategy) => {
        setSelectedStrategy(strategy)
        setView('detail')
        fetchStrategyTrades(strategy.id)
    }

    // 返回仪表盘
    const backToDashboard = () => {
        setView('dashboard')
        setSelectedStrategy(null)
        setStrategyTrades([])
    }
    // 获取 BTC 15m 预测
    const fetchBtcPrediction = async () => {
        setLoadingBtc(true)
        try {
            const res = await fetch('/api/predict/btc')
            if (res.ok) {
                const data = await res.json()
                setBtcPrediction(data)
            }
        } catch (err) {
            console.error('Failed to fetch BTC prediction:', err)
        } finally {
            setLoadingBtc(false)
        }
    }

    // WebSocket 连接
    useEffect(() => {
        const ws = new WebSocket(`ws://${window.location.host}/ws`)
        wsRef.current = ws

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data)
            if (message.type === 'signal') {
                setTrades(prev => [message.data, ...prev.slice(0, 99)])
                if (selectedStrategy && message.data.strategy_id === selectedStrategy.id) {
                    setStrategyTrades(prev => [message.data, ...prev.slice(0, 199)])
                }
            } else if (message.type === 'status_change') {
                fetchStrategies()
            }
        }

        return () => ws.close()
    }, [selectedStrategy])

    // 初始化数据
    useEffect(() => {
        fetchStrategies()
        fetchTrades()
        fetchBtcPrediction()
        const interval = setInterval(() => {
            fetchStrategies()
            fetchTrades()
            fetchBtcPrediction()
        }, 30000)
        return () => clearInterval(interval)
    }, [])

    // 统计数据
    const totalTrades = trades.length
    const winTrades = trades.filter(t => t.pnl > 0).length
    const winRate = totalTrades > 0 ? (winTrades / totalTrades * 100).toFixed(1) : 0
    const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0)
    const runningCount = strategies.filter(s => s.running).length

    // 策略详情页
    if (view === 'detail' && selectedStrategy) {
        const strategyWinTrades = strategyTrades.filter(t => t.pnl > 0).length
        const strategyWinRate = strategyTrades.length > 0 ? (strategyWinTrades / strategyTrades.length * 100).toFixed(1) : 0
        const strategyPnl = strategyTrades.reduce((sum, t) => sum + (t.pnl || 0), 0)

        return (
            <div className="dashboard">
                {/* 返回按钮 */}
                <header className="header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <button className="btn btn-secondary" onClick={backToDashboard}>
                            ← 返回
                        </button>
                        <div>
                            <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <span style={{
                                    width: '16px',
                                    height: '16px',
                                    borderRadius: '50%',
                                    backgroundColor: selectedStrategy.color
                                }}></span>
                                {selectedStrategy.name}
                            </h1>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
                                {selectedStrategy.description}
                            </p>
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '12px' }}>
                        {selectedStrategy.running ? (
                            <button className="btn btn-danger" onClick={() => stopStrategy(selectedStrategy.id)}>
                                ⏹️ 停止策略
                            </button>
                        ) : (
                            <button className="btn btn-primary" onClick={() => startStrategy(selectedStrategy.id)}>
                                ▶️ 启动策略
                            </button>
                        )}
                    </div>
                </header>

                {/* 策略统计 */}
                <div className="stats-grid" style={{ marginBottom: '24px' }}>
                    <div className="stat-card">
                        <div className="stat-value">{selectedStrategy.running ? '🟢' : '🔴'}</div>
                        <div className="stat-label">运行状态</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{strategyTrades.length}</div>
                        <div className="stat-label">交易次数</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{strategyWinRate}%</div>
                        <div className="stat-label">胜率</div>
                    </div>
                    <div className="stat-card">
                        <div className={`stat-value ${strategyPnl >= 0 ? 'positive' : 'negative'}`}>
                            {strategyPnl >= 0 ? '+' : ''}{strategyPnl.toFixed(2)}
                        </div>
                        <div className="stat-label">总盈亏</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{selectedStrategy.backtest_return?.toLocaleString()}%</div>
                        <div className="stat-label">回测收益</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{selectedStrategy.win_rate}%</div>
                        <div className="stat-label">回测胜率</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{selectedStrategy.threshold}</div>
                        <div className="stat-label">信号阈值</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">ETHUSDT</div>
                        <div className="stat-label">交易对</div>
                    </div>
                </div>

                {/* 交易记录表格 - OKX/币安风格 */}
                <div className="card">
                    <div className="card-header">
                        <div className="card-title">📋 历史交易记录</div>
                        <div className="card-subtitle">最近 200 条信号</div>
                    </div>

                    <div className="trade-table-container">
                        <table className="trade-table">
                            <thead>
                                <tr>
                                    <th>时间</th>
                                    <th>方向</th>
                                    <th>开仓价</th>
                                    <th>止盈</th>
                                    <th>止损</th>
                                    <th>得分</th>
                                    <th>状态</th>
                                    <th>盈亏</th>
                                </tr>
                            </thead>
                            <tbody>
                                {strategyTrades.length === 0 ? (
                                    <tr>
                                        <td colSpan="8" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
                                            暂无交易记录，启动策略后将在此显示
                                        </td>
                                    </tr>
                                ) : (
                                    strategyTrades.map((trade, index) => (
                                        <tr key={trade.id || index}>
                                            <td>{trade.timestamp}</td>
                                            <td>
                                                <span className={`direction-badge ${trade.direction === 'LONG' ? 'long' : 'short'}`}>
                                                    {trade.direction === 'LONG' ? '做多' : '做空'}
                                                </span>
                                            </td>
                                            <td>{trade.entry_price?.toFixed(2)}</td>
                                            <td style={{ color: 'var(--accent-green)' }}>{trade.take_profit?.toFixed(2)}</td>
                                            <td style={{ color: 'var(--accent-red)' }}>{trade.stop_loss?.toFixed(2)}</td>
                                            <td>{trade.score?.toFixed(6)}</td>
                                            <td>
                                                <span className={`status-badge ${trade.status === 'OPEN' ? 'status-running' : 'status-stopped'}`}>
                                                    {trade.status === 'OPEN' ? '持仓中' : '已平仓'}
                                                </span>
                                            </td>
                                            <td className={trade.pnl > 0 ? 'pnl-positive' : trade.pnl < 0 ? 'pnl-negative' : ''}>
                                                {trade.pnl ? (trade.pnl > 0 ? '+' : '') + trade.pnl.toFixed(2) : '-'}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        )
    }

    // 主仪表盘
    return (
        <div className="dashboard">
            {/* 头部 */}
            <header className="header">
                <div>
                    <h1>📊 AI 策略监控仪表盘</h1>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
                        实时监控 {strategies.length} 个策略的交易信号与收益表现
                    </p>
                </div>
                <div className="header-actions">
                    <button className="btn btn-primary" onClick={startAll}>
                        ▶️ 全部启动
                    </button>
                    <button className="btn btn-danger" onClick={stopAll}>
                        ⏹️ 全部停止
                    </button>
                </div>
            </header>

            {/* 统计卡片 */}
            <div className="stats-grid" style={{ marginBottom: '24px' }}>
                <div className="stat-card">
                    <div className="stat-value">{runningCount}/{strategies.length}</div>
                    <div className="stat-label">运行中策略</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{totalTrades}</div>
                    <div className="stat-label">总交易次数</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{winRate}%</div>
                    <div className="stat-label">胜率</div>
                </div>
                <div className="stat-card">
                    <div className={`stat-value ${totalPnl >= 0 ? 'positive' : 'negative'}`}>
                        {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
                    </div>
                    <div className="stat-label">总盈亏</div>
                </div>
            </div>

            {/* BTC 15m 高精准预测模块 */}
            <div className="card" style={{ marginBottom: '24px', background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)', border: '1px solid #333' }}>
                <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <div className="card-title" style={{ color: '#f7931a', fontSize: '1.2rem' }}>₿ BTC 15m Phase 4 高精准预测</div>
                        <div className="card-subtitle">基于 L2 代理特征与 Meta-Labeling 过滤</div>
                    </div>
                    <button 
                        className="btn btn-secondary btn-sm" 
                        onClick={fetchBtcPrediction} 
                        disabled={loadingBtc}
                        style={{ background: '#333', borderColor: '#444' }}
                    >
                        {loadingBtc ? '计算中...' : '🔄 立即刷新'}
                    </button>
                </div>
                
                {btcPrediction ? (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', padding: '16px' }}>
                        <div className="stat-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid #444' }}>
                            <div className="stat-label">最新预测时间</div>
                            <div className="stat-value" style={{ fontSize: '1.1rem', marginTop: '8px' }}>{btcPrediction.datetime.split(' ')[1]}</div>
                            <div className="stat-label" style={{ fontSize: '0.8rem', opacity: 0.6 }}>{btcPrediction.datetime.split(' ')[0]}</div>
                        </div>
                        <div className="stat-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid #444' }}>
                            <div className="stat-label">当前市场价格</div>
                            <div className="stat-value" style={{ fontSize: '1.5rem', marginTop: '8px', color: '#f7931a' }}>
                                ${btcPrediction.price?.toLocaleString()}
                            </div>
                        </div>
                        <div className="stat-card" style={{ 
                            background: btcPrediction.score > 0.6 ? 'rgba(38, 166, 154, 0.1)' : 'rgba(255,255,255,0.03)', 
                            border: btcPrediction.score > 0.6 ? '1px solid #26a69a' : '1px solid #444',
                            boxShadow: btcPrediction.score > 0.6 ? '0 0 15px rgba(38, 166, 154, 0.2)' : 'none'
                        }}>
                            <div className="stat-label">信号状态</div>
                            <div className="stat-value" style={{ 
                                fontSize: '1.2rem', 
                                marginTop: '8px',
                                color: btcPrediction.score > 0.6 ? '#26a69a' : (btcPrediction.score < 0.35 ? '#ef5350' : '#fff')
                            }}>
                                {btcPrediction.score > 0.6 ? '🚀 强烈看涨' : btcPrediction.signal}
                            </div>
                            {btcPrediction.score > 0.6 && <div style={{ fontSize: '0.7rem', color: '#26a69a', marginTop: '4px' }}>🔥 触发 Phase 4 高置信度阈值</div>}
                        </div>
                        <div className="stat-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid #444' }}>
                            <div className="stat-label">模型分值 (Score)</div>
                            <div className="stat-value" style={{ fontSize: '1.5rem', marginTop: '8px' }}>{btcPrediction.score.toFixed(4)}</div>
                            <div className="stat-label" style={{ fontSize: '0.7rem', marginTop: '4px' }}>
                                Target: Next 15m Alpha
                            </div>
                        </div>
                    </div>
                ) : (
                    <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                        正在拉取最新 BTC 15m 预测数据...
                    </div>
                )}
            </div>

            {/* 策略卡片网格 */}
            <div className="card" style={{ marginBottom: '24px' }}>
                <div className="card-header">
                    <div className="card-title">🎛️ 策略控制面板</div>
                    <div className="card-subtitle">点击策略卡片查看详情</div>
                </div>
                <div className="strategy-grid">
                    {strategies.map(strategy => (
                        <div
                            className="strategy-card"
                            key={strategy.id}
                            onClick={() => viewStrategyDetail(strategy)}
                            style={{ borderLeft: `4px solid ${strategy.color}` }}
                        >
                            <div className="strategy-card-header">
                                <span className="strategy-name">{strategy.name}</span>
                                <span className={`status-badge ${strategy.running ? 'status-running' : 'status-stopped'}`}>
                                    {strategy.running ? '运行中' : '已停止'}
                                </span>
                            </div>
                            <div className="strategy-card-stats">
                                <div className="strategy-stat">
                                    <span className="stat-num">{strategy.backtest_return?.toLocaleString()}%</span>
                                    <span className="stat-lbl">回测收益</span>
                                </div>
                                <div className="strategy-stat">
                                    <span className="stat-num">{strategy.win_rate}%</span>
                                    <span className="stat-lbl">胜率</span>
                                </div>
                                <div className="strategy-stat">
                                    <span className="stat-num">{strategy.threshold}</span>
                                    <span className="stat-lbl">阈值</span>
                                </div>
                            </div>
                            <div className="strategy-card-desc">{strategy.description}</div>
                            <div className="strategy-card-actions" onClick={e => e.stopPropagation()}>
                                {strategy.running ? (
                                    <button className="btn btn-danger btn-sm" onClick={() => stopStrategy(strategy.id)}>
                                        停止
                                    </button>
                                ) : (
                                    <button className="btn btn-primary btn-sm" onClick={() => startStrategy(strategy.id)}>
                                        启动
                                    </button>
                                )}
                                <button className="btn btn-secondary btn-sm" onClick={() => viewStrategyDetail(strategy)}>
                                    详情 →
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 实时信号流 */}
            <div className="card">
                <div className="card-header">
                    <div className="card-title">📡 实时信号流</div>
                    <div className="card-subtitle">全策略最近 20 条交易信号</div>
                </div>
                <div className="signal-feed">
                    {trades.length === 0 ? (
                        <p style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '40px' }}>
                            暂无交易信号，启动策略后将在此显示
                        </p>
                    ) : (
                        trades.slice(0, 20).map((trade, index) => (
                            <div className="signal-item" key={trade.id || index}>
                                <div className={`signal-icon ${trade.direction === 'LONG' ? 'signal-long' : 'signal-short'}`}>
                                    {trade.direction === 'LONG' ? '📈' : '📉'}
                                </div>
                                <div className="signal-content">
                                    <div className="signal-header">
                                        <span className="signal-strategy">{trade.strategy_id}</span>
                                        <span className="signal-time">{trade.timestamp}</span>
                                    </div>
                                    <div style={{ color: trade.direction === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
                                        {trade.direction === 'LONG' ? '做多' : '做空'} @ {trade.entry_price?.toFixed(2)}
                                    </div>
                                    <div className="signal-details">
                                        <span>止盈: <strong>{trade.take_profit?.toFixed(2)}</strong></span>
                                        <span>止损: <strong>{trade.stop_loss?.toFixed(2)}</strong></span>
                                        <span>得分: <strong>{trade.score?.toFixed(6)}</strong></span>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* 页脚 */}
            <footer style={{
                textAlign: 'center',
                marginTop: '40px',
                padding: '20px',
                color: 'var(--text-secondary)',
                fontSize: '12px'
            }}>
                <p>AI 策略监控仪表盘 v1.0.0 | 基于 LightGBM 机器学习模型</p>
                <p style={{ marginTop: '4px' }}>⚠️ 仅供研究使用，不构成投资建议</p>
            </footer>
        </div>
    )
}

export default App

# Polymarket BTC 15m Prediction Project (0208)

## 🎯 核心目标 (Core Objective)
**预测 Polymarket 平台上的 Bitcoin (BTC) 价格走势，专注于 15分钟 级别的短期预测。**

本文件夹包含了一整套从**数据准备**、**模型训练**、**AI 因子挖掘**到**实盘预测**的完整工作流。

---

## 📂 关键文件说明 (Key Files)

### 1. 🚀 自动化运行 (Automation)
- **`start_rdagent.sh`** (⭐ **最重要**): 一键启动脚本。自动修复环境、打补丁并运行 AI 因子挖掘。
  - **使用说明**: 终端运行 `./start_rdagent.sh`。
- `patch_rdagent.py`: 自动修复脚本。用于解决 RD-Agent 与 Gemini 模型不兼容的问题。

### 2. 🔮 实盘与预测 (Live Prediction)
- **`live_polymarket_qlib.py`** (⭐ **核心**): 实盘推理脚本。
  - **功能**: 连接交易所 API 获取最新数据 -> 计算 Qlib 因子 -> 调用训练好的模型 -> 输出预测结果 (Up/Down)。
  - **状态**: 已准备好，等待模型进一步优化。

### 3. 🧠 模型训练 (Model Training)
- `train_qlib_model.py`: 使用 LightGBM 训练预测模型。
  - **输入**: 处理好的 Qlib 二进制数据。
  - **输出**: `qlib_lgbm_btc_15m.pkl` (模型文件)。
- `prepare_qlib_btc_15m.py`: 数据清洗与格式转换工具。将 CSV 转换为 Qlib 可用的 `.bin` 格式。

### 4. 🕵️ AI 因子挖掘 (AI Researcher)
- `.venv_rdagent/`: 专用的虚拟环境，包含 RD-Agent 框架。
- `rdagent.log`: AI 运行日志。
- `conda_mock.log`: 回测调用日志。

---

## 🔄 工作流程 (Workflow)

1.  **数据准备 (Data)**: 使用 `prepare_qlib_btc_15m.py` 将 BTC 历史数据格式化。
2.  **基准模型 (Baseline)**: 运行 `train_qlib_model.py` 训练一个基础模型 (LightGBM)，作为评分基准。
3.  **AI 挖掘 (Mining)**:
    -   运行 `./start_rdagent.sh`。
    -   RD-Agent (AI 研究员) 会自动提出新因子假设 -> 写代码 -> 回测验证 -> 迭代优化。
    -   **目标**: 找到比基准模型更准的因子。
4.  **实盘部署 (Live)**:
    -   将挖掘到的有效因子更新到 `live_polymarket_qlib.py`。
    -   启动脚本进行实时预测。

## ⚠️ 当前状态 (Current Status)

- **实盘脚本**: ✅ 就绪 (`live_polymarket_qlib.py`)。
- **基准模型**: ✅ 已训练 (AUC ~0.53)。
- **AI 挖掘**: ⚠️ **运行中但有阻碍**。
    -   RD-Agent 正在尝试挖掘新因子。
    -   由于 Google Gemini 模型的兼容性问题 (JSON Mode & Rate Limit)，目前运行较慢且有报错重试。
    -   **建议**: 如果 AI 挖掘持续卡顿，为了达成“预测”这一核心目标，可以直接使用当前的基准模型与实盘脚本先行跑通流程，此时无需等待 RD-Agent 结果。

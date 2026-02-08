"""
进程管理器
管理多个策略脚本的启动、停止和输出捕获
"""

import subprocess
import threading
import json
import re
from pathlib import Path
from typing import Dict, Callable, Optional
from datetime import datetime

class StrategyProcess:
    """单个策略进程管理"""
    
    def __init__(self, strategy_id: str, script_path: str, on_signal: Callable = None):
        self.strategy_id = strategy_id
        self.script_path = script_path
        self.on_signal = on_signal
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
    def start(self, python_path: str = "/Users/zhangzc/7/20260123/.venv/bin/python"):
        """启动策略进程"""
        if self.running:
            return False
            
        try:
            # -u 禁用 Python 输出缓冲，确保信号立即可见
            self.process = subprocess.Popen(
                [python_path, '-u', self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.running = True
            
            # 启动输出读取线程
            self.thread = threading.Thread(target=self._read_output, daemon=True)
            self.thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to start {self.strategy_id}: {e}")
            return False
    
    def stop(self):
        """停止策略进程"""
        if self.process and self.running:
            self.running = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            return True
        return False
    
    def _read_output(self):
        """读取进程输出并解析信号"""
        if not self.process:
            return
            
        signal_pattern = re.compile(r'SIGNAL_JSON:(.+)')
        
        for line in iter(self.process.stdout.readline, ''):
            if not self.running:
                break
                
            line = line.strip()
            if not line:
                continue
                
            # 打印原始输出
            print(f"[{self.strategy_id}] {line}")
            
            # 检查是否是 JSON 信号
            match = signal_pattern.search(line)
            if match:
                try:
                    signal_data = json.loads(match.group(1))
                    signal_data['strategy_id'] = self.strategy_id
                    if self.on_signal:
                        self.on_signal(signal_data)
                except json.JSONDecodeError:
                    pass
            
            # 也解析中文格式的信号
            if "📈 方向：" in line:
                direction = "LONG" if "做多" in line else "SHORT" if "做空" in line else None
                if direction and self.on_signal:
                    # 简化解析，实际信号通过 JSON 格式传递
                    pass


class ProcessManager:
    """多策略进程管理器"""
    
    def __init__(self, on_signal: Callable = None):
        self.strategies: Dict[str, StrategyProcess] = {}
        self.on_signal = on_signal
        self.python_path = "/Users/zhangzc/7/20260123/.venv/bin/python"
        
    def load_strategies(self, config_path: str = None):
        """从配置文件加载策略"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "strategies.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for strategy in config['strategies']:
            self.add_strategy(
                strategy['id'],
                strategy['path']
            )
        
        return list(self.strategies.keys())
    
    def add_strategy(self, strategy_id: str, script_path: str):
        """添加策略"""
        self.strategies[strategy_id] = StrategyProcess(
            strategy_id,
            script_path,
            self.on_signal
        )
    
    def start_strategy(self, strategy_id: str) -> bool:
        """启动单个策略"""
        if strategy_id not in self.strategies:
            return False
        return self.strategies[strategy_id].start(self.python_path)
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """停止单个策略"""
        if strategy_id not in self.strategies:
            return False
        return self.strategies[strategy_id].stop()
    
    def start_all(self):
        """启动所有策略"""
        results = {}
        for strategy_id in self.strategies:
            results[strategy_id] = self.start_strategy(strategy_id)
        return results
    
    def stop_all(self):
        """停止所有策略"""
        results = {}
        for strategy_id in self.strategies:
            results[strategy_id] = self.stop_strategy(strategy_id)
        return results
    
    def get_status(self) -> Dict:
        """获取所有策略状态"""
        status = {}
        for strategy_id, process in self.strategies.items():
            status[strategy_id] = {
                "running": process.running,
                "pid": process.process.pid if process.process else None
            }
        return status


if __name__ == "__main__":
    # 测试
    def on_signal(signal):
        print(f"Signal received: {signal}")
    
    manager = ProcessManager(on_signal)
    manager.load_strategies()
    print(f"Loaded strategies: {list(manager.strategies.keys())}")

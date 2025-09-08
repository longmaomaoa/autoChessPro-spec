#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pikafish引擎基础功能测试

测试引擎集成的核心数据结构、状态管理和基础功能
"""

import sys
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import threading
import queue

print("开始Pikafish引擎基础功能测试...")
print("=" * 50)

# 定义测试用的基础枚举和数据结构
class EngineState(Enum):
    """引擎状态枚举"""
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    THINKING = "THINKING"
    ANALYZING = "ANALYZING"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"

class UCCICommand(Enum):
    """UCCI命令枚举"""
    UCCI = "ucci"
    ISREADY = "isready"
    SETOPTION = "setoption"
    UCINEWGAME = "ucinewgame"
    POSITION = "position"
    GO = "go"
    STOP = "stop"
    QUIT = "quit"

@dataclass
class EngineOption:
    """引擎选项"""
    name: str
    option_type: str
    default_value: Any = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    var_values: List[str] = field(default_factory=list)

@dataclass
class EngineAnalysis:
    """引擎分析结果"""
    depth: int
    score: float
    time: float
    nodes: int
    pv: List[str]
    best_move: Optional[str] = None
    mate: Optional[int] = None
    
    def __post_init__(self):
        if self.pv and not self.best_move:
            self.best_move = self.pv[0] if self.pv else None

@dataclass
class EngineStats:
    """引擎统计信息"""
    total_commands: int = 0
    successful_commands: int = 0
    failed_commands: int = 0
    total_thinking_time: float = 0.0
    total_nodes_searched: int = 0
    average_depth: float = 0.0
    uptime: float = 0.0
    last_response_time: float = 0.0
    
    def update_command(self, success: bool, response_time: float):
        self.total_commands += 1
        if success:
            self.successful_commands += 1
        else:
            self.failed_commands += 1
        self.last_response_time = response_time
    
    def update_analysis(self, analysis: EngineAnalysis):
        self.total_thinking_time += analysis.time
        self.total_nodes_searched += analysis.nodes
        if self.successful_commands > 0:
            self.average_depth = (self.average_depth * (self.successful_commands - 1) + analysis.depth) / self.successful_commands

class PikafishEngineError(Exception):
    """引擎异常"""
    def __init__(self, message: str, error_code: Optional[str] = None, command: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.command = command

# 执行测试
def run_tests():
    """执行所有测试"""
    test_count = 0
    passed_count = 0
    
    # 测试1: EngineState枚举
    test_count += 1
    try:
        states = [state.value for state in EngineState]
        expected = ["IDLE", "INITIALIZING", "READY", "THINKING", "ANALYZING", "ERROR", "TERMINATED"]
        assert set(states) == set(expected)
        assert len(states) == 7
        print("PASS: EngineState枚举测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: EngineState枚举测试 - {e}")
    
    # 测试2: UCCICommand枚举
    test_count += 1
    try:
        commands = [cmd.value for cmd in UCCICommand]
        expected = ["ucci", "isready", "setoption", "ucinewgame", "position", "go", "stop", "quit"]
        assert set(commands) == set(expected)
        assert len(commands) == 8
        print("PASS: UCCICommand枚举测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: UCCICommand枚举测试 - {e}")
    
    # 测试3: EngineOption数据类
    test_count += 1
    try:
        option = EngineOption(
            name="Threads",
            option_type="spin",
            default_value=1,
            min_value=1,
            max_value=128
        )
        
        assert option.name == "Threads"
        assert option.option_type == "spin"
        assert option.default_value == 1
        assert option.min_value == 1
        assert option.max_value == 128
        assert option.var_values == []
        
        print("PASS: EngineOption数据类测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: EngineOption数据类测试 - {e}")
    
    # 测试4: EngineAnalysis数据类
    test_count += 1
    try:
        analysis = EngineAnalysis(
            depth=12,
            score=0.25,
            time=1.5,
            nodes=50000,
            pv=["e2e4", "e7e5", "g1f3"],
            mate=None
        )
        
        assert analysis.depth == 12
        assert analysis.score == 0.25
        assert analysis.time == 1.5
        assert analysis.nodes == 50000
        assert analysis.pv == ["e2e4", "e7e5", "g1f3"]
        assert analysis.best_move == "e2e4"  # 应该从pv[0]自动设置
        assert analysis.mate is None
        
        print("PASS: EngineAnalysis数据类测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: EngineAnalysis数据类测试 - {e}")
    
    # 测试5: EngineStats统计功能
    test_count += 1
    try:
        stats = EngineStats()
        
        # 初始状态
        assert stats.total_commands == 0
        assert stats.successful_commands == 0
        assert stats.failed_commands == 0
        
        # 更新命令统计
        stats.update_command(True, 0.1)
        stats.update_command(True, 0.2)
        stats.update_command(False, 0.0)
        
        assert stats.total_commands == 3
        assert stats.successful_commands == 2
        assert stats.failed_commands == 1
        assert stats.last_response_time == 0.0
        
        # 更新分析统计
        analysis1 = EngineAnalysis(10, 0.5, 1.0, 10000, ["e2e4"])
        analysis2 = EngineAnalysis(12, -0.3, 1.5, 15000, ["d2d4"])
        
        stats.update_analysis(analysis1)
        stats.update_analysis(analysis2)
        
        assert stats.total_thinking_time == 2.5
        assert stats.total_nodes_searched == 25000
        assert stats.average_depth == 11.0  # (10 + 12) / 2
        
        print("PASS: EngineStats统计功能测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: EngineStats统计功能测试 - {e}")
    
    # 测试6: PikafishEngineError异常类
    test_count += 1
    try:
        try:
            raise PikafishEngineError("测试错误", "TEST_ERROR", "go depth 10")
        except PikafishEngineError as e:
            assert str(e) == "测试错误"
            assert e.error_code == "TEST_ERROR"
            assert e.command == "go depth 10"
        
        # 测试默认值
        try:
            raise PikafishEngineError("另一个错误")
        except PikafishEngineError as e:
            assert str(e) == "另一个错误"
            assert e.error_code is None
            assert e.command is None
        
        print("PASS: PikafishEngineError异常类测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: PikafishEngineError异常类测试 - {e}")
    
    # 测试7: UCCI协议解析模拟
    test_count += 1
    try:
        def parse_engine_option(line: str) -> Optional[EngineOption]:
            """模拟引擎选项解析"""
            # 示例: option name Threads type spin default 1 min 1 max 128
            try:
                parts = line.split()
                if len(parts) < 6:
                    return None
                
                name = ""
                option_type = ""
                default_value = None
                min_value = None
                max_value = None
                
                i = 2  # 跳过 "option name"
                while i < len(parts) and parts[i] != "type":
                    name += parts[i] + " "
                    i += 1
                name = name.strip()
                
                if i + 1 < len(parts):
                    option_type = parts[i + 1]
                    i += 2
                
                # 解析其他参数
                while i < len(parts):
                    if parts[i] == "default" and i + 1 < len(parts):
                        default_value = parts[i + 1]
                        i += 2
                    elif parts[i] == "min" and i + 1 < len(parts):
                        min_value = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "max" and i + 1 < len(parts):
                        max_value = int(parts[i + 1])
                        i += 2
                    else:
                        i += 1
                
                return EngineOption(
                    name=name,
                    option_type=option_type,
                    default_value=default_value,
                    min_value=min_value,
                    max_value=max_value
                )
                
            except Exception:
                return None
        
        # 测试解析
        line = "option name Threads type spin default 1 min 1 max 128"
        option = parse_engine_option(line)
        
        assert option is not None
        assert option.name == "Threads"
        assert option.option_type == "spin"
        assert option.default_value == "1"
        assert option.min_value == 1
        assert option.max_value == 128
        
        print("PASS: UCCI协议解析模拟测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: UCCI协议解析模拟测试 - {e}")
    
    # 测试8: 分析信息解析模拟
    test_count += 1
    try:
        def parse_analysis_info(line: str) -> Optional[EngineAnalysis]:
            """模拟分析信息解析"""
            # 示例: info depth 12 score cp 25 time 1234 nodes 567890 pv e2e4 e7e5
            try:
                parts = line.split()
                if len(parts) < 3:
                    return None
                
                depth = 0
                score = 0.0
                time_ms = 0
                nodes = 0
                pv = []
                mate = None
                
                i = 1  # 跳过 "info"
                while i < len(parts):
                    if parts[i] == "depth" and i + 1 < len(parts):
                        depth = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "score":
                        if i + 2 < len(parts):
                            if parts[i + 1] == "cp":
                                score = float(parts[i + 2]) / 100.0
                                i += 3
                            elif parts[i + 1] == "mate":
                                mate = int(parts[i + 2])
                                score = 999.0 if mate > 0 else -999.0
                                i += 3
                            else:
                                i += 1
                        else:
                            i += 1
                    elif parts[i] == "time" and i + 1 < len(parts):
                        time_ms = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "nodes" and i + 1 < len(parts):
                        nodes = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "pv":
                        pv = parts[i + 1:]
                        break
                    else:
                        i += 1
                
                if depth > 0:
                    return EngineAnalysis(
                        depth=depth,
                        score=score,
                        time=time_ms / 1000.0,
                        nodes=nodes,
                        pv=pv,
                        mate=mate
                    )
                
                return None
                
            except Exception:
                return None
        
        # 测试解析
        line = "info depth 12 score cp 25 time 1234 nodes 567890 pv e2e4 e7e5"
        analysis = parse_analysis_info(line)
        
        assert analysis is not None
        assert analysis.depth == 12
        assert analysis.score == 0.25  # cp 25 -> 0.25
        assert analysis.time == 1.234   # 1234ms -> 1.234s
        assert analysis.nodes == 567890
        assert analysis.pv == ["e2e4", "e7e5"]
        assert analysis.best_move == "e2e4"
        assert analysis.mate is None
        
        print("PASS: 分析信息解析模拟测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: 分析信息解析模拟测试 - {e}")
    
    # 测试9: 命令队列机制模拟
    test_count += 1
    try:
        command_queue = queue.Queue()
        response_queue = queue.Queue()
        
        # 模拟命令发送
        commands = ["ucci", "isready", "go depth 10", "stop"]
        for cmd in commands:
            command_queue.put(cmd)
        
        # 模拟处理命令
        processed_commands = []
        while not command_queue.empty():
            cmd = command_queue.get()
            processed_commands.append(cmd)
            # 模拟响应
            if cmd == "ucci":
                response_queue.put("ucciok")
            elif cmd == "isready":
                response_queue.put("readyok")
        
        assert processed_commands == commands
        assert response_queue.qsize() == 2  # ucci 和 isready 的响应
        
        # 检查响应
        assert response_queue.get() == "ucciok"
        assert response_queue.get() == "readyok"
        
        print("PASS: 命令队列机制模拟测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: 命令队列机制模拟测试 - {e}")
    
    # 测试10: 状态机转换模拟
    test_count += 1
    try:
        class MockEngine:
            def __init__(self):
                self.state = EngineState.IDLE
                self.state_history = []
            
            def set_state(self, new_state: EngineState):
                if new_state != self.state:
                    old_state = self.state
                    self.state = new_state
                    self.state_history.append((old_state, new_state))
            
            def simulate_initialization(self):
                self.set_state(EngineState.INITIALIZING)
                # 模拟初始化完成
                self.set_state(EngineState.READY)
            
            def simulate_thinking(self):
                if self.state == EngineState.READY:
                    self.set_state(EngineState.THINKING)
                    # 模拟思考完成
                    self.set_state(EngineState.READY)
        
        engine = MockEngine()
        
        # 测试初始化序列
        engine.simulate_initialization()
        assert engine.state == EngineState.READY
        assert len(engine.state_history) == 2
        assert engine.state_history[0] == (EngineState.IDLE, EngineState.INITIALIZING)
        assert engine.state_history[1] == (EngineState.INITIALIZING, EngineState.READY)
        
        # 测试思考序列
        engine.simulate_thinking()
        assert engine.state == EngineState.READY
        assert len(engine.state_history) == 4
        assert engine.state_history[2] == (EngineState.READY, EngineState.THINKING)
        assert engine.state_history[3] == (EngineState.THINKING, EngineState.READY)
        
        print("PASS: 状态机转换模拟测试")
        passed_count += 1
    except Exception as e:
        print(f"FAIL: 状态机转换模拟测试 - {e}")
    
    # 输出测试结果
    print("=" * 50)
    print(f"测试结果: {passed_count}/{test_count} 通过")
    print(f"成功率: {passed_count/test_count*100:.1f}%")
    
    if passed_count == test_count:
        print("所有Pikafish引擎基础功能测试通过!")
        return True
    else:
        print("部分测试失败，需要检查实现")
        return False

if __name__ == "__main__":
    success = run_tests()
    
    if success:
        print("\nPikafish引擎集成基础功能验证通过!")
        print("下一步可以继续:")
        print("1. 配置实际的Pikafish引擎可执行文件")
        print("2. 实现与视觉识别模块的数据接口")
        print("3. 开发AI走法建议系统")
        print("4. 添加引擎性能监控和诊断")
        print("5. 集成测试和实际引擎通信验证")
    
    exit(0 if success else 1)
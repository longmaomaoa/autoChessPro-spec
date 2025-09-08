#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pikafish象棋引擎集成模块

本模块负责集成Pikafish象棋引擎，实现UCCI协议通信和引擎管理。
主要功能：
1. Pikafish引擎进程管理和生命周期控制
2. UCCI协议处理器和命令解析
3. 引擎初始化和配置功能
4. 走法分析和评估接口
5. 引擎状态监控和诊断

技术特点：
- 基于subprocess的引擎进程管理
- 完整的UCCI协议支持
- 异步命令处理和响应解析
- 引擎参数动态配置
- 完善的错误处理和恢复机制
"""

import subprocess
import threading
import time
import queue
import os
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from pathlib import Path

from chess_ai.config.config_manager import ConfigManager
from chess_ai.data.pieces import Position, Move, BoardState, Piece
from chess_ai.utils.logger import Logger


class EngineState(Enum):
    """引擎状态枚举"""
    IDLE = "IDLE"                    # 空闲状态
    INITIALIZING = "INITIALIZING"    # 初始化中
    READY = "READY"                  # 准备就绪
    THINKING = "THINKING"            # 思考中
    ANALYZING = "ANALYZING"          # 分析中
    ERROR = "ERROR"                  # 错误状态
    TERMINATED = "TERMINATED"        # 已终止


class UCCICommand(Enum):
    """UCCI命令枚举"""
    UCCI = "ucci"                    # 引擎识别
    ISREADY = "isready"              # 检查就绪
    SETOPTION = "setoption"          # 设置选项
    UCINEWGAME = "ucinewgame"        # 新游戏
    POSITION = "position"            # 设置局面
    GO = "go"                        # 开始思考
    STOP = "stop"                    # 停止思考
    QUIT = "quit"                    # 退出引擎


@dataclass
class EngineOption:
    """引擎选项"""
    name: str
    option_type: str  # "check", "spin", "combo", "button", "string"
    default_value: Any = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    var_values: List[str] = field(default_factory=list)


@dataclass
class EngineAnalysis:
    """引擎分析结果"""
    depth: int
    score: float  # 以分为单位的评估值
    time: float   # 思考时间（秒）
    nodes: int    # 搜索节点数
    pv: List[str]  # 主要变着
    best_move: Optional[str] = None
    mate: Optional[int] = None  # 将杀步数，None表示非将杀
    
    def __post_init__(self):
        """初始化后处理"""
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
        """更新命令统计"""
        self.total_commands += 1
        if success:
            self.successful_commands += 1
        else:
            self.failed_commands += 1
        self.last_response_time = response_time
    
    def update_analysis(self, analysis: EngineAnalysis):
        """更新分析统计"""
        self.total_thinking_time += analysis.time
        self.total_nodes_searched += analysis.nodes
        if self.successful_commands > 0:
            self.average_depth = (self.average_depth * (self.successful_commands - 1) + analysis.depth) / self.successful_commands


class PikafishEngineError(Exception):
    """Pikafish引擎异常"""
    def __init__(self, message: str, error_code: Optional[str] = None, command: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.command = command


class PikafishEngine:
    """Pikafish象棋引擎"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化Pikafish引擎
        
        Args:
            config: 配置管理器
        """
        self.config = config or ConfigManager()
        self.logger = Logger(__name__)
        
        # 引擎状态
        self.state = EngineState.IDLE
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        
        # 通信管理
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.communication_thread: Optional[threading.Thread] = None
        
        # 引擎配置和信息
        self.engine_path: Optional[str] = None
        self.engine_options: Dict[str, EngineOption] = {}
        self.engine_info: Dict[str, str] = {}
        
        # 统计信息
        self.stats = EngineStats()
        self.start_time = time.time()
        
        # 回调函数
        self.analysis_callbacks: List[Callable[[EngineAnalysis], None]] = []
        self.error_callbacks: List[Callable[[PikafishEngineError], None]] = []
        self.state_change_callbacks: List[Callable[[EngineState, EngineState], None]] = []
        
        # 线程同步
        self.command_lock = threading.RLock()
        self.state_lock = threading.RLock()
        
        # 加载配置
        self._load_engine_config()
        
        self.logger.info("Pikafish引擎初始化完成")
    
    def _load_engine_config(self):
        """加载引擎配置"""
        engine_config = self.config.ai_engine.pikafish
        
        self.engine_path = engine_config.executable_path
        self.default_depth = engine_config.default_depth
        self.default_time = engine_config.default_time_limit
        self.max_threads = engine_config.max_threads
        self.hash_size = engine_config.hash_size_mb
        
        # 验证引擎路径
        if not self.engine_path or not os.path.exists(self.engine_path):
            self.logger.warning(f"引擎路径不存在: {self.engine_path}")
    
    def initialize(self) -> bool:
        """
        初始化引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self._set_state(EngineState.INITIALIZING)
            
            # 检查引擎路径
            if not self.engine_path or not os.path.exists(self.engine_path):
                raise PikafishEngineError(f"引擎可执行文件不存在: {self.engine_path}", "ENGINE_NOT_FOUND")
            
            # 启动引擎进程
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=0
            )
            
            # 启动通信线程
            self.is_running = True
            self.communication_thread = threading.Thread(target=self._communication_loop, daemon=True)
            self.communication_thread.start()
            
            # 发送UCCI初始化命令
            self._send_command("ucci")
            
            # 等待引擎响应
            timeout = 10.0  # 10秒超时
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=0.1)
                    if response.startswith("ucciok"):
                        self.logger.info("引擎UCCI初始化成功")
                        break
                except queue.Empty:
                    continue
            else:
                raise PikafishEngineError("引擎初始化超时", "INIT_TIMEOUT")
            
            # 设置引擎参数
            self._configure_engine()
            
            # 检查引擎就绪状态
            self._send_command("isready")
            
            # 等待就绪响应
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=0.1)
                    if response.strip() == "readyok":
                        self._set_state(EngineState.READY)
                        self.logger.info("引擎就绪完成")
                        return True
                except queue.Empty:
                    continue
            
            raise PikafishEngineError("引擎就绪检查超时", "READY_TIMEOUT")
            
        except Exception as e:
            self.logger.error(f"引擎初始化失败: {e}")
            self._set_state(EngineState.ERROR)
            self._handle_error(PikafishEngineError(f"引擎初始化失败: {e}", "INIT_FAILED"))
            return False
    
    def _configure_engine(self):
        """配置引擎参数"""
        try:
            # 设置线程数
            if self.max_threads > 0:
                self._send_command(f"setoption name Threads value {self.max_threads}")
            
            # 设置哈希表大小
            if self.hash_size > 0:
                self._send_command(f"setoption name Hash value {self.hash_size}")
            
            # 其他引擎特定配置
            additional_options = self.config.ai_engine.pikafish.additional_options
            for option_name, option_value in additional_options.items():
                self._send_command(f"setoption name {option_name} value {option_value}")
            
        except Exception as e:
            self.logger.error(f"引擎配置失败: {e}")
    
    def _communication_loop(self):
        """引擎通信主循环"""
        while self.is_running and self.process:
            try:
                # 处理引擎输出
                if self.process.stdout and self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            self._handle_engine_output(line)
                
                # 处理命令队列
                try:
                    command = self.command_queue.get(timeout=0.1)
                    self._send_to_engine(command)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"通信循环异常: {e}")
                self._handle_error(PikafishEngineError(f"通信异常: {e}", "COMMUNICATION_ERROR"))
                break
        
        self.logger.info("引擎通信循环结束")
    
    def _handle_engine_output(self, line: str):
        """
        处理引擎输出
        
        Args:
            line: 引擎输出行
        """
        try:
            line = line.strip()
            if not line:
                return
            
            # 解析不同类型的引擎输出
            if line.startswith("id name"):
                self.engine_info["name"] = line.split("id name", 1)[1].strip()
            elif line.startswith("id author"):
                self.engine_info["author"] = line.split("id author", 1)[1].strip()
            elif line.startswith("option name"):
                self._parse_engine_option(line)
            elif line.startswith("info"):
                self._parse_analysis_info(line)
            elif line.startswith("bestmove"):
                self._parse_best_move(line)
            elif line in ["ucciok", "readyok"]:
                self.response_queue.put(line)
            else:
                # 其他响应也放入队列
                self.response_queue.put(line)
                
        except Exception as e:
            self.logger.error(f"处理引擎输出异常: {e}")
    
    def _parse_engine_option(self, line: str):
        """解析引擎选项"""
        # 示例: option name Threads type spin default 1 min 1 max 128
        try:
            parts = line.split()
            if len(parts) < 6:
                return
            
            name = ""
            option_type = ""
            default_value = None
            min_value = None
            max_value = None
            var_values = []
            
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
                elif parts[i] == "var" and i + 1 < len(parts):
                    var_values.append(parts[i + 1])
                    i += 2
                else:
                    i += 1
            
            option = EngineOption(
                name=name,
                option_type=option_type,
                default_value=default_value,
                min_value=min_value,
                max_value=max_value,
                var_values=var_values
            )
            
            self.engine_options[name] = option
            
        except Exception as e:
            self.logger.error(f"解析引擎选项异常: {e}")
    
    def _parse_analysis_info(self, line: str):
        """解析分析信息"""
        # 示例: info depth 12 score cp 25 time 1234 nodes 567890 pv e2e4 e7e5
        try:
            parts = line.split()
            if len(parts) < 3:
                return
            
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
                            score = float(parts[i + 2]) / 100.0  # 转换为子力分数
                            i += 3
                        elif parts[i + 1] == "mate":
                            mate = int(parts[i + 2])
                            score = 999.0 if mate > 0 else -999.0  # 将杀分数
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
            
            if depth > 0:  # 只有当深度大于0时才创建分析结果
                analysis = EngineAnalysis(
                    depth=depth,
                    score=score,
                    time=time_ms / 1000.0,  # 转换为秒
                    nodes=nodes,
                    pv=pv,
                    mate=mate
                )
                
                # 更新统计
                self.stats.update_analysis(analysis)
                
                # 调用回调
                for callback in self.analysis_callbacks:
                    try:
                        callback(analysis)
                    except Exception as e:
                        self.logger.error(f"分析回调异常: {e}")
                        
        except Exception as e:
            self.logger.error(f"解析分析信息异常: {e}")
    
    def _parse_best_move(self, line: str):
        """解析最佳走法"""
        # 示例: bestmove e2e4 ponder e7e5
        try:
            parts = line.split()
            if len(parts) >= 2:
                best_move = parts[1]
                ponder_move = parts[3] if len(parts) >= 4 and parts[2] == "ponder" else None
                
                self.logger.info(f"引擎最佳走法: {best_move}" + (f", 预测对手走法: {ponder_move}" if ponder_move else ""))
                
                # 可以在这里触发最佳走法回调
                
        except Exception as e:
            self.logger.error(f"解析最佳走法异常: {e}")
    
    def _send_command(self, command: str):
        """
        发送命令到引擎
        
        Args:
            command: 要发送的命令
        """
        with self.command_lock:
            if self.is_running:
                self.command_queue.put(command)
                self.logger.debug(f"发送命令: {command}")
    
    def _send_to_engine(self, command: str):
        """
        直接发送命令到引擎进程
        
        Args:
            command: 命令字符串
        """
        try:
            if self.process and self.process.stdin:
                start_time = time.time()
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
                response_time = time.time() - start_time
                self.stats.update_command(True, response_time)
                self.logger.debug(f"已发送到引擎: {command}")
            else:
                raise PikafishEngineError("引擎进程未运行", "PROCESS_NOT_RUNNING", command)
                
        except Exception as e:
            self.stats.update_command(False, 0.0)
            self.logger.error(f"发送命令失败: {command}, 错误: {e}")
            raise PikafishEngineError(f"发送命令失败: {e}", "SEND_COMMAND_FAILED", command)
    
    def set_position(self, board_state: BoardState, moves: Optional[List[str]] = None):
        """
        设置棋局位置
        
        Args:
            board_state: 棋局状态
            moves: 走法列表（可选）
        """
        try:
            # 构造FEN记录
            fen = board_state.to_fen()
            
            # 构造position命令
            if moves:
                moves_str = " ".join(moves)
                command = f"position fen {fen} moves {moves_str}"
            else:
                command = f"position fen {fen}"
            
            self._send_command(command)
            self.logger.info(f"设置棋局位置: {command[:100]}...")  # 截断长命令用于日志
            
        except Exception as e:
            self.logger.error(f"设置棋局位置失败: {e}")
            raise PikafishEngineError(f"设置位置失败: {e}", "SET_POSITION_FAILED")
    
    def start_analysis(self, depth: Optional[int] = None, time_limit: Optional[float] = None):
        """
        开始分析
        
        Args:
            depth: 搜索深度
            time_limit: 时间限制（秒）
        """
        try:
            self._set_state(EngineState.THINKING)
            
            # 构造go命令
            go_params = []
            
            if depth is not None:
                go_params.append(f"depth {depth}")
            elif time_limit is not None:
                time_ms = int(time_limit * 1000)
                go_params.append(f"movetime {time_ms}")
            else:
                # 使用默认参数
                if self.default_depth > 0:
                    go_params.append(f"depth {self.default_depth}")
                elif self.default_time > 0:
                    time_ms = int(self.default_time * 1000)
                    go_params.append(f"movetime {time_ms}")
            
            command = "go " + " ".join(go_params) if go_params else "go"
            self._send_command(command)
            
            self.logger.info(f"开始分析: {command}")
            
        except Exception as e:
            self.logger.error(f"开始分析失败: {e}")
            self._set_state(EngineState.ERROR)
            raise PikafishEngineError(f"开始分析失败: {e}", "START_ANALYSIS_FAILED")
    
    def stop_analysis(self):
        """停止分析"""
        try:
            if self.state == EngineState.THINKING:
                self._send_command("stop")
                self._set_state(EngineState.READY)
                self.logger.info("停止分析")
            
        except Exception as e:
            self.logger.error(f"停止分析失败: {e}")
            raise PikafishEngineError(f"停止分析失败: {e}", "STOP_ANALYSIS_FAILED")
    
    def get_best_move(self, board_state: BoardState, depth: Optional[int] = None, 
                      time_limit: Optional[float] = None) -> Optional[str]:
        """
        获取最佳走法
        
        Args:
            board_state: 棋局状态
            depth: 搜索深度
            time_limit: 时间限制
            
        Returns:
            str: 最佳走法字符串，如果失败返回None
        """
        try:
            # 设置棋局
            self.set_position(board_state)
            
            # 开始分析
            self.start_analysis(depth, time_limit)
            
            # 等待结果
            timeout = (time_limit or self.default_time) + 5.0  # 额外5秒容错
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=0.1)
                    if response.startswith("bestmove"):
                        parts = response.split()
                        if len(parts) >= 2:
                            best_move = parts[1]
                            if best_move != "(none)":
                                self._set_state(EngineState.READY)
                                return best_move
                except queue.Empty:
                    continue
            
            # 超时处理
            self.stop_analysis()
            self.logger.warning("获取最佳走法超时")
            return None
            
        except Exception as e:
            self.logger.error(f"获取最佳走法失败: {e}")
            return None
    
    def _set_state(self, new_state: EngineState):
        """
        设置引擎状态
        
        Args:
            new_state: 新状态
        """
        with self.state_lock:
            if new_state != self.state:
                old_state = self.state
                self.state = new_state
                
                # 调用状态变化回调
                for callback in self.state_change_callbacks:
                    try:
                        callback(old_state, new_state)
                    except Exception as e:
                        self.logger.error(f"状态变化回调异常: {e}")
    
    def _handle_error(self, error: PikafishEngineError):
        """
        处理错误
        
        Args:
            error: 引擎错误
        """
        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"错误回调异常: {e}")
    
    def add_analysis_callback(self, callback: Callable[[EngineAnalysis], None]):
        """添加分析回调"""
        self.analysis_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[PikafishEngineError], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def add_state_change_callback(self, callback: Callable[[EngineState, EngineState], None]):
        """添加状态变化回调"""
        self.state_change_callbacks.append(callback)
    
    def get_state(self) -> EngineState:
        """获取引擎状态"""
        return self.state
    
    def get_stats(self) -> EngineStats:
        """获取统计信息"""
        self.stats.uptime = time.time() - self.start_time
        return self.stats
    
    def get_engine_info(self) -> Dict[str, str]:
        """获取引擎信息"""
        return self.engine_info.copy()
    
    def get_engine_options(self) -> Dict[str, EngineOption]:
        """获取引擎选项"""
        return self.engine_options.copy()
    
    def set_engine_option(self, name: str, value: Any) -> bool:
        """
        设置引擎选项
        
        Args:
            name: 选项名称
            value: 选项值
            
        Returns:
            bool: 设置是否成功
        """
        try:
            if name in self.engine_options:
                command = f"setoption name {name} value {value}"
                self._send_command(command)
                self.logger.info(f"设置引擎选项: {name} = {value}")
                return True
            else:
                self.logger.warning(f"未知的引擎选项: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"设置引擎选项失败: {name} = {value}, 错误: {e}")
            return False
    
    def new_game(self):
        """开始新游戏"""
        try:
            self._send_command("ucinewgame")
            self._set_state(EngineState.READY)
            self.logger.info("开始新游戏")
            
        except Exception as e:
            self.logger.error(f"开始新游戏失败: {e}")
            raise PikafishEngineError(f"开始新游戏失败: {e}", "NEW_GAME_FAILED")
    
    def shutdown(self):
        """关闭引擎"""
        try:
            if self.is_running:
                self.is_running = False
                
                # 发送退出命令
                if self.state != EngineState.ERROR:
                    self._send_command("quit")
                
                # 等待通信线程结束
                if self.communication_thread and self.communication_thread.is_alive():
                    self.communication_thread.join(timeout=3.0)
                
                # 终止进程
                if self.process:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                    finally:
                        self.process = None
                
                self._set_state(EngineState.TERMINATED)
                self.logger.info("引擎已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭引擎异常: {e}")
    
    def is_alive(self) -> bool:
        """检查引擎是否存活"""
        return (self.process is not None and 
                self.process.poll() is None and 
                self.is_running and 
                self.state not in [EngineState.ERROR, EngineState.TERMINATED])
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()


# 工厂函数
def create_pikafish_engine(config_path: Optional[str] = None) -> PikafishEngine:
    """
    创建Pikafish引擎的工厂函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        PikafishEngine: 引擎实例
    """
    config = ConfigManager(config_path) if config_path else ConfigManager()
    return PikafishEngine(config)


if __name__ == "__main__":
    # 测试代码
    def test_pikafish_engine():
        """测试Pikafish引擎功能"""
        print("测试Pikafish引擎...")
        
        # 创建引擎实例
        engine = create_pikafish_engine()
        
        # 添加回调函数
        def on_analysis(analysis: EngineAnalysis):
            print(f"分析结果: 深度{analysis.depth}, 评分{analysis.score}, 最佳走法{analysis.best_move}")
        
        def on_error(error: PikafishEngineError):
            print(f"引擎错误: {error}")
        
        engine.add_analysis_callback(on_analysis)
        engine.add_error_callback(on_error)
        
        # 获取统计信息
        stats = engine.get_stats()
        print(f"统计信息: {stats}")
        
        print("Pikafish引擎测试完成")
    
    test_pikafish_engine()
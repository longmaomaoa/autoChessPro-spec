#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋局状态变化监控模块

本模块负责整合棋盘检测和棋子识别功能，实现实时棋局状态监控和变化检测。
主要功能：
1. 集成棋盘检测器和棋子分类器
2. 实时监控棋局状态变化
3. 检测和验证走棋合法性
4. 提供棋局历史记录和回退功能
5. 异常棋局状态检测和处理

设计模式：
- 观察者模式：监控棋局状态变化
- 策略模式：不同的变化检测策略
- 状态模式：管理棋局状态转换
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any, Set
from pathlib import Path
import numpy as np
from collections import deque
import logging

from chess_ai.config.config_manager import ConfigManager
from chess_ai.core.move import Move
from chess_ai.core.board_state import BoardState
from chess_ai.core.piece import Piece, PieceColor, Position
from chess_ai.vision.board_detector import ChessBoardDetector, BoardRegion, DetectionStats as BoardDetectionStats
from chess_ai.vision.piece_classifier import ChessPieceClassifier, PieceType, ClassificationResult, RecognitionMethod
from chess_ai.utils.logger import get_logger


class MonitoringState(Enum):
    """监控状态枚举"""
    IDLE = "IDLE"                    # 空闲状态
    DETECTING = "DETECTING"          # 检测中
    ANALYZING = "ANALYZING"          # 分析中
    WAITING_MOVE = "WAITING_MOVE"    # 等待走棋
    MOVE_DETECTED = "MOVE_DETECTED"  # 检测到走棋
    ERROR = "ERROR"                  # 错误状态
    PAUSED = "PAUSED"               # 暂停状态


class ChangeDetectionMethod(Enum):
    """变化检测方法枚举"""
    PIXEL_DIFF = "PIXEL_DIFF"        # 像素差异检测
    PIECE_COUNT = "PIECE_COUNT"      # 棋子数量对比
    POSITION_HASH = "POSITION_HASH"  # 位置哈希对比
    HYBRID = "HYBRID"                # 混合方法


@dataclass
class BoardSnapshot:
    """棋盘快照"""
    timestamp: float
    board_state: BoardState
    image: np.ndarray
    board_region: Optional[BoardRegion]
    piece_detections: List[Any]  # PieceDetection objects
    confidence_score: float
    detection_method: RecognitionMethod
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp <= 0:
            self.timestamp = time.time()


@dataclass
class MoveDetection:
    """走棋检测结果"""
    move: Move
    confidence: float
    detection_time: float
    source_snapshot: BoardSnapshot
    target_snapshot: BoardSnapshot
    is_legal: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.detection_time <= 0:
            self.detection_time = time.time()


@dataclass
class MonitoringStats:
    """监控统计信息"""
    total_detections: int = 0
    successful_detections: int = 0
    failed_detections: int = 0
    illegal_moves_detected: int = 0
    average_detection_time: float = 0.0
    frames_processed: int = 0
    last_detection_time: float = 0.0
    uptime: float = 0.0
    detection_rate: float = 0.0  # 检测成功率
    
    def update_detection(self, success: bool, detection_time: float):
        """更新检测统计"""
        self.total_detections += 1
        if success:
            self.successful_detections += 1
        else:
            self.failed_detections += 1
        
        # 更新平均检测时间
        if self.total_detections > 1:
            self.average_detection_time = (
                (self.average_detection_time * (self.total_detections - 1) + detection_time) 
                / self.total_detections
            )
        else:
            self.average_detection_time = detection_time
        
        self.last_detection_time = time.time()
        self.detection_rate = self.successful_detections / self.total_detections if self.total_detections > 0 else 0.0


class BoardRecognitionError(Exception):
    """棋局识别异常"""
    def __init__(self, message: str, error_code: Optional[str] = None, snapshot: Optional[BoardSnapshot] = None):
        super().__init__(message)
        self.error_code = error_code
        self.snapshot = snapshot


class BoardRecognitionModule:
    """棋局识别监控模块"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化棋局识别模块
        
        Args:
            config: 配置管理器，如果为None则使用默认配置
        """
        self.config = config or ConfigManager()
        self.logger = get_logger(__name__)
        
        # 初始化核心组件
        self.board_detector = ChessBoardDetector(self.config)
        self.piece_classifier = ChessPieceClassifier(self.config)
        
        # 状态管理
        self.current_state = MonitoringState.IDLE
        self.is_monitoring = False
        self.is_paused = False
        
        # 数据存储
        self.current_snapshot: Optional[BoardSnapshot] = None
        self.previous_snapshot: Optional[BoardSnapshot] = None
        self.snapshot_history: deque = deque(maxlen=self.config.vision.board_recognition.max_history_size)
        self.move_history: List[MoveDetection] = []
        
        # 统计信息
        self.stats = MonitoringStats()
        self.start_time = time.time()
        
        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_lock = threading.RLock()
        
        # 事件回调
        self.move_detected_callbacks: List[Callable[[MoveDetection], None]] = []
        self.error_callbacks: List[Callable[[BoardRecognitionError], None]] = []
        self.state_change_callbacks: List[Callable[[MonitoringState, MonitoringState], None]] = []
        
        # 配置参数
        self._load_config_params()
        
        self.logger.info("棋局识别模块初始化完成")
    
    def _load_config_params(self):
        """加载配置参数"""
        recognition_config = self.config.vision.board_recognition
        
        self.detection_interval = recognition_config.detection_interval  # 检测间隔（秒）
        self.change_threshold = recognition_config.change_threshold      # 变化阈值
        self.confidence_threshold = recognition_config.confidence_threshold  # 置信度阈值
        self.max_retry_attempts = recognition_config.max_retry_attempts  # 最大重试次数
        self.move_validation_enabled = recognition_config.move_validation_enabled  # 走棋合法性验证
        self.change_detection_method = ChangeDetectionMethod(recognition_config.change_detection_method)
    
    def start_monitoring(self, image_source: Optional[Callable[[], np.ndarray]] = None):
        """
        开始监控棋局状态
        
        Args:
            image_source: 图像源函数，如果为None则使用屏幕捕获
        """
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.is_paused = False
        self._set_state(MonitoringState.DETECTING)
        
        # 设置图像源
        if image_source is None:
            # 使用默认的屏幕捕获
            from chess_ai.vision.screen_capture import ScreenCaptureModule
            screen_capture = ScreenCaptureModule(self.config)
            screen_capture.start_capture()
            self.image_source = lambda: screen_capture.capture_frame()
        else:
            self.image_source = image_source
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("开始监控棋局状态")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._set_state(MonitoringState.IDLE)
        
        # 等待监控线程结束
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("监控已停止")
    
    def pause_monitoring(self):
        """暂停监控"""
        if self.is_monitoring and not self.is_paused:
            self.is_paused = True
            self._set_state(MonitoringState.PAUSED)
            self.logger.info("监控已暂停")
    
    def resume_monitoring(self):
        """恢复监控"""
        if self.is_monitoring and self.is_paused:
            self.is_paused = False
            self._set_state(MonitoringState.DETECTING)
            self.logger.info("监控已恢复")
    
    def _monitoring_loop(self):
        """监控主循环"""
        self.start_time = time.time()
        
        while self.is_monitoring:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 捕获图像
                start_time = time.time()
                image = self.image_source()
                
                if image is None or image.size == 0:
                    self.logger.warning("获取图像失败")
                    time.sleep(self.detection_interval)
                    continue
                
                # 处理图像并检测变化
                self._process_image(image)
                
                # 更新统计信息
                processing_time = time.time() - start_time
                self.stats.frames_processed += 1
                self.stats.uptime = time.time() - self.start_time
                
                # 控制检测频率
                sleep_time = max(0, self.detection_interval - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                self._handle_error(BoardRecognitionError(f"监控异常: {e}", "MONITORING_ERROR"))
                time.sleep(1.0)  # 错误后等待一秒再继续
    
    def _process_image(self, image: np.ndarray):
        """
        处理图像并检测变化
        
        Args:
            image: 输入图像
        """
        try:
            self._set_state(MonitoringState.ANALYZING)
            
            # 检测棋盘
            board_region = self.board_detector.detect_board(image)
            if board_region is None or board_region.confidence < self.config.vision.board_detector.confidence_threshold:
                self.logger.debug("棋盘检测失败或置信度不足")
                return
            
            # 分类棋子
            classification_result = self.piece_classifier.classify_pieces(
                image, board_region, method=RecognitionMethod.HYBRID
            )
            
            if classification_result.average_confidence < self.confidence_threshold:
                self.logger.debug("棋子分类置信度不足")
                return
            
            # 创建棋局状态
            board_state = self._create_board_state(classification_result.detections)
            
            # 创建快照
            snapshot = BoardSnapshot(
                timestamp=time.time(),
                board_state=board_state,
                image=image.copy(),
                board_region=board_region,
                piece_detections=classification_result.detections,
                confidence_score=classification_result.average_confidence,
                detection_method=classification_result.method_used
            )
            
            # 检测变化
            if self.current_snapshot is not None:
                change_detected = self._detect_board_change(self.current_snapshot, snapshot)
                if change_detected:
                    move_detection = self._analyze_move(self.current_snapshot, snapshot)
                    if move_detection:
                        self._handle_move_detected(move_detection)
            
            # 更新快照
            self.previous_snapshot = self.current_snapshot
            self.current_snapshot = snapshot
            self.snapshot_history.append(snapshot)
            
            self._set_state(MonitoringState.WAITING_MOVE)
            
        except Exception as e:
            self.logger.error(f"图像处理异常: {e}")
            self._handle_error(BoardRecognitionError(f"图像处理异常: {e}", "IMAGE_PROCESSING_ERROR"))
    
    def _create_board_state(self, detections: List[Any]) -> BoardState:
        """
        根据检测结果创建棋局状态
        
        Args:
            detections: 棋子检测结果列表
            
        Returns:
            BoardState: 棋局状态对象
        """
        # 初始化空棋盘
        board = [[None for _ in range(9)] for _ in range(10)]
        
        # 根据检测结果填充棋盘
        for detection in detections:
            if detection.piece_type != PieceType.EMPTY and detection.confidence >= self.confidence_threshold:
                pos = detection.position
                if 0 <= pos.row < 10 and 0 <= pos.col < 9:
                    piece = self._piece_type_to_piece(detection.piece_type)
                    board[pos.row][pos.col] = piece
        
        return BoardState(board)
    
    def _piece_type_to_piece(self, piece_type: PieceType) -> Optional[Piece]:
        """
        将棋子类型转换为棋子对象
        
        Args:
            piece_type: 棋子类型
            
        Returns:
            Piece: 棋子对象
        """
        if piece_type == PieceType.EMPTY:
            return None
        
        # 确定颜色
        color = PieceColor.RED if piece_type.value.startswith("红") else PieceColor.BLACK
        
        # 根据类型创建棋子
        piece_map = {
            PieceType.RED_KING: "K", PieceType.BLACK_KING: "k",
            PieceType.RED_ADVISOR: "A", PieceType.BLACK_ADVISOR: "a",
            PieceType.RED_BISHOP: "B", PieceType.BLACK_BISHOP: "b",
            PieceType.RED_KNIGHT: "N", PieceType.BLACK_KNIGHT: "n",
            PieceType.RED_ROOK: "R", PieceType.BLACK_ROOK: "r",
            PieceType.RED_CANNON: "C", PieceType.BLACK_CANNON: "c",
            PieceType.RED_PAWN: "P", PieceType.BLACK_PAWN: "p",
        }
        
        piece_char = piece_map.get(piece_type)
        if piece_char:
            return Piece.from_fen_char(piece_char)
        
        return None
    
    def _detect_board_change(self, prev_snapshot: BoardSnapshot, curr_snapshot: BoardSnapshot) -> bool:
        """
        检测棋盘状态变化
        
        Args:
            prev_snapshot: 前一个快照
            curr_snapshot: 当前快照
            
        Returns:
            bool: 是否检测到变化
        """
        if self.change_detection_method == ChangeDetectionMethod.PIXEL_DIFF:
            return self._detect_pixel_difference(prev_snapshot, curr_snapshot)
        elif self.change_detection_method == ChangeDetectionMethod.PIECE_COUNT:
            return self._detect_piece_count_change(prev_snapshot, curr_snapshot)
        elif self.change_detection_method == ChangeDetectionMethod.POSITION_HASH:
            return self._detect_position_hash_change(prev_snapshot, curr_snapshot)
        else:  # HYBRID
            return (self._detect_piece_count_change(prev_snapshot, curr_snapshot) or 
                   self._detect_position_hash_change(prev_snapshot, curr_snapshot))
    
    def _detect_pixel_difference(self, prev_snapshot: BoardSnapshot, curr_snapshot: BoardSnapshot) -> bool:
        """基于像素差异的变化检测"""
        if prev_snapshot.board_region is None or curr_snapshot.board_region is None:
            return False
        
        # 计算图像差异
        prev_image = prev_snapshot.image
        curr_image = curr_snapshot.image
        
        if prev_image.shape != curr_image.shape:
            return True
        
        diff = np.mean(np.abs(prev_image.astype(float) - curr_image.astype(float)))
        return diff > self.change_threshold
    
    def _detect_piece_count_change(self, prev_snapshot: BoardSnapshot, curr_snapshot: BoardSnapshot) -> bool:
        """基于棋子数量变化的检测"""
        prev_count = len([d for d in prev_snapshot.piece_detections if d.piece_type != PieceType.EMPTY])
        curr_count = len([d for d in curr_snapshot.piece_detections if d.piece_type != PieceType.EMPTY])
        return prev_count != curr_count
    
    def _detect_position_hash_change(self, prev_snapshot: BoardSnapshot, curr_snapshot: BoardSnapshot) -> bool:
        """基于位置哈希的变化检测"""
        def get_position_hash(snapshot: BoardSnapshot) -> str:
            positions = []
            for detection in snapshot.piece_detections:
                if detection.piece_type != PieceType.EMPTY:
                    pos = detection.position
                    positions.append(f"{detection.piece_type.value}@{pos.row},{pos.col}")
            return hash(tuple(sorted(positions)))
        
        prev_hash = get_position_hash(prev_snapshot)
        curr_hash = get_position_hash(curr_snapshot)
        return prev_hash != curr_hash
    
    def _analyze_move(self, prev_snapshot: BoardSnapshot, curr_snapshot: BoardSnapshot) -> Optional[MoveDetection]:
        """
        分析棋局变化，识别具体走法
        
        Args:
            prev_snapshot: 前一个快照
            curr_snapshot: 当前快照
            
        Returns:
            MoveDetection: 走法检测结果
        """
        try:
            # 分析棋子位置变化
            prev_positions = self._get_piece_positions(prev_snapshot)
            curr_positions = self._get_piece_positions(curr_snapshot)
            
            # 找出消失和出现的棋子
            disappeared_pieces = prev_positions - curr_positions
            appeared_pieces = curr_positions - prev_positions
            
            # 简单情况：一个棋子移动
            if len(disappeared_pieces) == 1 and len(appeared_pieces) == 1:
                from_pos = list(disappeared_pieces)[0][1]  # (piece_type, position)
                to_pos = list(appeared_pieces)[0][1]
                piece_type = list(disappeared_pieces)[0][0]
                
                # 创建Move对象
                move = Move(from_pos, to_pos)
                
                # 验证走法合法性
                is_legal = True
                validation_errors = []
                if self.move_validation_enabled:
                    is_legal, validation_errors = self._validate_move(move, prev_snapshot.board_state)
                
                # 计算置信度
                confidence = min(prev_snapshot.confidence_score, curr_snapshot.confidence_score)
                
                move_detection = MoveDetection(
                    move=move,
                    confidence=confidence,
                    detection_time=time.time(),
                    source_snapshot=prev_snapshot,
                    target_snapshot=curr_snapshot,
                    is_legal=is_legal,
                    validation_errors=validation_errors
                )
                
                return move_detection
            
            # 复杂情况：吃子等
            elif len(disappeared_pieces) == 2 and len(appeared_pieces) == 1:
                # 可能是吃子走法
                # 这里需要更复杂的逻辑来判断哪个是移动的棋子，哪个是被吃的棋子
                # 暂时简化处理
                pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"走法分析异常: {e}")
            return None
    
    def _get_piece_positions(self, snapshot: BoardSnapshot) -> Set[Tuple[PieceType, Position]]:
        """
        获取快照中所有棋子的位置信息
        
        Args:
            snapshot: 棋盘快照
            
        Returns:
            Set[Tuple[PieceType, Position]]: 棋子类型和位置的集合
        """
        positions = set()
        for detection in snapshot.piece_detections:
            if detection.piece_type != PieceType.EMPTY and detection.confidence >= self.confidence_threshold:
                positions.add((detection.piece_type, detection.position))
        return positions
    
    def _validate_move(self, move: Move, board_state: BoardState) -> Tuple[bool, List[str]]:
        """
        验证走法的合法性
        
        Args:
            move: 走法
            board_state: 棋局状态
            
        Returns:
            Tuple[bool, List[str]]: 是否合法及错误信息列表
        """
        try:
            # 这里应该实现具体的中国象棋规则验证
            # 暂时返回简单验证结果
            errors = []
            
            # 基本边界检查
            if not (0 <= move.from_pos.row < 10 and 0 <= move.from_pos.col < 9):
                errors.append("起始位置超出棋盘范围")
            if not (0 <= move.to_pos.row < 10 and 0 <= move.to_pos.col < 9):
                errors.append("目标位置超出棋盘范围")
            
            # 检查起始位置是否有棋子
            from_piece = board_state.get_piece_at(move.from_pos)
            if from_piece is None:
                errors.append("起始位置没有棋子")
            
            # 检查是否是移动到相同位置
            if move.from_pos == move.to_pos:
                errors.append("起始位置和目标位置相同")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"走法验证异常: {e}"]
    
    def _handle_move_detected(self, move_detection: MoveDetection):
        """
        处理检测到的走法
        
        Args:
            move_detection: 走法检测结果
        """
        try:
            self._set_state(MonitoringState.MOVE_DETECTED)
            
            # 记录走法
            self.move_history.append(move_detection)
            
            # 更新统计信息
            self.stats.update_detection(move_detection.is_legal, move_detection.detection_time - move_detection.source_snapshot.timestamp)
            
            if not move_detection.is_legal:
                self.stats.illegal_moves_detected += 1
                self.logger.warning(f"检测到非法走法: {move_detection.move}, 错误: {move_detection.validation_errors}")
            else:
                self.logger.info(f"检测到走法: {move_detection.move}, 置信度: {move_detection.confidence:.3f}")
            
            # 调用回调函数
            for callback in self.move_detected_callbacks:
                try:
                    callback(move_detection)
                except Exception as e:
                    self.logger.error(f"走法检测回调异常: {e}")
            
        except Exception as e:
            self.logger.error(f"处理走法检测异常: {e}")
    
    def _handle_error(self, error: BoardRecognitionError):
        """
        处理错误
        
        Args:
            error: 识别错误
        """
        self._set_state(MonitoringState.ERROR)
        
        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"错误回调异常: {e}")
    
    def _set_state(self, new_state: MonitoringState):
        """
        设置监控状态
        
        Args:
            new_state: 新状态
        """
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            
            # 调用状态变化回调
            for callback in self.state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    self.logger.error(f"状态变化回调异常: {e}")
    
    def add_move_detected_callback(self, callback: Callable[[MoveDetection], None]):
        """添加走法检测回调"""
        self.move_detected_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[BoardRecognitionError], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def add_state_change_callback(self, callback: Callable[[MonitoringState, MonitoringState], None]):
        """添加状态变化回调"""
        self.state_change_callbacks.append(callback)
    
    def get_current_board_state(self) -> Optional[BoardState]:
        """获取当前棋局状态"""
        return self.current_snapshot.board_state if self.current_snapshot else None
    
    def get_move_history(self, limit: Optional[int] = None) -> List[MoveDetection]:
        """
        获取走法历史
        
        Args:
            limit: 限制返回的走法数量
            
        Returns:
            List[MoveDetection]: 走法历史列表
        """
        if limit is None:
            return self.move_history.copy()
        return self.move_history[-limit:]
    
    def get_stats(self) -> MonitoringStats:
        """获取监控统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = MonitoringStats()
        self.start_time = time.time()
        self.logger.info("统计信息已重置")
    
    def clear_history(self):
        """清除历史记录"""
        with self.monitoring_lock:
            self.snapshot_history.clear()
            self.move_history.clear()
        self.logger.info("历史记录已清除")
    
    def export_game_data(self, format_type: str = "json") -> str:
        """
        导出棋局数据
        
        Args:
            format_type: 导出格式 ("json", "pgn", "fen")
            
        Returns:
            str: 导出的数据
        """
        # 实现数据导出功能
        if format_type == "json":
            import json
            data = {
                "moves": [{"from": f"{m.move.from_pos.row},{m.move.from_pos.col}",
                          "to": f"{m.move.to_pos.row},{m.move.to_pos.col}",
                          "timestamp": m.detection_time,
                          "confidence": m.confidence,
                          "is_legal": m.is_legal} for m in self.move_history],
                "stats": {
                    "total_detections": self.stats.total_detections,
                    "successful_detections": self.stats.successful_detections,
                    "detection_rate": self.stats.detection_rate,
                    "average_detection_time": self.stats.average_detection_time
                }
            }
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        # 其他格式的导出可以后续实现
        return ""
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.is_monitoring:
            self.stop_monitoring()


# 使用示例和工厂函数
def create_board_recognition_module(config_path: Optional[str] = None) -> BoardRecognitionModule:
    """
    创建棋局识别模块的工厂函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        BoardRecognitionModule: 棋局识别模块实例
    """
    config = ConfigManager(config_path) if config_path else ConfigManager()
    return BoardRecognitionModule(config)


if __name__ == "__main__":
    # 测试代码
    def test_board_recognition():
        """测试棋局识别功能"""
        print("测试棋局识别模块...")
        
        # 创建模块实例
        module = create_board_recognition_module()
        
        # 添加回调函数
        def on_move_detected(move_detection: MoveDetection):
            print(f"检测到走法: {move_detection.move}")
        
        def on_error(error: BoardRecognitionError):
            print(f"发生错误: {error}")
        
        module.add_move_detected_callback(on_move_detected)
        module.add_error_callback(on_error)
        
        # 获取统计信息
        stats = module.get_stats()
        print(f"统计信息: {stats}")
        
        print("棋局识别模块测试完成")
    
    test_board_recognition()
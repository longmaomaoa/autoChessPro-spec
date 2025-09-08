#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉识别鲁棒性处理模块

本模块负责提高视觉识别系统的鲁棒性和可靠性，包括：
1. 低置信度和遮挡情况的处理
2. 异常棋局状态的检测和警告
3. 识别错误的恢复和回退机制
4. 多帧融合和时间一致性检查
5. 自适应阈值和参数调优

设计原则：
- 优雅降级：在识别质量下降时保持基本功能
- 多重验证：使用多种策略验证识别结果
- 自适应调整：根据环境变化动态调整参数
- 状态保持：维护历史状态以支持回退和恢复
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable, Set
from pathlib import Path
import numpy as np
from collections import deque, Counter
import logging
import statistics

from chess_ai.config.config_manager import ConfigManager
from chess_ai.data.pieces import Position, BoardState, Piece
from chess_ai.vision.piece_classifier import PieceType, ClassificationResult, PieceDetection
from chess_ai.vision.board_detector import BoardRegion
from chess_ai.utils.logger import Logger


class RobustnessLevel(Enum):
    """鲁棒性处理级别"""
    STRICT = "STRICT"        # 严格模式，高质量要求
    BALANCED = "BALANCED"    # 平衡模式，质量与性能兼顾
    TOLERANT = "TOLERANT"    # 宽容模式，优先保证连续性
    ADAPTIVE = "ADAPTIVE"    # 自适应模式，动态调整


class AnomalyType(Enum):
    """异常类型枚举"""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"          # 低置信度
    PARTIAL_OCCLUSION = "PARTIAL_OCCLUSION"    # 部分遮挡
    COMPLETE_OCCLUSION = "COMPLETE_OCCLUSION"  # 完全遮挡
    LIGHTING_CHANGE = "LIGHTING_CHANGE"        # 光照变化
    PERSPECTIVE_SHIFT = "PERSPECTIVE_SHIFT"    # 视角偏移
    PIECE_COUNT_ANOMALY = "PIECE_COUNT_ANOMALY"  # 棋子数量异常
    ILLEGAL_POSITION = "ILLEGAL_POSITION"      # 非法位置
    RECOGNITION_FAILURE = "RECOGNITION_FAILURE"  # 识别失败
    TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"  # 时间不一致性


@dataclass
class AnomalyDetection:
    """异常检测结果"""
    anomaly_type: AnomalyType
    severity: float  # 严重程度 [0.0, 1.0]
    confidence: float  # 检测置信度 [0.0, 1.0]
    affected_positions: List[Position] = field(default_factory=list)
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    recovery_suggestions: List[str] = field(default_factory=list)


@dataclass
class RecoveryAction:
    """恢复动作"""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 优先级，数字越大优先级越高
    description: str = ""
    
    def __lt__(self, other):
        """支持优先级排序"""
        return self.priority < other.priority


@dataclass
class RobustnessStats:
    """鲁棒性处理统计"""
    total_frames_processed: int = 0
    anomalies_detected: int = 0
    recovery_actions_taken: int = 0
    successful_recoveries: int = 0
    false_positives: int = 0  # 误报数量
    detection_accuracy: float = 0.0
    average_recovery_time: float = 0.0
    anomaly_type_counts: Dict[str, int] = field(default_factory=dict)
    
    def update_anomaly_detection(self, anomaly: AnomalyDetection, recovery_successful: bool = False):
        """更新异常检测统计"""
        self.anomalies_detected += 1
        anomaly_type_str = anomaly.anomaly_type.value
        self.anomaly_type_counts[anomaly_type_str] = self.anomaly_type_counts.get(anomaly_type_str, 0) + 1
        
        if recovery_successful:
            self.successful_recoveries += 1


class RobustnessHandler:
    """视觉识别鲁棒性处理器"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化鲁棒性处理器
        
        Args:
            config: 配置管理器
        """
        self.config = config or ConfigManager()
        self.logger = Logger(__name__)
        
        # 配置参数
        self._load_config_params()
        
        # 状态管理
        self.current_level = RobustnessLevel.BALANCED
        self.is_active = False
        
        # 历史数据存储
        self.detection_history: deque = deque(maxlen=self.max_history_size)
        self.confidence_history: deque = deque(maxlen=self.max_history_size)
        self.anomaly_history: deque = deque(maxlen=self.max_history_size)
        
        # 统计信息
        self.stats = RobustnessStats()
        
        # 自适应参数
        self.adaptive_confidence_threshold = self.base_confidence_threshold
        self.adaptive_quality_threshold = self.base_quality_threshold
        
        # 回调函数
        self.anomaly_callbacks: List[Callable[[AnomalyDetection], None]] = []
        self.recovery_callbacks: List[Callable[[RecoveryAction], None]] = []
        
        # 线程安全
        self.processing_lock = threading.RLock()
        
        self.logger.info("视觉识别鲁棒性处理器初始化完成")
    
    def _load_config_params(self):
        """加载配置参数"""
        robustness_config = self.config.vision.robustness
        
        # 基础阈值
        self.base_confidence_threshold = robustness_config.base_confidence_threshold
        self.base_quality_threshold = robustness_config.base_quality_threshold
        self.anomaly_detection_threshold = robustness_config.anomaly_detection_threshold
        
        # 历史数据相关
        self.max_history_size = robustness_config.max_history_size
        self.temporal_window_size = robustness_config.temporal_window_size
        
        # 多帧融合相关
        self.enable_multi_frame_fusion = robustness_config.enable_multi_frame_fusion
        self.fusion_frame_count = robustness_config.fusion_frame_count
        self.fusion_confidence_weight = robustness_config.fusion_confidence_weight
        
        # 自适应相关
        self.enable_adaptive_thresholds = robustness_config.enable_adaptive_thresholds
        self.threshold_adaptation_rate = robustness_config.threshold_adaptation_rate
        
        # 恢复机制相关
        self.enable_automatic_recovery = robustness_config.enable_automatic_recovery
        self.max_recovery_attempts = robustness_config.max_recovery_attempts
    
    def set_robustness_level(self, level: RobustnessLevel):
        """
        设置鲁棒性处理级别
        
        Args:
            level: 鲁棒性级别
        """
        self.current_level = level
        self._adjust_parameters_for_level(level)
        self.logger.info(f"鲁棒性处理级别设置为: {level.value}")
    
    def _adjust_parameters_for_level(self, level: RobustnessLevel):
        """根据级别调整参数"""
        if level == RobustnessLevel.STRICT:
            self.adaptive_confidence_threshold = self.base_confidence_threshold + 0.1
            self.adaptive_quality_threshold = self.base_quality_threshold + 0.15
            self.anomaly_detection_threshold = 0.3
        elif level == RobustnessLevel.BALANCED:
            self.adaptive_confidence_threshold = self.base_confidence_threshold
            self.adaptive_quality_threshold = self.base_quality_threshold
            self.anomaly_detection_threshold = 0.5
        elif level == RobustnessLevel.TOLERANT:
            self.adaptive_confidence_threshold = self.base_confidence_threshold - 0.1
            self.adaptive_quality_threshold = self.base_quality_threshold - 0.1
            self.anomaly_detection_threshold = 0.7
        else:  # ADAPTIVE
            # 自适应模式将根据历史表现动态调整
            pass
    
    def process_classification_result(self, result: ClassificationResult, 
                                    board_region: Optional[BoardRegion] = None) -> ClassificationResult:
        """
        处理分类结果，提高鲁棒性
        
        Args:
            result: 原始分类结果
            board_region: 棋盘区域信息
            
        Returns:
            ClassificationResult: 处理后的分类结果
        """
        with self.processing_lock:
            self.stats.total_frames_processed += 1
            
            # 检测异常
            anomalies = self._detect_anomalies(result, board_region)
            
            # 记录历史数据
            self._update_history(result, anomalies)
            
            # 处理检测到的异常
            processed_result = result
            if anomalies:
                processed_result = self._handle_anomalies(result, anomalies, board_region)
            
            # 应用多帧融合（如果启用）
            if self.enable_multi_frame_fusion and len(self.detection_history) >= self.fusion_frame_count:
                processed_result = self._apply_multi_frame_fusion(processed_result)
            
            # 更新自适应参数
            if self.enable_adaptive_thresholds:
                self._update_adaptive_thresholds(processed_result)
            
            return processed_result
    
    def _detect_anomalies(self, result: ClassificationResult, 
                         board_region: Optional[BoardRegion]) -> List[AnomalyDetection]:
        """
        检测识别结果中的异常
        
        Args:
            result: 分类结果
            board_region: 棋盘区域
            
        Returns:
            List[AnomalyDetection]: 检测到的异常列表
        """
        anomalies = []
        
        # 1. 检测低置信度
        low_confidence_detections = [d for d in result.detections 
                                   if d.confidence < self.adaptive_confidence_threshold]
        if low_confidence_detections:
            anomaly = AnomalyDetection(
                anomaly_type=AnomalyType.LOW_CONFIDENCE,
                severity=1.0 - result.average_confidence,
                confidence=0.9,
                affected_positions=[d.position for d in low_confidence_detections],
                description=f"检测到{len(low_confidence_detections)}个低置信度棋子",
                recovery_suggestions=["增加光照", "调整摄像头角度", "清洁镜头"]
            )
            anomalies.append(anomaly)
        
        # 2. 检测棋子数量异常
        piece_count = len([d for d in result.detections if d.piece_type != PieceType.EMPTY])
        if piece_count < 10 or piece_count > 32:  # 中国象棋棋子数量范围
            anomaly = AnomalyDetection(
                anomaly_type=AnomalyType.PIECE_COUNT_ANOMALY,
                severity=min(1.0, abs(piece_count - 32) / 32.0),
                confidence=0.95,
                description=f"检测到异常棋子数量: {piece_count}",
                recovery_suggestions=["检查棋盘完整性", "调整检测参数", "重新校准棋盘"]
            )
            anomalies.append(anomaly)
        
        # 3. 检测时间一致性问题
        if len(self.detection_history) >= 3:
            temporal_anomaly = self._detect_temporal_inconsistency(result)
            if temporal_anomaly:
                anomalies.append(temporal_anomaly)
        
        # 4. 检测识别质量问题
        if board_region and board_region.confidence < self.adaptive_quality_threshold:
            anomaly = AnomalyDetection(
                anomaly_type=AnomalyType.RECOGNITION_FAILURE,
                severity=1.0 - board_region.confidence,
                confidence=0.8,
                description=f"棋盘检测质量低: {board_region.confidence:.3f}",
                recovery_suggestions=["改善光照条件", "清理棋盘表面", "调整摄像头位置"]
            )
            anomalies.append(anomaly)
        
        # 5. 检测非法位置
        illegal_positions = self._detect_illegal_positions(result.detections)
        if illegal_positions:
            anomaly = AnomalyDetection(
                anomaly_type=AnomalyType.ILLEGAL_POSITION,
                severity=len(illegal_positions) / len(result.detections),
                confidence=0.9,
                affected_positions=illegal_positions,
                description=f"检测到{len(illegal_positions)}个非法棋子位置",
                recovery_suggestions=["检查棋盘标定", "重新校准坐标系统"]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_temporal_inconsistency(self, current_result: ClassificationResult) -> Optional[AnomalyDetection]:
        """检测时间一致性异常"""
        if len(self.detection_history) < 2:
            return None
        
        # 比较当前结果与历史结果的一致性
        prev_result = self.detection_history[-1]
        
        # 计算棋子位置变化
        current_positions = {(d.piece_type, d.position) for d in current_result.detections 
                           if d.piece_type != PieceType.EMPTY}
        prev_positions = {(d.piece_type, d.position) for d in prev_result.detections 
                        if d.piece_type != PieceType.EMPTY}
        
        # 计算变化率
        total_pieces = max(len(current_positions), len(prev_positions))
        if total_pieces == 0:
            return None
        
        unchanged_pieces = len(current_positions & prev_positions)
        change_rate = 1.0 - (unchanged_pieces / total_pieces)
        
        # 如果变化率过高，认为是时间不一致
        if change_rate > 0.5:  # 超过50%的棋子发生变化
            return AnomalyDetection(
                anomaly_type=AnomalyType.TEMPORAL_INCONSISTENCY,
                severity=change_rate,
                confidence=0.7,
                description=f"检测到异常高的棋子变化率: {change_rate:.2f}",
                recovery_suggestions=["检查相机稳定性", "降低检测频率", "使用时间滤波"]
            )
        
        return None
    
    def _detect_illegal_positions(self, detections: List[PieceDetection]) -> List[Position]:
        """检测非法的棋子位置"""
        illegal_positions = []
        
        for detection in detections:
            pos = detection.position
            # 检查位置是否在棋盘范围内
            if not (0 <= pos.row < 10 and 0 <= pos.col < 9):
                illegal_positions.append(pos)
        
        return illegal_positions
    
    def _handle_anomalies(self, result: ClassificationResult, 
                         anomalies: List[AnomalyDetection],
                         board_region: Optional[BoardRegion]) -> ClassificationResult:
        """
        处理检测到的异常
        
        Args:
            result: 原始结果
            anomalies: 异常列表
            board_region: 棋盘区域
            
        Returns:
            ClassificationResult: 处理后的结果
        """
        processed_result = result
        
        for anomaly in anomalies:
            self.stats.update_anomaly_detection(anomaly)
            
            # 调用异常回调
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    self.logger.error(f"异常回调错误: {e}")
            
            # 根据异常类型执行相应处理
            if anomaly.anomaly_type == AnomalyType.LOW_CONFIDENCE:
                processed_result = self._handle_low_confidence(processed_result, anomaly)
            elif anomaly.anomaly_type == AnomalyType.PIECE_COUNT_ANOMALY:
                processed_result = self._handle_piece_count_anomaly(processed_result, anomaly)
            elif anomaly.anomaly_type == AnomalyType.TEMPORAL_INCONSISTENCY:
                processed_result = self._handle_temporal_inconsistency(processed_result, anomaly)
            elif anomaly.anomaly_type == AnomalyType.ILLEGAL_POSITION:
                processed_result = self._handle_illegal_positions(processed_result, anomaly)
        
        return processed_result
    
    def _handle_low_confidence(self, result: ClassificationResult, 
                              anomaly: AnomalyDetection) -> ClassificationResult:
        """处理低置信度异常"""
        if not self.enable_automatic_recovery:
            return result
        
        # 策略1: 使用历史数据填补
        if len(self.detection_history) > 0:
            result = self._fill_with_historical_data(result, anomaly.affected_positions)
        
        # 策略2: 降低置信度阈值（临时）
        recovery_action = RecoveryAction(
            action_type="LOWER_CONFIDENCE_THRESHOLD",
            parameters={"original_threshold": self.adaptive_confidence_threshold,
                       "new_threshold": self.adaptive_confidence_threshold * 0.8},
            priority=5,
            description="临时降低置信度阈值"
        )
        
        self._execute_recovery_action(recovery_action)
        return result
    
    def _handle_piece_count_anomaly(self, result: ClassificationResult, 
                                   anomaly: AnomalyDetection) -> ClassificationResult:
        """处理棋子数量异常"""
        # 如果棋子数量过少，可能是检测遗漏
        piece_count = len([d for d in result.detections if d.piece_type != PieceType.EMPTY])
        
        if piece_count < 20 and len(self.detection_history) > 0:
            # 尝试使用历史数据补充缺失的棋子
            return self._supplement_missing_pieces(result)
        
        return result
    
    def _handle_temporal_inconsistency(self, result: ClassificationResult, 
                                     anomaly: AnomalyDetection) -> ClassificationResult:
        """处理时间不一致性异常"""
        # 使用时间加权平均来平滑结果
        if len(self.detection_history) >= 2:
            return self._apply_temporal_smoothing(result)
        
        return result
    
    def _handle_illegal_positions(self, result: ClassificationResult, 
                                anomaly: AnomalyDetection) -> ClassificationResult:
        """处理非法位置异常"""
        # 过滤掉非法位置的检测结果
        valid_detections = []
        for detection in result.detections:
            pos = detection.position
            if 0 <= pos.row < 10 and 0 <= pos.col < 9:
                valid_detections.append(detection)
        
        # 更新结果
        filtered_result = ClassificationResult(
            detections=valid_detections,
            method_used=result.method_used,
            processing_time=result.processing_time,
            total_pieces_found=len(valid_detections),
            average_confidence=np.mean([d.confidence for d in valid_detections]) if valid_detections else 0.0
        )
        
        return filtered_result
    
    def _fill_with_historical_data(self, result: ClassificationResult, 
                                  affected_positions: List[Position]) -> ClassificationResult:
        """使用历史数据填补低置信度位置"""
        if not self.detection_history:
            return result
        
        # 获取最近的历史数据
        recent_history = list(self.detection_history)[-3:]  # 最近3帧
        
        # 为每个受影响的位置寻找历史数据
        enhanced_detections = result.detections.copy()
        
        for pos in affected_positions:
            # 在历史数据中查找该位置的稳定检测结果
            historical_detections = []
            for hist_result in recent_history:
                for detection in hist_result.detections:
                    if detection.position == pos and detection.confidence > self.adaptive_confidence_threshold:
                        historical_detections.append(detection)
            
            if historical_detections:
                # 使用最频繁的棋子类型
                piece_types = [d.piece_type for d in historical_detections]
                most_common_type = Counter(piece_types).most_common(1)[0][0]
                avg_confidence = np.mean([d.confidence for d in historical_detections 
                                        if d.piece_type == most_common_type])
                
                # 更新或添加检测结果
                for i, detection in enumerate(enhanced_detections):
                    if detection.position == pos:
                        enhanced_detections[i] = PieceDetection(
                            position=pos,
                            piece_type=most_common_type,
                            confidence=min(avg_confidence, self.adaptive_confidence_threshold + 0.1),
                            bbox=detection.bbox
                        )
                        break
        
        # 返回增强后的结果
        return ClassificationResult(
            detections=enhanced_detections,
            method_used=result.method_used,
            processing_time=result.processing_time,
            total_pieces_found=len([d for d in enhanced_detections if d.piece_type != PieceType.EMPTY]),
            average_confidence=np.mean([d.confidence for d in enhanced_detections])
        )
    
    def _supplement_missing_pieces(self, result: ClassificationResult) -> ClassificationResult:
        """补充缺失的棋子"""
        if not self.detection_history:
            return result
        
        # 分析历史数据中经常出现但当前缺失的棋子
        recent_history = list(self.detection_history)[-5:]  # 最近5帧
        
        # 统计各位置的棋子出现频率
        position_counts = {}
        for hist_result in recent_history:
            for detection in hist_result.detections:
                if detection.piece_type != PieceType.EMPTY:
                    key = (detection.position, detection.piece_type)
                    position_counts[key] = position_counts.get(key, 0) + 1
        
        # 找出经常出现但当前缺失的棋子
        current_positions = {(d.position, d.piece_type) for d in result.detections 
                           if d.piece_type != PieceType.EMPTY}
        
        supplemented_detections = result.detections.copy()
        
        for (pos, piece_type), count in position_counts.items():
            if count >= 3 and (pos, piece_type) not in current_positions:  # 至少出现3次
                # 添加缺失的棋子检测
                supplemented_detections.append(PieceDetection(
                    position=pos,
                    piece_type=piece_type,
                    confidence=self.adaptive_confidence_threshold * 0.9,  # 稍低的置信度
                    bbox=None
                ))
        
        return ClassificationResult(
            detections=supplemented_detections,
            method_used=result.method_used,
            processing_time=result.processing_time,
            total_pieces_found=len([d for d in supplemented_detections if d.piece_type != PieceType.EMPTY]),
            average_confidence=np.mean([d.confidence for d in supplemented_detections])
        )
    
    def _apply_temporal_smoothing(self, result: ClassificationResult) -> ClassificationResult:
        """应用时间平滑"""
        if len(self.detection_history) < 2:
            return result
        
        # 获取历史结果进行加权平均
        recent_results = list(self.detection_history)[-3:]  # 最近3帧
        weights = [0.5, 0.3, 0.2]  # 时间加权，越近权重越大
        
        # 位置映射：每个位置的检测结果列表
        position_detections = {}
        
        # 收集当前结果
        for detection in result.detections:
            pos = detection.position
            if pos not in position_detections:
                position_detections[pos] = []
            position_detections[pos].append((detection, 1.0))  # 当前帧权重最大
        
        # 收集历史结果
        for i, hist_result in enumerate(recent_results):
            weight = weights[min(i, len(weights)-1)]
            for detection in hist_result.detections:
                pos = detection.position
                if pos not in position_detections:
                    position_detections[pos] = []
                position_detections[pos].append((detection, weight))
        
        # 对每个位置进行加权融合
        smoothed_detections = []
        for pos, detections_with_weights in position_detections.items():
            if not detections_with_weights:
                continue
            
            # 按棋子类型分组
            type_groups = {}
            for detection, weight in detections_with_weights:
                piece_type = detection.piece_type
                if piece_type not in type_groups:
                    type_groups[piece_type] = []
                type_groups[piece_type].append((detection, weight))
            
            # 选择加权得分最高的类型
            best_type = None
            best_score = 0.0
            best_confidence = 0.0
            
            for piece_type, group in type_groups.items():
                total_weight = sum(w for _, w in group)
                avg_confidence = sum(d.confidence * w for d, w in group) / total_weight
                score = total_weight * avg_confidence
                
                if score > best_score:
                    best_score = score
                    best_type = piece_type
                    best_confidence = avg_confidence
            
            if best_type is not None:
                smoothed_detections.append(PieceDetection(
                    position=pos,
                    piece_type=best_type,
                    confidence=best_confidence,
                    bbox=None
                ))
        
        return ClassificationResult(
            detections=smoothed_detections,
            method_used=result.method_used,
            processing_time=result.processing_time,
            total_pieces_found=len([d for d in smoothed_detections if d.piece_type != PieceType.EMPTY]),
            average_confidence=np.mean([d.confidence for d in smoothed_detections])
        )
    
    def _apply_multi_frame_fusion(self, result: ClassificationResult) -> ClassificationResult:
        """应用多帧融合"""
        if len(self.detection_history) < self.fusion_frame_count:
            return result
        
        # 获取用于融合的帧
        fusion_frames = list(self.detection_history)[-self.fusion_frame_count:]
        fusion_frames.append(result)  # 包含当前帧
        
        # 创建位置-检测映射
        position_map = {}
        
        for frame_idx, frame_result in enumerate(fusion_frames):
            frame_weight = (frame_idx + 1) / len(fusion_frames)  # 线性增加权重
            
            for detection in frame_result.detections:
                pos = detection.position
                if pos not in position_map:
                    position_map[pos] = []
                
                position_map[pos].append({
                    'detection': detection,
                    'weight': frame_weight * detection.confidence * self.fusion_confidence_weight,
                    'frame_idx': frame_idx
                })
        
        # 融合每个位置的检测结果
        fused_detections = []
        
        for pos, detection_list in position_map.items():
            if not detection_list:
                continue
            
            # 按棋子类型分组
            type_groups = {}
            for item in detection_list:
                piece_type = item['detection'].piece_type
                if piece_type not in type_groups:
                    type_groups[piece_type] = []
                type_groups[piece_type].append(item)
            
            # 选择加权得分最高的类型
            best_type = None
            best_score = 0.0
            best_confidence = 0.0
            
            for piece_type, group in type_groups.items():
                total_weight = sum(item['weight'] for item in group)
                avg_confidence = sum(item['detection'].confidence * item['weight'] for item in group) / total_weight if total_weight > 0 else 0
                
                if total_weight > best_score:
                    best_score = total_weight
                    best_type = piece_type
                    best_confidence = min(avg_confidence, 1.0)
            
            if best_type is not None and best_confidence > self.adaptive_confidence_threshold * 0.7:
                fused_detections.append(PieceDetection(
                    position=pos,
                    piece_type=best_type,
                    confidence=best_confidence,
                    bbox=None
                ))
        
        return ClassificationResult(
            detections=fused_detections,
            method_used=result.method_used,
            processing_time=result.processing_time,
            total_pieces_found=len([d for d in fused_detections if d.piece_type != PieceType.EMPTY]),
            average_confidence=np.mean([d.confidence for d in fused_detections])
        )
    
    def _update_adaptive_thresholds(self, result: ClassificationResult):
        """更新自适应阈值"""
        if self.current_level != RobustnessLevel.ADAPTIVE:
            return
        
        # 记录当前置信度
        self.confidence_history.append(result.average_confidence)
        
        if len(self.confidence_history) >= 10:  # 有足够的历史数据
            # 计算置信度统计
            recent_confidences = list(self.confidence_history)[-10:]
            avg_confidence = statistics.mean(recent_confidences)
            confidence_std = statistics.stdev(recent_confidences) if len(recent_confidences) > 1 else 0
            
            # 动态调整置信度阈值
            if avg_confidence > self.base_confidence_threshold + 0.15:
                # 识别质量很好，可以提高要求
                new_threshold = min(
                    self.adaptive_confidence_threshold + self.threshold_adaptation_rate,
                    self.base_confidence_threshold + 0.2
                )
            elif avg_confidence < self.base_confidence_threshold - 0.1:
                # 识别质量不佳，降低要求
                new_threshold = max(
                    self.adaptive_confidence_threshold - self.threshold_adaptation_rate,
                    self.base_confidence_threshold - 0.15
                )
            else:
                # 保持当前阈值
                new_threshold = self.adaptive_confidence_threshold
            
            # 平滑更新阈值
            self.adaptive_confidence_threshold = (
                0.9 * self.adaptive_confidence_threshold + 0.1 * new_threshold
            )
    
    def _execute_recovery_action(self, action: RecoveryAction):
        """执行恢复动作"""
        try:
            self.stats.recovery_actions_taken += 1
            
            # 调用恢复回调
            for callback in self.recovery_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    self.logger.error(f"恢复回调错误: {e}")
            
            self.logger.info(f"执行恢复动作: {action.description}")
            
            # 这里可以添加具体的恢复动作实现
            
        except Exception as e:
            self.logger.error(f"执行恢复动作失败: {e}")
    
    def _update_history(self, result: ClassificationResult, anomalies: List[AnomalyDetection]):
        """更新历史数据"""
        self.detection_history.append(result)
        if anomalies:
            self.anomaly_history.extend(anomalies)
    
    def add_anomaly_callback(self, callback: Callable[[AnomalyDetection], None]):
        """添加异常检测回调"""
        self.anomaly_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[RecoveryAction], None]):
        """添加恢复动作回调"""
        self.recovery_callbacks.append(callback)
    
    def get_stats(self) -> RobustnessStats:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = RobustnessStats()
        self.logger.info("鲁棒性处理统计信息已重置")
    
    def clear_history(self):
        """清除历史数据"""
        with self.processing_lock:
            self.detection_history.clear()
            self.confidence_history.clear()
            self.anomaly_history.clear()
        self.logger.info("历史数据已清除")
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """获取当前阈值"""
        return {
            "confidence_threshold": self.adaptive_confidence_threshold,
            "quality_threshold": self.adaptive_quality_threshold,
            "anomaly_detection_threshold": self.anomaly_detection_threshold
        }
    
    def export_diagnostics(self) -> Dict[str, Any]:
        """导出诊断信息"""
        return {
            "stats": self.stats.__dict__,
            "current_level": self.current_level.value,
            "thresholds": self.get_current_thresholds(),
            "history_sizes": {
                "detection_history": len(self.detection_history),
                "confidence_history": len(self.confidence_history),
                "anomaly_history": len(self.anomaly_history)
            },
            "recent_anomalies": [
                {
                    "type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity,
                    "timestamp": anomaly.timestamp
                }
                for anomaly in list(self.anomaly_history)[-10:]  # 最近10个异常
            ]
        }


def create_robustness_handler(config_path: Optional[str] = None) -> RobustnessHandler:
    """
    创建鲁棒性处理器的工厂函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        RobustnessHandler: 鲁棒性处理器实例
    """
    config = ConfigManager(config_path) if config_path else ConfigManager()
    return RobustnessHandler(config)


if __name__ == "__main__":
    # 测试代码
    def test_robustness_handler():
        """测试鲁棒性处理器功能"""
        print("测试视觉识别鲁棒性处理器...")
        
        # 创建处理器实例
        handler = create_robustness_handler()
        
        # 添加回调函数
        def on_anomaly_detected(anomaly: AnomalyDetection):
            print(f"检测到异常: {anomaly.anomaly_type.value} - {anomaly.description}")
        
        def on_recovery_action(action: RecoveryAction):
            print(f"执行恢复动作: {action.description}")
        
        handler.add_anomaly_callback(on_anomaly_detected)
        handler.add_recovery_callback(on_recovery_action)
        
        # 获取统计信息
        stats = handler.get_stats()
        print(f"统计信息: {stats}")
        
        # 导出诊断信息
        diagnostics = handler.export_diagnostics()
        print(f"诊断信息: {diagnostics}")
        
        print("鲁棒性处理器测试完成")
    
    test_robustness_handler()
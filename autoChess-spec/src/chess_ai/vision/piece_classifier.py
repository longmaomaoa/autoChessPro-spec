#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋棋子识别分类器 - 基于YOLOv8和传统图像处理的混合识别系统
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

from chess_ai.utils.logger import get_logger

logger = get_logger(__name__)


class PieceType(Enum):
    """象棋棋子类型枚举"""
    # 红方棋子
    RED_KING = "红帅"      # K - King
    RED_ADVISOR = "红仕"   # A - Advisor  
    RED_BISHOP = "红相"    # B - Bishop
    RED_KNIGHT = "红马"    # N - Knight
    RED_ROOK = "红车"      # R - Rook
    RED_CANNON = "红炮"    # C - Cannon
    RED_PAWN = "红兵"      # P - Pawn
    
    # 黑方棋子
    BLACK_KING = "黑将"    # k - King
    BLACK_ADVISOR = "黑士" # a - Advisor
    BLACK_BISHOP = "黑象"  # b - Bishop  
    BLACK_KNIGHT = "黑马"  # n - Knight
    BLACK_ROOK = "黑车"    # r - Rook
    BLACK_CANNON = "黑炮"  # c - Cannon
    BLACK_PAWN = "黑卒"    # p - Pawn
    
    EMPTY = "空"           # Empty position
    
    @property
    def color(self) -> str:
        """获取棋子颜色"""
        if self.name.startswith('RED'):
            return "red"
        elif self.name.startswith('BLACK'):
            return "black"
        return "none"
    
    @property
    def piece_char(self) -> str:
        """获取FEN记法字符"""
        char_map = {
            PieceType.RED_KING: 'K', PieceType.RED_ADVISOR: 'A', PieceType.RED_BISHOP: 'B',
            PieceType.RED_KNIGHT: 'N', PieceType.RED_ROOK: 'R', PieceType.RED_CANNON: 'C',
            PieceType.RED_PAWN: 'P', PieceType.BLACK_KING: 'k', PieceType.BLACK_ADVISOR: 'a', 
            PieceType.BLACK_BISHOP: 'b', PieceType.BLACK_KNIGHT: 'n', PieceType.BLACK_ROOK: 'r',
            PieceType.BLACK_CANNON: 'c', PieceType.BLACK_PAWN: 'p', PieceType.EMPTY: '.'
        }
        return char_map.get(self, '.')
    
    @classmethod
    def from_char(cls, char: str) -> 'PieceType':
        """从FEN字符创建棋子类型"""
        char_map = {
            'K': cls.RED_KING, 'A': cls.RED_ADVISOR, 'B': cls.RED_BISHOP,
            'N': cls.RED_KNIGHT, 'R': cls.RED_ROOK, 'C': cls.RED_CANNON,
            'P': cls.RED_PAWN, 'k': cls.BLACK_KING, 'a': cls.BLACK_ADVISOR,
            'b': cls.BLACK_BISHOP, 'n': cls.BLACK_KNIGHT, 'r': cls.BLACK_ROOK,
            'c': cls.BLACK_CANNON, 'p': cls.BLACK_PAWN, '.': cls.EMPTY
        }
        return char_map.get(char, cls.EMPTY)


@dataclass
class PieceDetection:
    """棋子检测结果"""
    piece_type: PieceType
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    grid_position: Optional[Tuple[int, int]] = None  # (row, col)
    center_point: Optional[Tuple[int, int]] = None   # (x, y)
    features: Optional[Dict[str, Any]] = None        # 特征信息
    
    def __post_init__(self):
        if self.center_point is None and self.bbox:
            x, y, w, h = self.bbox
            self.center_point = (x + w // 2, y + h // 2)


@dataclass
class ClassificationResult:
    """分类结果"""
    detections: List[PieceDetection]
    detection_time: float
    method_used: str
    total_pieces: int
    confidence_distribution: Dict[str, int]
    
    def get_pieces_by_type(self, piece_type: PieceType) -> List[PieceDetection]:
        """获取指定类型的棋子"""
        return [d for d in self.detections if d.piece_type == piece_type]
    
    def get_average_confidence(self) -> float:
        """获取平均置信度"""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


class RecognitionMethod(Enum):
    """识别方法枚举"""
    YOLO_ONLY = "yolo_only"                    # 仅使用YOLO
    TEMPLATE_MATCHING = "template_matching"     # 模板匹配
    FEATURE_MATCHING = "feature_matching"       # 特征匹配  
    HYBRID = "hybrid"                          # 混合方法
    TRADITIONAL_CV = "traditional_cv"          # 传统计算机视觉


@dataclass
class ClassifierStats:
    """分类器统计信息"""
    total_classifications: int = 0
    successful_classifications: int = 0
    average_confidence: float = 0.0
    average_classification_time: float = 0.0
    method_usage: Dict[str, int] = None
    piece_count_distribution: Dict[str, int] = None
    error_count: int = 0
    last_classification_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.method_usage is None:
            self.method_usage = {}
        if self.piece_count_distribution is None:
            self.piece_count_distribution = {}


class ChessPieceClassifier:
    """
    中国象棋棋子识别分类器
    
    主要功能：
    1. 基于YOLOv8的深度学习棋子检测
    2. 传统计算机视觉方法作为补充
    3. 14种棋子类型的准确识别
    4. 置信度评估和验证机制
    5. 多线程并行处理提升性能
    """
    
    # 棋子类型映射
    PIECE_CLASSES = [pt for pt in PieceType if pt != PieceType.EMPTY]
    
    # 置信度阈值
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.3
    
    def __init__(self, config: Any = None):
        """
        初始化棋子分类器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.stats = ClassifierStats()
        
        # 模型路径配置
        self._setup_model_paths()
        
        # 识别参数配置
        self._setup_recognition_params()
        
        # 初始化模型
        self.yolo_model: Optional[YOLO] = None
        self.template_images: Dict[PieceType, np.ndarray] = {}
        self.feature_descriptors: Dict[PieceType, List[np.ndarray]] = {}
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
        logger.info("棋子分类器初始化完成")
    
    def _setup_model_paths(self) -> None:
        """设置模型路径"""
        # YOLOv8模型路径
        self.yolo_model_path = "models/chess_pieces_yolo.pt"
        self.custom_model_path = "models/custom_chess_model.pt"
        
        # 模板图像目录
        self.template_dir = Path("data/piece_templates")
        
        # 特征数据目录
        self.features_dir = Path("data/piece_features")
        
        # 如果有配置，使用配置中的路径
        if hasattr(self.config, 'vision_config'):
            vision_config = self.config.vision_config
            self.yolo_model_path = getattr(vision_config, 'yolo_model_path', self.yolo_model_path)
            self.template_dir = Path(getattr(vision_config, 'template_directory', self.template_dir))
    
    def _setup_recognition_params(self) -> None:
        """设置识别参数"""
        # 置信度阈值
        self.confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD
        self.nms_threshold = 0.4  # 非最大抑制阈值
        
        # 图像预处理参数
        self.input_size = (640, 640)  # YOLO输入尺寸
        self.normalize_brightness = True
        self.enhance_contrast = True
        
        # 模板匹配参数
        self.template_match_threshold = 0.7
        self.template_scales = [0.8, 1.0, 1.2]  # 多尺度匹配
        
        # 特征匹配参数
        self.feature_match_ratio = 0.75
        self.min_match_count = 10
        
        # 如果有配置，使用配置参数
        if hasattr(self.config, 'vision_config'):
            vision_config = self.config.vision_config
            self.confidence_threshold = getattr(vision_config, 'piece_confidence_threshold', self.confidence_threshold)
            self.template_match_threshold = getattr(vision_config, 'template_match_threshold', self.template_match_threshold)
    
    def initialize(self) -> bool:
        """
        初始化分类器
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("开始初始化棋子分类器...")
            
            # 初始化YOLO模型
            success = self._initialize_yolo_model()
            if not success:
                logger.warning("YOLO模型初始化失败，将使用传统方法")
            
            # 加载模板图像
            self._load_template_images()
            
            # 加载特征描述符
            self._load_feature_descriptors()
            
            # 验证初始化结果
            if not self._validate_initialization():
                logger.error("分类器初始化验证失败")
                return False
            
            logger.info("棋子分类器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"棋子分类器初始化失败: {e}", exc_info=True)
            return False
    
    def _initialize_yolo_model(self) -> bool:
        """初始化YOLO模型"""
        if not YOLO_AVAILABLE:
            logger.warning("Ultralytics YOLO未安装，无法使用YOLO模型")
            return False
        
        try:
            # 尝试加载自定义模型
            if Path(self.custom_model_path).exists():
                logger.info(f"加载自定义YOLO模型: {self.custom_model_path}")
                self.yolo_model = YOLO(self.custom_model_path)
            elif Path(self.yolo_model_path).exists():
                logger.info(f"加载预训练YOLO模型: {self.yolo_model_path}")
                self.yolo_model = YOLO(self.yolo_model_path)
            else:
                logger.info("未找到本地模型，使用预训练YOLOv8模型")
                self.yolo_model = YOLO('yolov8n.pt')  # 使用nano版本
            
            # 配置模型参数
            if self.yolo_model:
                # 设置置信度阈值
                self.yolo_model.overrides['conf'] = self.confidence_threshold
                self.yolo_model.overrides['iou'] = self.nms_threshold
                
                logger.info("YOLO模型初始化成功")
                return True
                
        except Exception as e:
            logger.error(f"YOLO模型初始化失败: {e}")
            self.yolo_model = None
            
        return False
    
    def _load_template_images(self) -> None:
        """加载棋子模板图像"""
        if not self.template_dir.exists():
            logger.warning(f"模板目录不存在: {self.template_dir}")
            return
        
        for piece_type in self.PIECE_CLASSES:
            template_path = self.template_dir / f"{piece_type.name.lower()}.png"
            if template_path.exists():
                try:
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        self.template_images[piece_type] = template
                        logger.debug(f"加载模板: {piece_type.value}")
                except Exception as e:
                    logger.warning(f"加载模板图像失败 {template_path}: {e}")
        
        logger.info(f"已加载 {len(self.template_images)} 个棋子模板")
    
    def _load_feature_descriptors(self) -> None:
        """加载特征描述符"""
        if not self.features_dir.exists():
            logger.warning(f"特征目录不存在: {self.features_dir}")
            return
        
        for piece_type in self.PIECE_CLASSES:
            feature_path = self.features_dir / f"{piece_type.name.lower()}_features.json"
            if feature_path.exists():
                try:
                    with open(feature_path, 'r') as f:
                        feature_data = json.load(f)
                    
                    # 转换为numpy数组
                    descriptors = [np.array(desc) for desc in feature_data.get('descriptors', [])]
                    if descriptors:
                        self.feature_descriptors[piece_type] = descriptors
                        logger.debug(f"加载特征: {piece_type.value} ({len(descriptors)}个)")
                        
                except Exception as e:
                    logger.warning(f"加载特征描述符失败 {feature_path}: {e}")
        
        logger.info(f"已加载 {len(self.feature_descriptors)} 种棋子特征")
    
    def _validate_initialization(self) -> bool:
        """验证初始化结果"""
        # 至少要有一种识别方法可用
        has_yolo = self.yolo_model is not None
        has_templates = len(self.template_images) > 0
        has_features = len(self.feature_descriptors) > 0
        
        if not (has_yolo or has_templates or has_features):
            logger.error("没有可用的识别方法")
            return False
        
        # 记录可用方法
        methods = []
        if has_yolo:
            methods.append("YOLO深度学习")
        if has_templates:
            methods.append(f"模板匹配({len(self.template_images)}个)")
        if has_features:
            methods.append(f"特征匹配({len(self.feature_descriptors)}种)")
        
        logger.info(f"可用识别方法: {', '.join(methods)}")
        return True
    
    def classify_pieces(self, board_image: np.ndarray, 
                       board_region=None,
                       method: RecognitionMethod = RecognitionMethod.HYBRID) -> ClassificationResult:
        """
        识别棋盘上的所有棋子
        
        Args:
            board_image: 棋盘图像
            board_region: 棋盘区域信息
            method: 识别方法
            
        Returns:
            分类结果
        """
        start_time = datetime.now()
        self.stats.total_classifications += 1
        
        try:
            # 预处理图像
            processed_image = self._preprocess_image(board_image)
            
            # 根据方法选择识别算法
            detections = []
            method_used = method.value
            
            if method == RecognitionMethod.YOLO_ONLY:
                detections = self._classify_with_yolo(processed_image)
            elif method == RecognitionMethod.TEMPLATE_MATCHING:
                detections = self._classify_with_templates(processed_image)
            elif method == RecognitionMethod.FEATURE_MATCHING:
                detections = self._classify_with_features(processed_image)
            elif method == RecognitionMethod.TRADITIONAL_CV:
                detections = self._classify_with_traditional_cv(processed_image)
            else:  # HYBRID
                detections = self._classify_with_hybrid_method(processed_image)
                method_used = "hybrid"
            
            # 后处理检测结果
            detections = self._post_process_detections(detections, board_region)
            
            # 计算统计信息
            detection_time = (datetime.now() - start_time).total_seconds()
            confidence_dist = self._calculate_confidence_distribution(detections)
            
            # 创建结果
            result = ClassificationResult(
                detections=detections,
                detection_time=detection_time,
                method_used=method_used,
                total_pieces=len([d for d in detections if d.piece_type != PieceType.EMPTY]),
                confidence_distribution=confidence_dist
            )
            
            # 更新统计信息
            self._update_stats(result)
            
            logger.info(f"棋子识别完成: 检测到{result.total_pieces}个棋子，耗时{detection_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"棋子识别失败: {e}", exc_info=True)
            return ClassificationResult(
                detections=[],
                detection_time=(datetime.now() - start_time).total_seconds(),
                method_used=method.value,
                total_pieces=0,
                confidence_distribution={}
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        processed = image.copy()
        
        # 亮度归一化
        if self.normalize_brightness:
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=10)
        
        # 对比度增强
        if self.enhance_contrast:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def _classify_with_yolo(self, image: np.ndarray) -> List[PieceDetection]:
        """使用YOLO进行分类"""
        if self.yolo_model is None:
            return []
        
        try:
            # YOLO预测
            results = self.yolo_model(image, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 解析检测结果
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 转换为棋子类型
                        if cls < len(self.PIECE_CLASSES):
                            piece_type = self.PIECE_CLASSES[cls]
                            
                            detection = PieceDetection(
                                piece_type=piece_type,
                                confidence=float(conf),
                                bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1))
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO分类失败: {e}")
            return []
    
    def _classify_with_templates(self, image: np.ndarray) -> List[PieceDetection]:
        """使用模板匹配进行分类"""
        detections = []
        
        if not self.template_images:
            return detections
        
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for piece_type, template in self.template_images.items():
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # 多尺度模板匹配
                for scale in self.template_scales:
                    # 缩放模板
                    scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale)
                    h, w = scaled_template.shape
                    
                    # 模板匹配
                    result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                    
                    # 找到匹配位置
                    locations = np.where(result >= self.template_match_threshold)
                    
                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]
                        
                        detection = PieceDetection(
                            piece_type=piece_type,
                            confidence=float(confidence),
                            bbox=(pt[0], pt[1], w, h)
                        )
                        detections.append(detection)
            
            # 非极大值抑制去除重复检测
            detections = self._apply_nms(detections)
            
        except Exception as e:
            logger.error(f"模板匹配失败: {e}")
        
        return detections
    
    def _classify_with_features(self, image: np.ndarray) -> List[PieceDetection]:
        """使用特征匹配进行分类"""
        detections = []
        
        if not self.feature_descriptors:
            return detections
        
        try:
            # 提取图像特征
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(image, None)
            
            if descriptors is None:
                return detections
            
            # 特征匹配器
            matcher = cv2.FlannBasedMatcher()
            
            for piece_type, piece_descriptors in self.feature_descriptors.items():
                for piece_desc in piece_descriptors:
                    try:
                        # 特征匹配
                        matches = matcher.knnMatch(piece_desc, descriptors, k=2)
                        
                        # Lowe's ratio test
                        good_matches = []
                        for m, n in matches:
                            if m.distance < self.feature_match_ratio * n.distance:
                                good_matches.append(m)
                        
                        # 如果有足够的匹配点
                        if len(good_matches) >= self.min_match_count:
                            # 计算置信度
                            confidence = len(good_matches) / len(matches)
                            
                            # 估算边界框(简化实现)
                            matched_keypoints = [keypoints[m.trainIdx] for m in good_matches]
                            if matched_keypoints:
                                x_coords = [kp.pt[0] for kp in matched_keypoints]
                                y_coords = [kp.pt[1] for kp in matched_keypoints]
                                
                                x1, x2 = int(min(x_coords)), int(max(x_coords))
                                y1, y2 = int(min(y_coords)), int(max(y_coords))
                                
                                detection = PieceDetection(
                                    piece_type=piece_type,
                                    confidence=confidence,
                                    bbox=(x1, y1, x2-x1, y2-y1)
                                )
                                detections.append(detection)
                                
                    except Exception as e:
                        logger.debug(f"特征匹配异常: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"特征匹配失败: {e}")
        
        return detections
    
    def _classify_with_traditional_cv(self, image: np.ndarray) -> List[PieceDetection]:
        """使用传统计算机视觉方法进行分类"""
        detections = []
        
        try:
            # 轮廓检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # 过滤小轮廓
                if area < 100:
                    continue
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 提取ROI进行分析
                roi = image[y:y+h, x:x+w]
                
                # 简化的特征分析(颜色直方图)
                piece_type, confidence = self._analyze_piece_features(roi)
                
                if confidence > self.LOW_CONFIDENCE_THRESHOLD:
                    detection = PieceDetection(
                        piece_type=piece_type,
                        confidence=confidence,
                        bbox=(x, y, w, h)
                    )
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"传统CV方法失败: {e}")
        
        return detections
    
    def _classify_with_hybrid_method(self, image: np.ndarray) -> List[PieceDetection]:
        """使用混合方法进行分类"""
        all_detections = []
        
        # 并行执行多种方法
        futures = []
        
        if self.yolo_model:
            future = self.thread_pool.submit(self._classify_with_yolo, image)
            futures.append(('yolo', future))
        
        if self.template_images:
            future = self.thread_pool.submit(self._classify_with_templates, image)
            futures.append(('template', future))
        
        if self.feature_descriptors:
            future = self.thread_pool.submit(self._classify_with_features, image)
            futures.append(('feature', future))
        
        # 收集结果
        method_results = {}
        for method_name, future in futures:
            try:
                result = future.result(timeout=10)  # 10秒超时
                method_results[method_name] = result
                all_detections.extend(result)
            except Exception as e:
                logger.warning(f"{method_name}方法执行失败: {e}")
        
        # 融合结果
        fused_detections = self._fuse_detections(method_results)
        
        return fused_detections
    
    def _analyze_piece_features(self, roi: np.ndarray) -> Tuple[PieceType, float]:
        """分析棋子特征(简化实现)"""
        if roi.size == 0:
            return PieceType.EMPTY, 0.0
        
        # 计算颜色直方图
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 红色检测
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = red_mask1 + red_mask2
        
        # 黑色检测
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 30))
        
        red_ratio = cv2.countNonZero(red_mask) / roi.size
        black_ratio = cv2.countNonZero(black_mask) / roi.size
        
        # 简单的颜色分类
        if red_ratio > 0.1:
            # 红棋，需要进一步分析形状
            return PieceType.RED_PAWN, 0.6  # 简化处理
        elif black_ratio > 0.2:
            # 黑棋
            return PieceType.BLACK_PAWN, 0.6  # 简化处理
        
        return PieceType.EMPTY, 0.3
    
    def _fuse_detections(self, method_results: Dict[str, List[PieceDetection]]) -> List[PieceDetection]:
        """融合多种方法的检测结果"""
        if not method_results:
            return []
        
        # 简化的融合策略：优先使用YOLO结果，模板匹配作为补充
        final_detections = []
        
        # 如果有YOLO结果且质量好，优先使用
        if 'yolo' in method_results:
            yolo_detections = method_results['yolo']
            high_conf_yolo = [d for d in yolo_detections if d.confidence > self.HIGH_CONFIDENCE_THRESHOLD]
            
            if high_conf_yolo:
                final_detections.extend(high_conf_yolo)
                logger.debug(f"采用{len(high_conf_yolo)}个高置信度YOLO检测结果")
        
        # 如果YOLO结果不足，补充模板匹配结果
        if len(final_detections) < 16:  # 象棋标准32个子，至少应该有一些
            if 'template' in method_results:
                template_detections = method_results['template']
                # 去除与YOLO结果重叠的检测
                non_overlap = self._remove_overlapping_detections(template_detections, final_detections)
                final_detections.extend(non_overlap)
        
        # 应用非极大值抑制
        final_detections = self._apply_nms(final_detections)
        
        return final_detections
    
    def _remove_overlapping_detections(self, detections: List[PieceDetection], 
                                     existing: List[PieceDetection], 
                                     iou_threshold: float = 0.3) -> List[PieceDetection]:
        """移除重叠的检测结果"""
        non_overlapping = []
        
        for detection in detections:
            is_overlapping = False
            
            for existing_detection in existing:
                iou = self._calculate_iou(detection.bbox, existing_detection.bbox)
                if iou > iou_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                non_overlapping.append(detection)
        
        return non_overlapping
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """计算IoU(Intersection over Union)"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_nms(self, detections: List[PieceDetection]) -> List[PieceDetection]:
        """应用非极大值抑制"""
        if not detections:
            return detections
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # NMS算法
        keep = []
        
        while detections:
            # 保留置信度最高的
            best = detections.pop(0)
            keep.append(best)
            
            # 移除与最佳检测重叠度高的其他检测
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(best.bbox, detection.bbox)
                if iou <= self.nms_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return keep
    
    def _post_process_detections(self, detections: List[PieceDetection], 
                               board_region=None) -> List[PieceDetection]:
        """后处理检测结果"""
        processed = []
        
        for detection in detections:
            # 置信度过滤
            if detection.confidence < self.confidence_threshold:
                continue
            
            # 如果有棋盘区域信息，计算网格位置
            if board_region and hasattr(board_region, 'grid_points'):
                grid_pos = self._calculate_grid_position(detection, board_region)
                detection.grid_position = grid_pos
            
            # 添加特征信息
            detection.features = {
                'detection_method': 'hybrid',
                'processing_time': datetime.now().isoformat()
            }
            
            processed.append(detection)
        
        return processed
    
    def _calculate_grid_position(self, detection: PieceDetection, board_region) -> Optional[Tuple[int, int]]:
        """计算检测结果在棋盘网格中的位置"""
        if not detection.center_point:
            return None
        
        try:
            # 这里需要与棋盘检测器配合，将像素坐标转换为网格坐标
            # 简化实现，实际需要调用棋盘检测器的方法
            x, y = detection.center_point
            
            # 假设9x10网格，根据相对位置计算
            # 这是简化计算，实际需要更精确的映射
            if hasattr(board_region, 'width') and hasattr(board_region, 'height'):
                col = int((x / board_region.width) * 9)
                row = int((y / board_region.height) * 10)
                
                # 确保在有效范围内
                col = max(0, min(8, col))
                row = max(0, min(9, row))
                
                return (row, col)
                
        except Exception as e:
            logger.debug(f"网格位置计算失败: {e}")
        
        return None
    
    def _calculate_confidence_distribution(self, detections: List[PieceDetection]) -> Dict[str, int]:
        """计算置信度分布"""
        distribution = {
            "high": 0,      # > 0.8
            "medium": 0,    # 0.5-0.8
            "low": 0        # < 0.5
        }
        
        for detection in detections:
            if detection.confidence > 0.8:
                distribution["high"] += 1
            elif detection.confidence > 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _update_stats(self, result: ClassificationResult) -> None:
        """更新统计信息"""
        with self._lock:
            if result.detections:
                self.stats.successful_classifications += 1
            
            # 更新平均置信度
            avg_conf = result.get_average_confidence()
            if self.stats.average_confidence == 0:
                self.stats.average_confidence = avg_conf
            else:
                self.stats.average_confidence = (self.stats.average_confidence + avg_conf) / 2
            
            # 更新平均时间
            if self.stats.average_classification_time == 0:
                self.stats.average_classification_time = result.detection_time
            else:
                self.stats.average_classification_time = (self.stats.average_classification_time + result.detection_time) / 2
            
            # 更新方法使用统计
            method = result.method_used
            if method not in self.stats.method_usage:
                self.stats.method_usage[method] = 0
            self.stats.method_usage[method] += 1
            
            # 更新棋子数量分布
            piece_count = str(result.total_pieces)
            if piece_count not in self.stats.piece_count_distribution:
                self.stats.piece_count_distribution[piece_count] = 0
            self.stats.piece_count_distribution[piece_count] += 1
            
            self.stats.last_classification_time = datetime.now()
    
    def get_classification_stats(self) -> ClassifierStats:
        """获取分类器统计信息"""
        return self.stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = ClassifierStats()
        logger.info("棋子分类器统计信息已重置")
    
    def set_confidence_threshold(self, threshold: float) -> bool:
        """设置置信度阈值"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            if self.yolo_model:
                self.yolo_model.overrides['conf'] = threshold
            logger.info(f"置信度阈值已设置为: {threshold}")
            return True
        else:
            logger.error(f"无效的置信度阈值: {threshold}")
            return False
    
    def get_supported_piece_types(self) -> List[PieceType]:
        """获取支持的棋子类型"""
        return self.PIECE_CLASSES.copy()
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return (self.yolo_model is not None or 
                len(self.template_images) > 0 or 
                len(self.feature_descriptors) > 0)
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 关闭线程池
            self.thread_pool.shutdown(wait=True)
            
            # 清理模型
            if self.yolo_model:
                del self.yolo_model
                self.yolo_model = None
            
            # 清理缓存
            self.template_images.clear()
            self.feature_descriptors.clear()
            
            logger.info("棋子分类器资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()
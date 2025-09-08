#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋盘检测器 - 基于OpenCV的中国象棋棋盘识别
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from chess_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BoardCorners:
    """棋盘四角坐标"""
    top_left: Tuple[int, int]
    top_right: Tuple[int, int] 
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组用于透视变换"""
        return np.array([
            self.top_left,
            self.top_right, 
            self.bottom_right,
            self.bottom_left
        ], dtype=np.float32)


@dataclass
class BoardRegion:
    """棋盘区域信息"""
    corners: BoardCorners
    width: int
    height: int
    transform_matrix: Optional[np.ndarray] = None
    confidence: float = 0.0
    grid_points: Optional[np.ndarray] = None
    
    def get_board_image(self, source_image: np.ndarray) -> np.ndarray:
        """提取标准化的棋盘图像"""
        if self.transform_matrix is None:
            raise ValueError("未设置变换矩阵")
        return cv2.warpPerspective(source_image, self.transform_matrix, 
                                 (self.width, self.height))


class DetectionMethod(Enum):
    """检测方法枚举"""
    HOUGH_LINES = "hough_lines"
    CONTOUR_DETECTION = "contour_detection" 
    TEMPLATE_MATCHING = "template_matching"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


@dataclass
class DetectionStats:
    """检测统计信息"""
    detections_attempted: int = 0
    detections_successful: int = 0
    average_detection_time: float = 0.0
    last_detection_time: Optional[datetime] = None
    average_confidence: float = 0.0
    method_stats: Dict[str, int] = None
    
    def __post_init__(self):
        if self.method_stats is None:
            self.method_stats = {}


class ChessBoardDetector:
    """
    中国象棋棋盘检测器
    
    主要功能：
    1. 从图像中检测棋盘区域
    2. 执行透视变换获得标准棋盘图像
    3. 生成9x10网格坐标映射
    4. 支持多种检测算法和参数调优
    """
    
    # 中国象棋棋盘标准尺寸比例
    BOARD_ASPECT_RATIO = 8.0 / 9.0  # 宽/高比例
    STANDARD_BOARD_WIDTH = 720      # 标准棋盘宽度
    STANDARD_BOARD_HEIGHT = 810     # 标准棋盘高度
    
    # 网格配置
    GRID_COLS = 9  # 9条竖线
    GRID_ROWS = 10  # 10条横线
    
    def __init__(self, config: Any = None):
        """
        初始化棋盘检测器
        
        Args:
            config: 配置对象，包含检测参数
        """
        self.config = config
        self.stats = DetectionStats()
        self.last_detected_board: Optional[BoardRegion] = None
        
        # 检测参数配置
        self._setup_detection_params()
        
        # 初始化检测算法
        self._initialize_detectors()
        
        logger.info("棋盘检测器初始化完成")
    
    def _setup_detection_params(self) -> None:
        """设置检测参数"""
        # Canny边缘检测参数
        self.canny_low = 50
        self.canny_high = 150
        self.canny_aperture = 3
        
        # Hough直线检测参数  
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 100
        self.hough_min_line_length = 50
        self.hough_max_line_gap = 10
        
        # 轮廓检测参数
        self.contour_area_threshold = 1000
        self.contour_arc_length_ratio = 0.02
        
        # 检测置信度阈值
        self.confidence_threshold = 0.7
        
        # 如果有配置对象，使用配置值覆盖默认参数
        if hasattr(self.config, 'vision_config'):
            vision_config = self.config.vision_config
            self.canny_low = getattr(vision_config, 'canny_low_threshold', self.canny_low)
            self.canny_high = getattr(vision_config, 'canny_high_threshold', self.canny_high)
            self.confidence_threshold = getattr(vision_config, 'detection_confidence_threshold', self.confidence_threshold)
    
    def _initialize_detectors(self) -> None:
        """初始化检测算法组件"""
        # 创建形态学操作核
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 创建线检测器
        self.line_detector = cv2.createLineSegmentDetector()
        
        logger.info("检测算法组件初始化完成")
    
    def detect_board(self, image: np.ndarray, method: DetectionMethod = DetectionMethod.HOUGH_LINES) -> Optional[BoardRegion]:
        """
        检测棋盘区域
        
        Args:
            image: 输入图像
            method: 检测方法
            
        Returns:
            BoardRegion: 检测到的棋盘区域，失败返回None
        """
        start_time = datetime.now()
        self.stats.detections_attempted += 1
        
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 根据方法选择检测算法
            board_region = None
            if method == DetectionMethod.HOUGH_LINES:
                board_region = self._detect_by_hough_lines(processed_image, image)
            elif method == DetectionMethod.CONTOUR_DETECTION:
                board_region = self._detect_by_contours(processed_image, image)
            elif method == DetectionMethod.ADAPTIVE_THRESHOLD:
                board_region = self._detect_by_adaptive_threshold(processed_image, image)
            else:
                logger.warning(f"不支持的检测方法: {method}")
                return None
            
            # 验证检测结果
            if board_region and self._validate_board_region(board_region, image.shape[:2]):
                # 生成网格坐标
                board_region.grid_points = self._generate_grid_points(board_region)
                
                # 更新统计信息
                self.stats.detections_successful += 1
                self.last_detected_board = board_region
                
                detection_time = (datetime.now() - start_time).total_seconds()
                self.stats.last_detection_time = datetime.now()
                self._update_detection_stats(detection_time, board_region.confidence, method)
                
                logger.info(f"棋盘检测成功，置信度: {board_region.confidence:.3f}")
                return board_region
            else:
                logger.warning("棋盘检测失败或结果验证不通过")
                return None
                
        except Exception as e:
            logger.error(f"棋盘检测异常: {e}", exc_info=True)
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 原始图像
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 直方图均衡化增强对比度
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def _detect_by_hough_lines(self, processed_image: np.ndarray, original_image: np.ndarray) -> Optional[BoardRegion]:
        """
        使用Hough直线检测棋盘
        
        Args:
            processed_image: 预处理后的图像
            original_image: 原始图像
            
        Returns:
            检测到的棋盘区域
        """
        # Canny边缘检测
        edges = cv2.Canny(processed_image, self.canny_low, self.canny_high, 
                         apertureSize=self.canny_aperture)
        
        # Hough直线检测
        lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta, 
                               self.hough_threshold, minLineLength=self.hough_min_line_length,
                               maxLineGap=self.hough_max_line_gap)
        
        if lines is None or len(lines) < 4:
            logger.debug("Hough检测到的直线数量不足")
            return None
        
        # 分离水平线和垂直线
        horizontal_lines, vertical_lines = self._classify_lines(lines)
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.debug("水平线或垂直线数量不足")
            return None
        
        # 寻找棋盘边界
        corners = self._find_board_corners(horizontal_lines, vertical_lines)
        if corners is None:
            return None
        
        # 计算置信度
        confidence = self._calculate_hough_confidence(lines, corners)
        
        # 创建透视变换矩阵
        transform_matrix = self._create_transform_matrix(corners)
        
        return BoardRegion(
            corners=corners,
            width=self.STANDARD_BOARD_WIDTH,
            height=self.STANDARD_BOARD_HEIGHT,
            transform_matrix=transform_matrix,
            confidence=confidence
        )
    
    def _detect_by_contours(self, processed_image: np.ndarray, original_image: np.ndarray) -> Optional[BoardRegion]:
        """
        使用轮廓检测棋盘
        
        Args:
            processed_image: 预处理后的图像
            original_image: 原始图像
            
        Returns:
            检测到的棋盘区域
        """
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选合适的轮廓
        candidate_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.contour_area_threshold:
                # 多边形逼近
                epsilon = self.contour_arc_length_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 寻找四边形
                if len(approx) == 4:
                    candidate_contours.append((contour, approx, area))
        
        if not candidate_contours:
            logger.debug("未找到合适的四边形轮廓")
            return None
        
        # 选择面积最大的四边形作为棋盘
        best_contour = max(candidate_contours, key=lambda x: x[2])
        approx_points = best_contour[1].reshape(4, 2)
        
        # 排序角点
        corners = self._sort_corners(approx_points)
        if corners is None:
            return None
        
        # 计算置信度
        confidence = self._calculate_contour_confidence(best_contour[0], original_image.shape[:2])
        
        # 创建透视变换矩阵
        transform_matrix = self._create_transform_matrix(corners)
        
        return BoardRegion(
            corners=corners,
            width=self.STANDARD_BOARD_WIDTH,
            height=self.STANDARD_BOARD_HEIGHT,
            transform_matrix=transform_matrix,
            confidence=confidence
        )
    
    def _detect_by_adaptive_threshold(self, processed_image: np.ndarray, original_image: np.ndarray) -> Optional[BoardRegion]:
        """
        使用自适应阈值检测棋盘
        
        Args:
            processed_image: 预处理后的图像
            original_image: 原始图像
            
        Returns:
            检测到的棋盘区域
        """
        # 多种自适应阈值方法
        methods = [
            (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 3),
        ]
        
        for method, block_size, c_param in methods:
            binary = cv2.adaptiveThreshold(processed_image, 255, method, 
                                         cv2.THRESH_BINARY, block_size, c_param)
            
            # 尝试轮廓检测
            result = self._detect_by_contours(binary, original_image)
            if result and result.confidence > self.confidence_threshold:
                return result
        
        return None
    
    def _classify_lines(self, lines: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        分类水平线和垂直线
        
        Args:
            lines: 检测到的直线数组
            
        Returns:
            (水平线列表, 垂直线列表)
        """
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算直线角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 分类直线 (允许一定角度偏差)
            if angle < 30 or angle > 150:  # 水平线
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:  # 垂直线
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def _find_board_corners(self, horizontal_lines: List[np.ndarray], 
                          vertical_lines: List[np.ndarray]) -> Optional[BoardCorners]:
        """
        从直线中寻找棋盘四角
        
        Args:
            horizontal_lines: 水平线列表
            vertical_lines: 垂直线列表
            
        Returns:
            棋盘四角坐标
        """
        # 找到最极端的直线
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # 按y坐标排序找上下边界
        h_lines_by_y = sorted(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)
        top_line = h_lines_by_y[0]
        bottom_line = h_lines_by_y[-1]
        
        # 按x坐标排序找左右边界
        v_lines_by_x = sorted(vertical_lines, key=lambda line: (line[0] + line[2]) / 2)
        left_line = v_lines_by_x[0]
        right_line = v_lines_by_x[-1]
        
        # 计算交点
        corners = self._calculate_line_intersections(top_line, bottom_line, left_line, right_line)
        
        return corners
    
    def _calculate_line_intersections(self, top_line: np.ndarray, bottom_line: np.ndarray,
                                    left_line: np.ndarray, right_line: np.ndarray) -> Optional[BoardCorners]:
        """
        计算直线交点得到棋盘四角
        
        Args:
            top_line: 上边界直线
            bottom_line: 下边界直线  
            left_line: 左边界直线
            right_line: 右边界直线
            
        Returns:
            棋盘四角坐标
        """
        def line_intersection(line1, line2):
            """计算两直线交点"""
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            
            return (int(x), int(y))
        
        # 计算四个交点
        top_left = line_intersection(top_line, left_line)
        top_right = line_intersection(top_line, right_line)
        bottom_left = line_intersection(bottom_line, left_line)
        bottom_right = line_intersection(bottom_line, right_line)
        
        if not all([top_left, top_right, bottom_left, bottom_right]):
            return None
        
        return BoardCorners(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right
        )
    
    def _sort_corners(self, points: np.ndarray) -> Optional[BoardCorners]:
        """
        排序角点为标准顺序（左上、右上、左下、右下）
        
        Args:
            points: 四个角点坐标
            
        Returns:
            排序后的角点
        """
        if len(points) != 4:
            return None
        
        # 计算中心点
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        # 分类角点
        top_left = top_right = bottom_left = bottom_right = None
        
        for point in points:
            x, y = point
            if x < center_x and y < center_y:
                top_left = (int(x), int(y))
            elif x > center_x and y < center_y:
                top_right = (int(x), int(y))
            elif x < center_x and y > center_y:
                bottom_left = (int(x), int(y))
            elif x > center_x and y > center_y:
                bottom_right = (int(x), int(y))
        
        if not all([top_left, top_right, bottom_left, bottom_right]):
            return None
        
        return BoardCorners(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right
        )
    
    def _create_transform_matrix(self, corners: BoardCorners) -> np.ndarray:
        """
        创建透视变换矩阵
        
        Args:
            corners: 棋盘四角坐标
            
        Returns:
            透视变换矩阵
        """
        # 源坐标（检测到的四角）
        src_points = corners.to_array()
        
        # 目标坐标（标准矩形）
        dst_points = np.array([
            [0, 0],
            [self.STANDARD_BOARD_WIDTH, 0],
            [self.STANDARD_BOARD_WIDTH, self.STANDARD_BOARD_HEIGHT],
            [0, self.STANDARD_BOARD_HEIGHT]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return transform_matrix
    
    def _generate_grid_points(self, board_region: BoardRegion) -> np.ndarray:
        """
        生成9x10网格交点坐标
        
        Args:
            board_region: 棋盘区域
            
        Returns:
            网格交点坐标数组 shape: (10, 9, 2)
        """
        grid_points = np.zeros((self.GRID_ROWS, self.GRID_COLS, 2), dtype=np.float32)
        
        # 计算网格间距
        col_step = self.STANDARD_BOARD_WIDTH / (self.GRID_COLS - 1)
        row_step = self.STANDARD_BOARD_HEIGHT / (self.GRID_ROWS - 1)
        
        # 生成标准网格坐标
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                x = col * col_step
                y = row * row_step
                grid_points[row, col] = [x, y]
        
        return grid_points
    
    def _validate_board_region(self, board_region: BoardRegion, image_shape: Tuple[int, int]) -> bool:
        """
        验证检测到的棋盘区域是否合理
        
        Args:
            board_region: 棋盘区域
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            验证结果
        """
        corners = board_region.corners
        img_height, img_width = image_shape
        
        # 检查角点是否在图像范围内
        points = [corners.top_left, corners.top_right, corners.bottom_left, corners.bottom_right]
        for x, y in points:
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                logger.debug(f"角点超出图像范围: ({x}, {y})")
                return False
        
        # 检查棋盘长宽比
        width = abs(corners.top_right[0] - corners.top_left[0])
        height = abs(corners.bottom_left[1] - corners.top_left[1])
        
        if width == 0 or height == 0:
            return False
        
        aspect_ratio = width / height
        expected_ratio = self.BOARD_ASPECT_RATIO
        
        # 允许20%的比例偏差
        if abs(aspect_ratio - expected_ratio) > expected_ratio * 0.2:
            logger.debug(f"棋盘长宽比不符合: {aspect_ratio:.3f} vs {expected_ratio:.3f}")
            return False
        
        # 检查置信度
        if board_region.confidence < self.confidence_threshold:
            logger.debug(f"置信度过低: {board_region.confidence:.3f}")
            return False
        
        return True
    
    def _calculate_hough_confidence(self, lines: np.ndarray, corners: BoardCorners) -> float:
        """
        计算Hough检测的置信度
        
        Args:
            lines: 检测到的直线
            corners: 棋盘角点
            
        Returns:
            置信度分数 (0-1)
        """
        if lines is None or len(lines) == 0:
            return 0.0
        
        # 基于直线数量的置信度
        line_count_score = min(len(lines) / 20, 1.0)
        
        # 基于角点规整性的置信度
        geometry_score = self._calculate_geometry_score(corners)
        
        # 综合置信度
        confidence = (line_count_score * 0.3 + geometry_score * 0.7)
        
        return min(confidence, 1.0)
    
    def _calculate_contour_confidence(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """
        计算轮廓检测的置信度
        
        Args:
            contour: 检测到的轮廓
            image_shape: 图像尺寸
            
        Returns:
            置信度分数 (0-1)
        """
        # 基于面积的置信度
        area = cv2.contourArea(contour)
        total_area = image_shape[0] * image_shape[1]
        area_ratio = area / total_area
        area_score = min(area_ratio * 10, 1.0)  # 假设棋盘占图像10%以上
        
        # 基于周长的置信度
        perimeter = cv2.arcLength(contour, True)
        expected_perimeter = 2 * (image_shape[0] + image_shape[1]) * 0.3  # 估算周长
        perimeter_score = 1.0 - abs(perimeter - expected_perimeter) / expected_perimeter
        perimeter_score = max(perimeter_score, 0.0)
        
        # 综合置信度
        confidence = (area_score * 0.6 + perimeter_score * 0.4)
        
        return min(confidence, 1.0)
    
    def _calculate_geometry_score(self, corners: BoardCorners) -> float:
        """
        计算几何形状得分
        
        Args:
            corners: 棋盘角点
            
        Returns:
            几何得分 (0-1)
        """
        # 计算四边长度
        top_length = np.linalg.norm(np.array(corners.top_right) - np.array(corners.top_left))
        bottom_length = np.linalg.norm(np.array(corners.bottom_right) - np.array(corners.bottom_left))
        left_length = np.linalg.norm(np.array(corners.bottom_left) - np.array(corners.top_left))
        right_length = np.linalg.norm(np.array(corners.bottom_right) - np.array(corners.top_right))
        
        # 检查对边长度相似性
        horizontal_similarity = 1.0 - abs(top_length - bottom_length) / max(top_length, bottom_length)
        vertical_similarity = 1.0 - abs(left_length - right_length) / max(left_length, right_length)
        
        # 检查长宽比
        width = (top_length + bottom_length) / 2
        height = (left_length + right_length) / 2
        aspect_ratio = width / height if height > 0 else 0
        expected_ratio = self.BOARD_ASPECT_RATIO
        ratio_score = 1.0 - abs(aspect_ratio - expected_ratio) / expected_ratio
        ratio_score = max(ratio_score, 0.0)
        
        # 综合得分
        geometry_score = (horizontal_similarity * 0.3 + vertical_similarity * 0.3 + ratio_score * 0.4)
        
        return min(geometry_score, 1.0)
    
    def _update_detection_stats(self, detection_time: float, confidence: float, method: DetectionMethod) -> None:
        """
        更新检测统计信息
        
        Args:
            detection_time: 检测耗时
            confidence: 检测置信度
            method: 检测方法
        """
        # 更新平均检测时间
        if self.stats.average_detection_time == 0:
            self.stats.average_detection_time = detection_time
        else:
            self.stats.average_detection_time = (self.stats.average_detection_time + detection_time) / 2
        
        # 更新平均置信度
        if self.stats.average_confidence == 0:
            self.stats.average_confidence = confidence
        else:
            self.stats.average_confidence = (self.stats.average_confidence + confidence) / 2
        
        # 更新方法统计
        method_name = method.value
        if method_name not in self.stats.method_stats:
            self.stats.method_stats[method_name] = 0
        self.stats.method_stats[method_name] += 1
    
    def get_grid_position(self, board_region: BoardRegion, pixel_coords: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        将像素坐标转换为棋盘网格坐标
        
        Args:
            board_region: 棋盘区域
            pixel_coords: 像素坐标
            
        Returns:
            网格坐标 (row, col)，范围: row[0-9], col[0-8]
        """
        if board_region.grid_points is None:
            return None
        
        x, y = pixel_coords
        
        # 转换到标准棋盘坐标系
        if board_region.transform_matrix is not None:
            # 应用逆变换
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), 
                                                 np.linalg.inv(board_region.transform_matrix))
            x, y = transformed[0, 0]
        
        # 寻找最近的网格点
        min_distance = float('inf')
        closest_grid = None
        
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                grid_x, grid_y = board_region.grid_points[row, col]
                distance = np.sqrt((x - grid_x)**2 + (y - grid_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_grid = (row, col)
        
        # 如果距离太远，认为不在网格上
        if min_distance > 30:  # 阈值可调
            return None
        
        return closest_grid
    
    def get_pixel_coords(self, board_region: BoardRegion, grid_coords: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        将网格坐标转换为像素坐标
        
        Args:
            board_region: 棋盘区域
            grid_coords: 网格坐标 (row, col)
            
        Returns:
            像素坐标 (x, y)
        """
        if board_region.grid_points is None:
            return None
        
        row, col = grid_coords
        
        # 检查坐标范围
        if not (0 <= row < self.GRID_ROWS and 0 <= col < self.GRID_COLS):
            return None
        
        # 获取标准网格坐标
        grid_x, grid_y = board_region.grid_points[row, col]
        
        # 如果有变换矩阵，转换到原图坐标系
        if board_region.transform_matrix is not None:
            point = np.array([[grid_x, grid_y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), 
                                                 np.linalg.inv(board_region.transform_matrix))
            x, y = transformed[0, 0]
            return (int(x), int(y))
        
        return (int(grid_x), int(grid_y))
    
    def get_detection_stats(self) -> DetectionStats:
        """
        获取检测统计信息
        
        Returns:
            统计信息
        """
        return self.stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = DetectionStats()
        logger.info("棋盘检测器统计信息已重置")
    
    def initialize(self) -> bool:
        """
        初始化检测器
        
        Returns:
            初始化是否成功
        """
        try:
            # 重新设置参数
            self._setup_detection_params()
            
            # 重新初始化检测器
            self._initialize_detectors()
            
            # 重置统计
            self.reset_stats()
            
            logger.info("棋盘检测器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"棋盘检测器初始化失败: {e}", exc_info=True)
            return False
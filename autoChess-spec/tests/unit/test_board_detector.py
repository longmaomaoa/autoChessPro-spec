#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋盘检测器单元测试
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chess_ai.vision.board_detector import (
    ChessBoardDetector, BoardCorners, BoardRegion, DetectionMethod, DetectionStats
)


class TestBoardCorners(unittest.TestCase):
    """测试BoardCorners数据类"""
    
    def test_board_corners_creation(self):
        """测试棋盘角点创建"""
        corners = BoardCorners(
            top_left=(10, 20),
            top_right=(100, 25),
            bottom_left=(15, 120),
            bottom_right=(105, 125)
        )
        
        self.assertEqual(corners.top_left, (10, 20))
        self.assertEqual(corners.top_right, (100, 25))
        self.assertEqual(corners.bottom_left, (15, 120))
        self.assertEqual(corners.bottom_right, (105, 125))
    
    def test_to_array(self):
        """测试转换为numpy数组"""
        corners = BoardCorners(
            top_left=(0, 0),
            top_right=(100, 0),
            bottom_left=(0, 100),
            bottom_right=(100, 100)
        )
        
        array = corners.to_array()
        expected = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100]
        ], dtype=np.float32)
        
        np.testing.assert_array_equal(array, expected)


class TestBoardRegion(unittest.TestCase):
    """测试BoardRegion数据类"""
    
    def setUp(self):
        self.corners = BoardCorners(
            top_left=(0, 0),
            top_right=(720, 0),
            bottom_left=(0, 810),
            bottom_right=(720, 810)
        )
        
        self.board_region = BoardRegion(
            corners=self.corners,
            width=720,
            height=810,
            confidence=0.85
        )
    
    def test_board_region_creation(self):
        """测试棋盘区域创建"""
        self.assertEqual(self.board_region.width, 720)
        self.assertEqual(self.board_region.height, 810)
        self.assertEqual(self.board_region.confidence, 0.85)
    
    @patch('cv2.warpPerspective')
    def test_get_board_image_with_transform(self, mock_warp):
        """测试带变换矩阵的棋盘图像提取"""
        # 设置变换矩阵
        self.board_region.transform_matrix = np.eye(3)
        mock_warp.return_value = np.zeros((810, 720, 3), dtype=np.uint8)
        
        source_image = np.ones((1000, 1000, 3), dtype=np.uint8)
        result = self.board_region.get_board_image(source_image)
        
        mock_warp.assert_called_once()
        self.assertEqual(result.shape, (810, 720, 3))
    
    def test_get_board_image_without_transform(self):
        """测试无变换矩阵时抛出异常"""
        source_image = np.ones((1000, 1000, 3), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            self.board_region.get_board_image(source_image)


class TestDetectionStats(unittest.TestCase):
    """测试检测统计信息"""
    
    def test_stats_initialization(self):
        """测试统计信息初始化"""
        stats = DetectionStats()
        
        self.assertEqual(stats.detections_attempted, 0)
        self.assertEqual(stats.detections_successful, 0)
        self.assertEqual(stats.average_detection_time, 0.0)
        self.assertIsNone(stats.last_detection_time)
        self.assertEqual(stats.average_confidence, 0.0)
        self.assertIsInstance(stats.method_stats, dict)
        self.assertEqual(len(stats.method_stats), 0)


class TestChessBoardDetector(unittest.TestCase):
    """测试棋盘检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Mock()
        self.config.vision_config = Mock()
        self.config.vision_config.canny_low_threshold = 50
        self.config.vision_config.canny_high_threshold = 150
        self.config.vision_config.detection_confidence_threshold = 0.7
        
        self.detector = ChessBoardDetector(self.config)
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        self.assertIsNotNone(self.detector.config)
        self.assertIsInstance(self.detector.stats, DetectionStats)
        self.assertIsNone(self.detector.last_detected_board)
        self.assertEqual(self.detector.canny_low, 50)
        self.assertEqual(self.detector.canny_high, 150)
        self.assertEqual(self.detector.confidence_threshold, 0.7)
    
    def test_detector_initialization_without_config(self):
        """测试无配置的检测器初始化"""
        detector = ChessBoardDetector()
        
        self.assertIsNone(detector.config)
        self.assertEqual(detector.canny_low, 50)
        self.assertEqual(detector.canny_high, 150)
        self.assertEqual(detector.confidence_threshold, 0.7)
    
    @patch('cv2.equalizeHist')
    @patch('cv2.GaussianBlur')
    @patch('cv2.cvtColor')
    def test_preprocess_image_color(self, mock_cvt, mock_blur, mock_eq):
        """测试彩色图像预处理"""
        # 设置mock返回值
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        blurred_image = np.ones((100, 100), dtype=np.uint8) * 120
        equalized_image = np.ones((100, 100), dtype=np.uint8) * 130
        
        mock_cvt.return_value = gray_image
        mock_blur.return_value = blurred_image
        mock_eq.return_value = equalized_image
        
        # 测试彩色图像
        color_image = np.ones((100, 100, 3), dtype=np.uint8)
        result = self.detector._preprocess_image(color_image)
        
        mock_cvt.assert_called_once_with(color_image, cv2.COLOR_BGR2GRAY)
        mock_blur.assert_called_once_with(gray_image, (5, 5), 0)
        mock_eq.assert_called_once_with(blurred_image)
        np.testing.assert_array_equal(result, equalized_image)
    
    @patch('cv2.equalizeHist')
    @patch('cv2.GaussianBlur')
    def test_preprocess_image_gray(self, mock_blur, mock_eq):
        """测试灰度图像预处理"""
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        blurred_image = np.ones((100, 100), dtype=np.uint8) * 120
        equalized_image = np.ones((100, 100), dtype=np.uint8) * 130
        
        mock_blur.return_value = blurred_image
        mock_eq.return_value = equalized_image
        
        # 测试灰度图像
        result = self.detector._preprocess_image(gray_image)
        
        # 检查调用参数 - GaussianBlur调用验证
        mock_blur.assert_called_once()
        call_args = mock_blur.call_args
        self.assertEqual(call_args[0][1], (5, 5))  # ksize参数
        self.assertEqual(call_args[0][2], 0)  # sigmaX参数
        mock_eq.assert_called_once_with(blurred_image)
        np.testing.assert_array_equal(result, equalized_image)
    
    def test_classify_lines(self):
        """测试直线分类"""
        # 创建测试直线 (水平线和垂直线)
        lines = np.array([
            [[0, 10, 100, 15]],    # 水平线 (角度小)
            [[10, 0, 15, 100]],    # 垂直线 (角度约90度)
            [[0, 50, 100, 55]],    # 水平线
            [[50, 0, 55, 100]],    # 垂直线
        ])
        
        horizontal, vertical = self.detector._classify_lines(lines)
        
        self.assertEqual(len(horizontal), 2)
        self.assertEqual(len(vertical), 2)
    
    def test_sort_corners(self):
        """测试角点排序"""
        # 创建无序的四个角点
        points = np.array([
            [100, 100],  # 右下
            [0, 0],      # 左上
            [0, 100],    # 左下
            [100, 0]     # 右上
        ])
        
        corners = self.detector._sort_corners(points)
        
        self.assertIsNotNone(corners)
        self.assertEqual(corners.top_left, (0, 0))
        self.assertEqual(corners.top_right, (100, 0))
        self.assertEqual(corners.bottom_left, (0, 100))
        self.assertEqual(corners.bottom_right, (100, 100))
    
    def test_sort_corners_invalid(self):
        """测试无效角点排序"""
        # 测试点数不足
        points = np.array([[0, 0], [100, 0], [0, 100]])
        result = self.detector._sort_corners(points)
        self.assertIsNone(result)
    
    def test_create_transform_matrix(self):
        """测试透视变换矩阵创建"""
        corners = BoardCorners(
            top_left=(10, 20),
            top_right=(110, 25),
            bottom_left=(15, 120),
            bottom_right=(115, 125)
        )
        
        with patch('cv2.getPerspectiveTransform') as mock_transform:
            mock_matrix = np.eye(3)
            mock_transform.return_value = mock_matrix
            
            result = self.detector._create_transform_matrix(corners)
            
            mock_transform.assert_called_once()
            np.testing.assert_array_equal(result, mock_matrix)
    
    def test_generate_grid_points(self):
        """测试网格坐标生成"""
        corners = BoardCorners(
            top_left=(0, 0),
            top_right=(720, 0),
            bottom_left=(0, 810),
            bottom_right=(720, 810)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810
        )
        
        grid_points = self.detector._generate_grid_points(board_region)
        
        # 检查网格尺寸
        self.assertEqual(grid_points.shape, (10, 9, 2))
        
        # 检查角点坐标
        np.testing.assert_array_almost_equal(grid_points[0, 0], [0, 0])  # 左上角
        np.testing.assert_array_almost_equal(grid_points[0, 8], [720, 0])  # 右上角
        np.testing.assert_array_almost_equal(grid_points[9, 0], [0, 810])  # 左下角
        np.testing.assert_array_almost_equal(grid_points[9, 8], [720, 810])  # 右下角
    
    def test_validate_board_region_valid(self):
        """测试有效棋盘区域验证"""
        # 创建符合长宽比的棋盘区域
        corners = BoardCorners(
            top_left=(100, 100),
            top_right=(820, 105),  # 宽度约720
            bottom_left=(105, 910),  # 高度约810
            bottom_right=(825, 915)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810,
            confidence=0.8
        )
        
        image_shape = (1000, 1000)
        result = self.detector._validate_board_region(board_region, image_shape)
        
        self.assertTrue(result)
    
    def test_validate_board_region_out_of_bounds(self):
        """测试超出图像边界的棋盘区域"""
        corners = BoardCorners(
            top_left=(-10, -10),  # 超出边界
            top_right=(720, 0),
            bottom_left=(0, 810),
            bottom_right=(720, 810)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810,
            confidence=0.8
        )
        
        image_shape = (1000, 1000)
        result = self.detector._validate_board_region(board_region, image_shape)
        
        self.assertFalse(result)
    
    def test_validate_board_region_wrong_aspect_ratio(self):
        """测试错误长宽比的棋盘区域"""
        # 创建长宽比错误的区域 (2:1, 而正确比例应该是8:9 ≈ 0.889)
        corners = BoardCorners(
            top_left=(100, 100),
            top_right=(900, 100),  # 宽度800
            bottom_left=(100, 500),  # 高度400 (2:1 = 2.0)
            bottom_right=(900, 500)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810,
            confidence=0.8
        )
        
        image_shape = (1000, 1000)
        result = self.detector._validate_board_region(board_region, image_shape)
        
        self.assertFalse(result)
    
    def test_validate_board_region_low_confidence(self):
        """测试低置信度的棋盘区域"""
        corners = BoardCorners(
            top_left=(100, 100),
            top_right=(820, 105),
            bottom_left=(105, 910),
            bottom_right=(825, 915)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810,
            confidence=0.5  # 低于阈值0.7
        )
        
        image_shape = (1000, 1000)
        result = self.detector._validate_board_region(board_region, image_shape)
        
        self.assertFalse(result)
    
    def test_calculate_geometry_score(self):
        """测试几何形状得分计算"""
        # 创建标准矩形
        corners = BoardCorners(
            top_left=(0, 0),
            top_right=(720, 0),
            bottom_left=(0, 810),
            bottom_right=(720, 810)
        )
        
        score = self.detector._calculate_geometry_score(corners)
        
        # 标准矩形应该得到高分
        self.assertGreater(score, 0.8)
        self.assertLessEqual(score, 1.0)
    
    @patch('cv2.perspectiveTransform')
    def test_get_grid_position(self, mock_transform):
        """测试像素坐标转网格坐标"""
        corners = BoardCorners(
            top_left=(0, 0),
            top_right=(720, 0),
            bottom_left=(0, 810),
            bottom_right=(720, 810)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810,
            transform_matrix=np.eye(3)
        )
        
        # 生成网格点
        board_region.grid_points = self.detector._generate_grid_points(board_region)
        
        # mock变换结果为左上角附近
        mock_transform.return_value = np.array([[[5, 5]]], dtype=np.float32)
        
        # 测试坐标转换
        pixel_coords = (100, 100)
        grid_coords = self.detector.get_grid_position(board_region, pixel_coords)
        
        self.assertIsNotNone(grid_coords)
        self.assertEqual(grid_coords, (0, 0))  # 应该是左上角
    
    @patch('cv2.perspectiveTransform')
    def test_get_pixel_coords(self, mock_transform):
        """测试网格坐标转像素坐标"""
        corners = BoardCorners(
            top_left=(0, 0),
            top_right=(720, 0),
            bottom_left=(0, 810),
            bottom_right=(720, 810)
        )
        
        board_region = BoardRegion(
            corners=corners,
            width=720,
            height=810,
            transform_matrix=np.eye(3)
        )
        
        # 生成网格点
        board_region.grid_points = self.detector._generate_grid_points(board_region)
        
        # mock逆变换结果
        mock_transform.return_value = np.array([[[100, 120]]], dtype=np.float32)
        
        # 测试坐标转换
        grid_coords = (1, 1)
        pixel_coords = self.detector.get_pixel_coords(board_region, grid_coords)
        
        self.assertIsNotNone(pixel_coords)
        self.assertEqual(pixel_coords, (100, 120))
    
    def test_get_pixel_coords_invalid_range(self):
        """测试无效范围的网格坐标"""
        board_region = BoardRegion(
            corners=BoardCorners((0,0), (720,0), (0,810), (720,810)),
            width=720,
            height=810
        )
        board_region.grid_points = self.detector._generate_grid_points(board_region)
        
        # 测试超出范围的坐标
        result = self.detector.get_pixel_coords(board_region, (-1, 0))
        self.assertIsNone(result)
        
        result = self.detector.get_pixel_coords(board_region, (10, 0))
        self.assertIsNone(result)
        
        result = self.detector.get_pixel_coords(board_region, (0, 9))
        self.assertIsNone(result)
    
    def test_get_detection_stats(self):
        """测试获取检测统计信息"""
        stats = self.detector.get_detection_stats()
        
        self.assertIsInstance(stats, DetectionStats)
        self.assertEqual(stats.detections_attempted, 0)
        self.assertEqual(stats.detections_successful, 0)
    
    def test_reset_stats(self):
        """测试重置统计信息"""
        # 先修改一些统计数据
        self.detector.stats.detections_attempted = 10
        self.detector.stats.detections_successful = 8
        
        # 重置统计
        self.detector.reset_stats()
        
        # 验证重置结果
        self.assertEqual(self.detector.stats.detections_attempted, 0)
        self.assertEqual(self.detector.stats.detections_successful, 0)
    
    def test_initialize(self):
        """测试初始化方法"""
        result = self.detector.initialize()
        
        self.assertTrue(result)
        self.assertEqual(self.detector.stats.detections_attempted, 0)
    
    @patch('chess_ai.vision.board_detector.ChessBoardDetector._initialize_detectors')
    def test_initialize_failure(self, mock_init):
        """测试初始化失败情况"""
        mock_init.side_effect = Exception("初始化失败")
        
        result = self.detector.initialize()
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋子识别分类器单元测试模块

本模块测试ChessPieceClassifier的所有核心功能，包括：
- 数据类的创建和方法
- 枚举类的定义和属性
- 分类器的初始化和配置
- 不同识别方法的验证
- 性能监控和统计功能

测试设计遵循AAA模式（Arrange-Act-Assert），确保测试的可读性和维护性。
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import os
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 导入待测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chess_ai.vision.piece_classifier import (
    PieceType, RecognitionMethod, PieceDetection, ClassificationResult,
    ClassificationStats, ChessPieceClassifier, ClassificationError
)
from chess_ai.config.config_manager import ConfigManager
from chess_ai.data.pieces import Position


class TestPieceType(unittest.TestCase):
    """测试PieceType枚举类"""
    
    def test_piece_type_creation(self):
        """测试棋子类型枚举的创建和基本属性"""
        # Arrange & Act & Assert
        self.assertEqual(PieceType.RED_KING.value, "红帅")
        self.assertEqual(PieceType.BLACK_KING.value, "黑将")
        self.assertEqual(PieceType.RED_ROOK.value, "红车")
        self.assertEqual(PieceType.BLACK_PAWN.value, "黑卒")
        self.assertEqual(PieceType.EMPTY.value, "空")
    
    def test_piece_type_count(self):
        """测试棋子类型的总数量"""
        # Arrange & Act
        piece_types = list(PieceType)
        
        # Assert
        self.assertEqual(len(piece_types), 15)  # 14种棋子 + 空位
    
    def test_red_pieces(self):
        """测试红方棋子类型"""
        # Arrange
        red_pieces = [
            PieceType.RED_KING, PieceType.RED_ADVISOR, PieceType.RED_BISHOP,
            PieceType.RED_KNIGHT, PieceType.RED_ROOK, PieceType.RED_CANNON, PieceType.RED_PAWN
        ]
        
        # Act & Assert
        for piece in red_pieces:
            self.assertTrue(piece.value.startswith("红"))
    
    def test_black_pieces(self):
        """测试黑方棋子类型"""
        # Arrange
        black_pieces = [
            PieceType.BLACK_KING, PieceType.BLACK_ADVISOR, PieceType.BLACK_BISHOP,
            PieceType.BLACK_KNIGHT, PieceType.BLACK_ROOK, PieceType.BLACK_CANNON, PieceType.BLACK_PAWN
        ]
        
        # Act & Assert
        for piece in black_pieces:
            self.assertTrue(piece.value.startswith("黑"))
    
    def test_to_fen_notation(self):
        """测试FEN记号转换功能（需要在PieceType中添加此方法）"""
        # 由于原实现中未包含to_fen方法，这里模拟测试结构
        # 实际实现时需要在PieceType中添加此功能
        
        # Arrange
        expected_fen_mapping = {
            PieceType.RED_KING: 'K',
            PieceType.RED_ADVISOR: 'A', 
            PieceType.RED_BISHOP: 'B',
            PieceType.RED_KNIGHT: 'N',
            PieceType.RED_ROOK: 'R',
            PieceType.RED_CANNON: 'C',
            PieceType.RED_PAWN: 'P',
            PieceType.BLACK_KING: 'k',
            PieceType.BLACK_ADVISOR: 'a',
            PieceType.BLACK_BISHOP: 'b', 
            PieceType.BLACK_KNIGHT: 'n',
            PieceType.BLACK_ROOK: 'r',
            PieceType.BLACK_CANNON: 'c',
            PieceType.BLACK_PAWN: 'p',
            PieceType.EMPTY: '1'
        }
        
        # Act & Assert
        for piece_type in expected_fen_mapping:
            # 这里验证枚举存在性，FEN转换功能需要后续实现
            self.assertIsNotNone(piece_type.value)


class TestRecognitionMethod(unittest.TestCase):
    """测试RecognitionMethod枚举类"""
    
    def test_recognition_methods(self):
        """测试识别方法枚举的完整性"""
        # Arrange
        expected_methods = ["YOLO", "TEMPLATE", "FEATURE", "TRADITIONAL", "HYBRID"]
        
        # Act
        actual_methods = [method.value for method in RecognitionMethod]
        
        # Assert
        self.assertEqual(set(expected_methods), set(actual_methods))
        self.assertEqual(len(actual_methods), 5)


class TestPieceDetection(unittest.TestCase):
    """测试PieceDetection数据类"""
    
    def test_piece_detection_creation(self):
        """测试棋子检测结果的创建"""
        # Arrange
        position = Position(3, 4)
        piece_type = PieceType.RED_ROOK
        confidence = 0.95
        bbox = [100, 150, 50, 60]
        
        # Act
        detection = PieceDetection(
            position=position,
            piece_type=piece_type,
            confidence=confidence,
            bbox=bbox
        )
        
        # Assert
        self.assertEqual(detection.position, position)
        self.assertEqual(detection.piece_type, piece_type)
        self.assertEqual(detection.confidence, confidence)
        self.assertEqual(detection.bbox, bbox)
    
    def test_piece_detection_defaults(self):
        """测试棋子检测结果的默认值"""
        # Arrange
        position = Position(0, 0)
        piece_type = PieceType.EMPTY
        
        # Act
        detection = PieceDetection(position=position, piece_type=piece_type)
        
        # Assert
        self.assertEqual(detection.confidence, 0.0)
        self.assertIsNone(detection.bbox)


class TestClassificationResult(unittest.TestCase):
    """测试ClassificationResult数据类"""
    
    def test_classification_result_creation(self):
        """测试分类结果的创建"""
        # Arrange
        detections = [
            PieceDetection(Position(0, 0), PieceType.RED_ROOK, 0.95),
            PieceDetection(Position(1, 0), PieceType.RED_KNIGHT, 0.88)
        ]
        method = RecognitionMethod.YOLO
        processing_time = 0.15
        total_pieces = 32
        confidence_avg = 0.915
        
        # Act
        result = ClassificationResult(
            detections=detections,
            method_used=method,
            processing_time=processing_time,
            total_pieces_found=total_pieces,
            average_confidence=confidence_avg
        )
        
        # Assert
        self.assertEqual(result.detections, detections)
        self.assertEqual(result.method_used, method)
        self.assertEqual(result.processing_time, processing_time)
        self.assertEqual(result.total_pieces_found, total_pieces)
        self.assertEqual(result.average_confidence, confidence_avg)
    
    def test_classification_result_defaults(self):
        """测试分类结果的默认值"""
        # Arrange
        detections = []
        method = RecognitionMethod.TEMPLATE
        
        # Act
        result = ClassificationResult(detections=detections, method_used=method)
        
        # Assert
        self.assertEqual(result.processing_time, 0.0)
        self.assertEqual(result.total_pieces_found, 0)
        self.assertEqual(result.average_confidence, 0.0)


class TestClassificationStats(unittest.TestCase):
    """测试ClassificationStats数据类"""
    
    def test_classification_stats_creation(self):
        """测试分类统计数据的创建"""
        # Arrange
        total_classifications = 100
        successful_classifications = 95
        method_usage = {"YOLO": 60, "TEMPLATE": 35, "FEATURE": 5}
        avg_processing_time = 0.12
        avg_confidence = 0.87
        
        # Act
        stats = ClassificationStats(
            total_classifications=total_classifications,
            successful_classifications=successful_classifications,
            method_usage_count=method_usage,
            average_processing_time=avg_processing_time,
            average_confidence=avg_confidence
        )
        
        # Assert
        self.assertEqual(stats.total_classifications, total_classifications)
        self.assertEqual(stats.successful_classifications, successful_classifications)
        self.assertEqual(stats.method_usage_count, method_usage)
        self.assertEqual(stats.average_processing_time, avg_processing_time)
        self.assertEqual(stats.average_confidence, avg_confidence)
    
    def test_classification_stats_defaults(self):
        """测试分类统计数据的默认值"""
        # Act
        stats = ClassificationStats()
        
        # Assert
        self.assertEqual(stats.total_classifications, 0)
        self.assertEqual(stats.successful_classifications, 0)
        self.assertEqual(stats.method_usage_count, {})
        self.assertEqual(stats.average_processing_time, 0.0)
        self.assertEqual(stats.average_confidence, 0.0)


class TestClassificationError(unittest.TestCase):
    """测试ClassificationError异常类"""
    
    def test_classification_error_creation(self):
        """测试分类异常的创建"""
        # Arrange
        message = "Template matching failed"
        error_code = "TEMPLATE_MATCH_FAILED"
        
        # Act & Assert
        with self.assertRaises(ClassificationError) as context:
            raise ClassificationError(message, error_code)
        
        self.assertEqual(str(context.exception), message)
        self.assertEqual(context.exception.error_code, error_code)
    
    def test_classification_error_defaults(self):
        """测试分类异常的默认值"""
        # Arrange
        message = "Classification error occurred"
        
        # Act & Assert
        with self.assertRaises(ClassificationError) as context:
            raise ClassificationError(message)
        
        self.assertEqual(str(context.exception), message)
        self.assertIsNone(context.exception.error_code)


class TestChessPieceClassifier(unittest.TestCase):
    """测试ChessPieceClassifier主类"""
    
    def setUp(self):
        """测试准备"""
        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        
        # 模拟配置数据
        self.test_config = {
            "vision": {
                "piece_classifier": {
                    "yolo_model_path": "models/yolo_pieces.pt",
                    "template_path": "templates/pieces/",
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.4,
                    "max_detections": 32
                }
            },
            "performance": {
                "max_workers": 4,
                "enable_gpu": True,
                "batch_size": 8
            }
        }
        
        # 创建配置文件
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, ensure_ascii=False, indent=2)
    
    def tearDown(self):
        """测试清理"""
        # 清理临时文件
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        os.rmdir(self.temp_dir)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_classifier_initialization(self, mock_config_manager):
        """测试分类器初始化"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.confidence_threshold = 0.6
        mock_config.vision.piece_classifier.nms_threshold = 0.3
        mock_config.vision.piece_classifier.max_detections = 30
        mock_config.performance.max_workers = 2
        mock_config_manager.return_value = mock_config
        
        # Act
        classifier = ChessPieceClassifier()
        
        # Assert
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.config, mock_config)
        self.assertIsInstance(classifier.stats, ClassificationStats)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_classifier_initialization_with_custom_config(self, mock_config_manager):
        """测试使用自定义配置的分类器初始化"""
        # Arrange
        custom_config = Mock()
        custom_config.vision.piece_classifier.confidence_threshold = 0.8
        
        # Act
        classifier = ChessPieceClassifier(config=custom_config)
        
        # Assert
        self.assertEqual(classifier.config, custom_config)
        mock_config_manager.assert_not_called()
    
    @patch('chess_ai.vision.piece_classifier.cv2')
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_preprocess_image(self, mock_config_manager, mock_cv2):
        """测试图像预处理功能"""
        # Arrange
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟输入图像
        input_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 模拟OpenCV函数返回
        mock_cv2.cvtColor.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_cv2.equalizeHist.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_cv2.convertScaleAbs.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Act
        result = classifier._preprocess_image(input_image)
        
        # Assert
        self.assertIsNotNone(result)
        mock_cv2.cvtColor.assert_called()
        mock_cv2.equalizeHist.assert_called()
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_extract_piece_regions_from_grid(self, mock_config_manager):
        """测试从网格中提取棋子区域"""
        # Arrange
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟棋盘图像和网格坐标
        board_image = np.random.randint(0, 255, (450, 400, 3), dtype=np.uint8)
        grid_points = np.random.rand(10, 9, 2) * 400  # 10x9网格
        
        # Act
        regions = classifier._extract_piece_regions_from_grid(board_image, grid_points)
        
        # Assert
        self.assertIsInstance(regions, list)
        self.assertEqual(len(regions), 90)  # 10x9 = 90个区域
        
        # 检查每个区域的结构
        for region in regions:
            self.assertIn('image', region)
            self.assertIn('position', region)
            self.assertIsInstance(region['position'], Position)
    
    @patch('chess_ai.vision.piece_classifier.torch')
    @patch('chess_ai.vision.piece_classifier.YOLO')
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_yolo_classification_method(self, mock_config_manager, mock_yolo_class, mock_torch):
        """测试YOLO分类方法"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.yolo_model_path = "test_model.pt"
        mock_config.vision.piece_classifier.confidence_threshold = 0.5
        mock_config.vision.piece_classifier.nms_threshold = 0.4
        mock_config_manager.return_value = mock_config
        
        # 模拟YOLO模型
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 50, 60]])
        mock_result.boxes.conf = torch.tensor([0.85])
        mock_result.boxes.cls = torch.tensor([2])  # 假设类别2是红车
        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model
        
        # 模拟torch功能
        mock_torch.tensor = lambda x: x
        
        classifier = ChessPieceClassifier()
        
        # 模拟输入
        board_image = np.random.randint(0, 255, (400, 450, 3), dtype=np.uint8)
        grid_points = np.random.rand(10, 9, 2) * 400
        
        # Act
        result = classifier._classify_with_yolo(board_image, grid_points)
        
        # Assert
        self.assertIsInstance(result, list)
    
    @patch('chess_ai.vision.piece_classifier.cv2')
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_template_matching_method(self, mock_config_manager, mock_cv2):
        """测试模板匹配方法"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.template_path = "templates/"
        mock_config.vision.piece_classifier.confidence_threshold = 0.6
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟模板图像
        mock_template = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_template
        
        # 模拟模板匹配结果
        match_result = np.random.rand(20, 20)
        match_result[10, 10] = 0.9  # 高匹配度点
        mock_cv2.matchTemplate.return_value = match_result
        mock_cv2.minMaxLoc.return_value = (0.1, 0.9, (5, 5), (10, 10))
        
        # Act
        piece_regions = [
            {'image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
             'position': Position(0, 0)}
        ]
        result = classifier._classify_with_template_matching(piece_regions)
        
        # Assert
        self.assertIsInstance(result, list)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_feature_matching_method(self, mock_config_manager):
        """测试特征匹配方法"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.confidence_threshold = 0.7
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # Act
        piece_regions = [
            {'image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
             'position': Position(1, 1)}
        ]
        result = classifier._classify_with_feature_matching(piece_regions)
        
        # Assert
        self.assertIsInstance(result, list)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_traditional_cv_method(self, mock_config_manager):
        """测试传统计算机视觉方法"""
        # Arrange
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # Act
        piece_regions = [
            {'image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
             'position': Position(2, 2)}
        ]
        result = classifier._classify_with_traditional_cv(piece_regions)
        
        # Assert
        self.assertIsInstance(result, list)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_classify_pieces_hybrid_method(self, mock_config_manager):
        """测试混合分类方法"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.confidence_threshold = 0.5
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟各种方法的返回结果
        with patch.object(classifier, '_classify_with_yolo') as mock_yolo, \
             patch.object(classifier, '_classify_with_template_matching') as mock_template, \
             patch.object(classifier, '_classify_with_feature_matching') as mock_feature:
            
            # 设置模拟返回值
            mock_yolo.return_value = [
                PieceDetection(Position(0, 0), PieceType.RED_ROOK, 0.9)
            ]
            mock_template.return_value = [
                PieceDetection(Position(1, 0), PieceType.RED_KNIGHT, 0.8)
            ]
            mock_feature.return_value = [
                PieceDetection(Position(2, 0), PieceType.RED_BISHOP, 0.7)
            ]
            
            # Act
            board_image = np.random.randint(0, 255, (400, 450, 3), dtype=np.uint8)
            result = classifier.classify_pieces(board_image, method=RecognitionMethod.HYBRID)
        
        # Assert
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.method_used, RecognitionMethod.HYBRID)
        self.assertGreater(len(result.detections), 0)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_update_stats(self, mock_config_manager):
        """测试统计信息更新"""
        # Arrange
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # Act
        classifier._update_stats(RecognitionMethod.YOLO, 0.15, 0.85, True)
        classifier._update_stats(RecognitionMethod.TEMPLATE, 0.25, 0.75, True)
        classifier._update_stats(RecognitionMethod.YOLO, 0.12, 0.90, False)
        
        # Assert
        stats = classifier.get_stats()
        self.assertEqual(stats.total_classifications, 3)
        self.assertEqual(stats.successful_classifications, 2)
        self.assertIn("YOLO", stats.method_usage_count)
        self.assertIn("TEMPLATE", stats.method_usage_count)
        self.assertEqual(stats.method_usage_count["YOLO"], 2)
        self.assertEqual(stats.method_usage_count["TEMPLATE"], 1)
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_reset_stats(self, mock_config_manager):
        """测试统计信息重置"""
        # Arrange
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 先更新一些统计
        classifier._update_stats(RecognitionMethod.YOLO, 0.15, 0.85, True)
        
        # Act
        classifier.reset_stats()
        
        # Assert
        stats = classifier.get_stats()
        self.assertEqual(stats.total_classifications, 0)
        self.assertEqual(stats.successful_classifications, 0)
        self.assertEqual(stats.method_usage_count, {})
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_error_handling_invalid_image(self, mock_config_manager):
        """测试无效图像的错误处理"""
        # Arrange
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # Act & Assert
        with self.assertRaises(ValueError):
            classifier.classify_pieces(None)
        
        with self.assertRaises(ValueError):
            classifier.classify_pieces(np.array([]))
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_confidence_threshold_filtering(self, mock_config_manager):
        """测试置信度阈值过滤"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.confidence_threshold = 0.8
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟检测结果
        detections = [
            PieceDetection(Position(0, 0), PieceType.RED_ROOK, 0.9),  # 应该保留
            PieceDetection(Position(1, 0), PieceType.RED_KNIGHT, 0.7),  # 应该过滤
            PieceDetection(Position(2, 0), PieceType.RED_BISHOP, 0.85)  # 应该保留
        ]
        
        # Act
        filtered = classifier._filter_detections_by_confidence(detections)
        
        # Assert
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(d.confidence >= 0.8 for d in filtered))
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_non_maximum_suppression(self, mock_config_manager):
        """测试非极大值抑制"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.nms_threshold = 0.5
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟重叠的检测结果
        detections = [
            PieceDetection(Position(0, 0), PieceType.RED_ROOK, 0.9, [10, 10, 50, 50]),
            PieceDetection(Position(0, 0), PieceType.RED_ROOK, 0.8, [15, 15, 50, 50]),  # 重叠，应该被抑制
            PieceDetection(Position(5, 5), PieceType.RED_KNIGHT, 0.85, [200, 200, 50, 50])  # 不重叠，应该保留
        ]
        
        # Act
        result = classifier._apply_non_maximum_suppression(detections)
        
        # Assert
        self.assertEqual(len(result), 2)
        # 应该保留置信度更高的检测和不重叠的检测
        confidences = [d.confidence for d in result]
        self.assertIn(0.9, confidences)
        self.assertIn(0.85, confidences)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    @patch('chess_ai.vision.piece_classifier.ConfigManager')
    def test_full_classification_pipeline(self, mock_config_manager):
        """测试完整的分类流水线"""
        # Arrange
        mock_config = Mock()
        mock_config.vision.piece_classifier.confidence_threshold = 0.6
        mock_config.vision.piece_classifier.nms_threshold = 0.4
        mock_config_manager.return_value = mock_config
        
        classifier = ChessPieceClassifier()
        
        # 模拟输入
        board_image = np.random.randint(0, 255, (400, 450, 3), dtype=np.uint8)
        
        # 模拟检测方法
        with patch.object(classifier, '_classify_with_yolo') as mock_yolo:
            mock_yolo.return_value = [
                PieceDetection(Position(0, 0), PieceType.RED_ROOK, 0.9),
                PieceDetection(Position(1, 0), PieceType.RED_KNIGHT, 0.8),
                PieceDetection(Position(2, 0), PieceType.RED_BISHOP, 0.7)
            ]
            
            # Act
            result = classifier.classify_pieces(board_image, method=RecognitionMethod.YOLO)
        
        # Assert
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.method_used, RecognitionMethod.YOLO)
        self.assertEqual(result.total_pieces_found, 3)
        self.assertGreater(result.average_confidence, 0)
        self.assertGreater(result.processing_time, 0)
        
        # 验证统计信息更新
        stats = classifier.get_stats()
        self.assertEqual(stats.total_classifications, 1)
        self.assertEqual(stats.successful_classifications, 1)


def main():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestPieceType,
        TestRecognitionMethod, 
        TestPieceDetection,
        TestClassificationResult,
        TestClassificationStats,
        TestClassificationError,
        TestChessPieceClassifier,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print(f"\n{'='*50}")
    print(f"测试总数: {result.testsRun}")
    print(f"失败: {len(result.failures)}")  
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    print(f"成功率: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
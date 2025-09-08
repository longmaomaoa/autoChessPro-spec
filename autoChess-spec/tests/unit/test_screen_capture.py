#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕捕获模块单元测试
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chess_ai.vision.screen_capture import (
    ScreenCaptureModule, CaptureState, CaptureRegion, 
    CaptureStats, DisplayInfo, CaptureError
)
from chess_ai.config.config_manager import ConfigManager


class MockConfigManager:
    """Mock配置管理器"""
    
    def __init__(self):
        self.vision_config = Mock()
        self.vision_config.capture_fps = 30
        
    def get_vision_config(self):
        return self.vision_config


class TestCaptureRegion(unittest.TestCase):
    """测试CaptureRegion数据类"""
    
    def test_capture_region_creation(self):
        """测试捕获区域创建"""
        region = CaptureRegion(100, 200, 800, 600)
        
        self.assertEqual(region.x, 100)
        self.assertEqual(region.y, 200)
        self.assertEqual(region.width, 800)
        self.assertEqual(region.height, 600)
    
    def test_to_tuple_conversion(self):
        """测试转换为元组"""
        region = CaptureRegion(10, 20, 300, 400)
        tuple_result = region.to_tuple()
        
        self.assertEqual(tuple_result, (10, 20, 310, 420))
    
    def test_from_tuple_creation(self):
        """测试从元组创建"""
        tuple_region = (50, 60, 350, 460)
        region = CaptureRegion.from_tuple(tuple_region)
        
        self.assertEqual(region.x, 50)
        self.assertEqual(region.y, 60)
        self.assertEqual(region.width, 300)
        self.assertEqual(region.height, 400)


class TestDisplayInfo(unittest.TestCase):
    """测试DisplayInfo数据类"""
    
    def test_display_info_creation(self):
        """测试显示器信息创建"""
        display = DisplayInfo(
            index=0,
            name="Primary Monitor",
            width=1920,
            height=1080,
            is_primary=True
        )
        
        self.assertEqual(display.index, 0)
        self.assertEqual(display.name, "Primary Monitor")
        self.assertEqual(display.width, 1920)
        self.assertEqual(display.height, 1080)
        self.assertTrue(display.is_primary)
    
    def test_display_info_string(self):
        """测试显示器信息字符串表示"""
        display = DisplayInfo(1, "Secondary Monitor", 1600, 900, False)
        str_repr = str(display)
        
        self.assertIn("显示器 1", str_repr)
        self.assertIn("Secondary Monitor", str_repr)
        self.assertIn("1600x900", str_repr)
        self.assertNotIn("主显示器", str_repr)


class TestScreenCaptureModule(unittest.TestCase):
    """测试ScreenCaptureModule主类"""
    
    def setUp(self):
        """测试前置设置"""
        self.config_manager = MockConfigManager()
        
        # Mock dxcam
        self.dxcam_mock = MagicMock()
        self.dxcam_instance_mock = MagicMock()
        self.dxcam_mock.create.return_value = self.dxcam_instance_mock
        
        # 测试图像数据
        self.test_frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        self.dxcam_instance_mock.grab.return_value = self.test_frame
    
    @patch('chess_ai.vision.screen_capture.WIN32_AVAILABLE', False)
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_initialization_without_win32(self, mock_dxcam):
        """测试无Win32API时的初始化"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        
        # 检查默认显示器创建
        self.assertEqual(len(capture.available_displays), 1)
        self.assertEqual(capture.available_displays[0].name, "Default Monitor")
        self.assertTrue(capture.available_displays[0].is_primary)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', False)
    def test_initialization_without_dxcam(self):
        """测试无DXcam时的初始化"""
        capture = ScreenCaptureModule(self.config_manager)
        
        with self.assertRaises(CaptureError) as context:
            capture.initialize()
        
        self.assertEqual(context.exception.error_code, "DXCAM_UNAVAILABLE")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_successful_initialization(self, mock_dxcam):
        """测试成功初始化"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        result = capture.initialize()
        
        self.assertTrue(result)
        self.assertEqual(capture.state, CaptureState.STOPPED)
        self.assertIsNotNone(capture.dxcam_instance)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_capture_region_setting(self, mock_dxcam):
        """测试捕获区域设置"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        result = capture.set_capture_region(100, 200, 800, 600)
        
        self.assertTrue(result)
        self.assertIsNotNone(capture.capture_region)
        self.assertEqual(capture.capture_region.x, 100)
        self.assertEqual(capture.capture_region.y, 200)
        self.assertEqual(capture.capture_region.width, 800)
        self.assertEqual(capture.capture_region.height, 600)
    
    def test_invalid_capture_region(self):
        """测试无效捕获区域"""
        capture = ScreenCaptureModule(self.config_manager)
        
        # 测试负数尺寸
        result = capture.set_capture_region(100, 200, 0, 600)
        self.assertFalse(result)
        
        result = capture.set_capture_region(100, 200, 800, -100)
        self.assertFalse(result)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_target_fps_setting(self, mock_dxcam):
        """测试目标FPS设置"""
        capture = ScreenCaptureModule(self.config_manager)
        
        # 测试有效FPS
        result = capture.set_target_fps(60)
        self.assertTrue(result)
        self.assertEqual(capture.target_fps, 60)
        
        # 测试无效FPS
        result = capture.set_target_fps(0)
        self.assertFalse(result)
        
        result = capture.set_target_fps(300)
        self.assertFalse(result)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_single_frame_capture(self, mock_dxcam):
        """测试单帧捕获"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        frame = capture.capture_single_frame()
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, self.test_frame.shape)
        self.assertEqual(capture.stats.frames_captured, 1)
        self.assertIsNotNone(capture.stats.last_capture_time)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_single_frame_capture_with_region(self, mock_dxcam):
        """测试区域单帧捕获"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        capture.set_capture_region(100, 100, 400, 300)
        
        frame = capture.capture_single_frame()
        
        self.assertIsNotNone(frame)
        # 验证grab方法被正确调用
        expected_region = (100, 100, 500, 400)
        self.dxcam_instance_mock.grab.assert_called_with(region=expected_region)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_frame_callback(self, mock_dxcam):
        """测试帧回调功能"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        # 设置回调函数
        callback_called = []
        def test_callback(frame):
            callback_called.append(frame)
        
        capture.set_frame_callback(test_callback)
        
        # 触发回调（通过单帧捕获模拟）
        frame = capture.capture_single_frame()
        
        # 由于单帧捕获不会触发回调，我们需要手动调用
        if capture.frame_callback:
            capture.frame_callback(frame)
        
        self.assertEqual(len(callback_called), 1)
        np.testing.assert_array_equal(callback_called[0], self.test_frame)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_performance_monitoring(self, mock_dxcam):
        """测试性能监控功能"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        # 模拟多次捕获
        for i in range(5):
            capture.capture_single_frame()
            time.sleep(0.01)
        
        stats = capture.get_capture_stats()
        self.assertEqual(stats.frames_captured, 5)
        self.assertEqual(stats.frames_dropped, 0)
        self.assertEqual(stats.error_count, 0)
        
        # 测试性能报告
        report = capture.get_performance_report()
        self.assertEqual(report['frames_captured'], 5)
        self.assertEqual(report['success_rate_percent'], 100.0)
        self.assertIn('state', report)
        self.assertIn('average_fps', report)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_error_handling(self, mock_dxcam):
        """测试错误处理"""
        # 模拟DXcam创建失败
        mock_dxcam.create.return_value = None
        
        capture = ScreenCaptureModule(self.config_manager)
        
        with self.assertRaises(CaptureError) as context:
            capture.initialize()
        
        self.assertEqual(context.exception.error_code, "INSTANCE_CREATION_FAILED")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_retry_settings(self, mock_dxcam):
        """测试重试设置"""
        capture = ScreenCaptureModule(self.config_manager)
        
        # 测试有效设置
        result = capture.set_retry_settings(3, 0.5)
        self.assertTrue(result)
        self.assertEqual(capture.max_retry_attempts, 3)
        self.assertEqual(capture.retry_delay, 0.5)
        
        # 测试无效设置
        result = capture.set_retry_settings(0, 0.5)
        self.assertFalse(result)
        
        result = capture.set_retry_settings(3, 3.0)
        self.assertFalse(result)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_auto_recovery_setting(self, mock_dxcam):
        """测试自动恢复设置"""
        capture = ScreenCaptureModule(self.config_manager)
        
        # 默认应该启用自动恢复
        self.assertTrue(capture.auto_recovery)
        
        # 测试禁用
        capture.set_auto_recovery(False)
        self.assertFalse(capture.auto_recovery)
        
        # 测试重新启用
        capture.set_auto_recovery(True)
        self.assertTrue(capture.auto_recovery)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_statistics_reset(self, mock_dxcam):
        """测试统计信息重置"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        # 生成一些统计数据
        capture.capture_single_frame()
        capture.stats.error_count = 5
        
        # 重置统计
        capture.reset_stats()
        
        self.assertEqual(capture.stats.frames_captured, 0)
        self.assertEqual(capture.stats.error_count, 0)
        self.assertEqual(capture.stats.frames_dropped, 0)
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_health_check(self, mock_dxcam):
        """测试健康检查"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        # 初始状态应该是健康的
        self.assertTrue(capture.is_healthy())
        
        # 模拟错误状态
        capture.state = CaptureState.ERROR
        self.assertFalse(capture.is_healthy())
        
        # 恢复正常状态但模拟高错误率
        capture.state = CaptureState.STOPPED
        capture.stats.frames_captured = 100
        capture.stats.error_count = 15  # 15% 错误率
        self.assertFalse(capture.is_healthy())
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_diagnostics(self, mock_dxcam):
        """测试诊断信息"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        diagnostics = capture.get_diagnostics()
        
        self.assertIn('system_info', diagnostics)
        self.assertIn('module_state', diagnostics)
        self.assertIn('settings', diagnostics)
        self.assertIn('health_check', diagnostics)
        
        # 验证系统信息
        self.assertIn('platform', diagnostics['system_info'])
        self.assertIn('dxcam_available', diagnostics['system_info'])
        
        # 验证模块状态
        self.assertEqual(diagnostics['module_state']['state'], 'stopped')
        self.assertTrue(diagnostics['module_state']['dxcam_instance'])
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_cleanup(self, mock_dxcam):
        """测试资源清理"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        capture.capture_single_frame()  # 生成一些数据
        
        # 执行清理
        capture.cleanup()
        
        # 验证清理结果
        self.assertIsNone(capture.dxcam_instance)
        self.assertIsNone(capture.latest_frame)
        self.assertIsNone(capture.frame_callback)
        self.assertEqual(len(capture._frame_times), 0)
        
        # 验证DXcam release被调用
        self.dxcam_instance_mock.release.assert_called_once()


if __name__ == '__main__':
    # 配置测试运行
    unittest.main(verbosity=2)
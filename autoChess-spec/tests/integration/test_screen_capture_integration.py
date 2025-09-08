#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕捕获模块集成测试
"""

import unittest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chess_ai.vision.screen_capture import ScreenCaptureModule, CaptureState, CaptureError
from chess_ai.config.config_manager import ConfigManager


class MockConfigManager:
    """Mock配置管理器"""
    
    def __init__(self):
        self.vision_config = Mock()
        self.vision_config.capture_fps = 10  # 使用较低的FPS进行测试
        
    def get_vision_config(self):
        return self.vision_config


class TestScreenCaptureIntegration(unittest.TestCase):
    """屏幕捕获模块集成测试"""
    
    def setUp(self):
        """测试前置设置"""
        self.config_manager = MockConfigManager()
        
        # Mock dxcam
        self.dxcam_mock = MagicMock()
        self.dxcam_instance_mock = MagicMock()
        self.dxcam_mock.create.return_value = self.dxcam_instance_mock
        
        # 创建测试图像序列
        self.test_frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # 在每帧中添加一些标识以便验证
            frame[0:20, 0:100, :] = i * 25  # 在左上角添加标识
            self.test_frames.append(frame)
        
        self.frame_index = 0
        
        def mock_grab(*args, **kwargs):
            """模拟DXcam grab方法"""
            if self.frame_index < len(self.test_frames):
                frame = self.test_frames[self.frame_index]
                self.frame_index += 1
                return frame
            return None
        
        self.dxcam_instance_mock.grab.side_effect = mock_grab
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_full_capture_lifecycle(self, mock_dxcam):
        """测试完整的捕获生命周期"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        # 创建捕获模块并初始化
        capture = ScreenCaptureModule(self.config_manager)
        self.assertTrue(capture.initialize())
        self.assertEqual(capture.state, CaptureState.STOPPED)
        
        # 收集捕获的帧
        captured_frames = []
        def frame_callback(frame):
            captured_frames.append(frame.copy())
        
        capture.set_frame_callback(frame_callback)
        
        # 开始捕获
        self.assertTrue(capture.start_capture())
        self.assertEqual(capture.state, CaptureState.RUNNING)
        
        # 等待一段时间让捕获进行
        time.sleep(1.0)
        
        # 停止捕获
        capture.stop_capture()
        self.assertEqual(capture.state, CaptureState.STOPPED)
        
        # 验证统计信息
        stats = capture.get_capture_stats()
        self.assertGreater(stats.frames_captured, 0)
        self.assertIsNotNone(stats.last_capture_time)
        
        # 清理
        capture.cleanup()
        print(f"✓ 完整生命周期测试通过：捕获了 {stats.frames_captured} 帧")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_pause_resume_functionality(self, mock_dxcam):
        """测试暂停和恢复功能"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        capture.start_capture()
        
        # 等待一些帧
        time.sleep(0.3)
        initial_count = capture.stats.frames_captured
        
        # 暂停捕获
        capture.pause_capture()
        self.assertEqual(capture.state, CaptureState.PAUSED)
        
        # 等待暂停期间
        time.sleep(0.3)
        paused_count = capture.stats.frames_captured
        
        # 恢复捕获
        capture.resume_capture()
        self.assertEqual(capture.state, CaptureState.RUNNING)
        
        # 等待更多帧
        time.sleep(0.3)
        final_count = capture.stats.frames_captured
        
        # 验证暂停期间帧数没有增加
        self.assertEqual(initial_count, paused_count)
        self.assertGreater(final_count, paused_count)
        
        capture.stop_capture()
        capture.cleanup()
        print(f"✓ 暂停/恢复功能测试通过：初始={initial_count}, 暂停={paused_count}, 最终={final_count}")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_multi_display_switching(self, mock_dxcam):
        """测试多显示器切换"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        
        # 模拟多个显示器
        capture.available_displays = [
            Mock(index=0, name="Primary", width=1920, height=1080, is_primary=True),
            Mock(index=1, name="Secondary", width=1680, height=1050, is_primary=False)
        ]
        
        # 初始化并开始捕获
        capture.initialize()
        initial_display = capture.output_idx
        
        capture.start_capture()
        time.sleep(0.2)
        first_count = capture.stats.frames_captured
        
        # 切换到第二个显示器
        success = capture.set_output_display(1)
        self.assertTrue(success)
        self.assertEqual(capture.output_idx, 1)
        
        # 继续捕获
        time.sleep(0.2)
        second_count = capture.stats.frames_captured
        
        # 验证切换后继续工作
        self.assertGreater(second_count, first_count)
        
        capture.stop_capture()
        capture.cleanup()
        print(f"✓ 多显示器切换测试通过：从显示器{initial_display}切换到显示器1")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_fps_control_accuracy(self, mock_dxcam):
        """测试FPS控制精度"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        # 测试不同的FPS设置
        test_fps_values = [5, 10, 20]
        
        for target_fps in test_fps_values:
            with self.subTest(fps=target_fps):
                capture = ScreenCaptureModule(self.config_manager)
                capture.set_target_fps(target_fps)
                capture.initialize()
                
                # 重置帧索引
                self.frame_index = 0
                
                start_time = time.time()
                capture.start_capture()
                
                # 等待足够的时间收集数据
                time.sleep(1.0)
                
                capture.stop_capture()
                elapsed_time = time.time() - start_time
                
                actual_fps = capture.stats.frames_captured / elapsed_time
                fps_tolerance = target_fps * 0.3  # 30% 容差
                
                self.assertGreaterEqual(actual_fps, target_fps - fps_tolerance)
                self.assertLessEqual(actual_fps, target_fps + fps_tolerance)
                
                capture.cleanup()
                print(f"✓ FPS控制测试通过：目标={target_fps}, 实际={actual_fps:.1f}")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_error_recovery_mechanism(self, mock_dxcam):
        """测试错误恢复机制"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.set_retry_settings(3, 0.1)  # 快速重试用于测试
        capture.initialize()
        
        # 模拟间歇性错误
        call_count = [0]
        original_grab = self.dxcam_instance_mock.grab
        
        def error_grab(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:  # 每第3次调用失败
                return None
            return original_grab(*args, **kwargs)
        
        self.dxcam_instance_mock.grab.side_effect = error_grab
        
        capture.start_capture()
        time.sleep(0.5)  # 让它运行一段时间
        capture.stop_capture()
        
        # 验证系统处理了错误但继续运行
        self.assertGreater(capture.stats.frames_captured, 0)
        self.assertGreater(capture.stats.frames_dropped, 0)
        self.assertEqual(capture.state, CaptureState.STOPPED)  # 应该正常停止而不是错误状态
        
        capture.cleanup()
        print(f"✓ 错误恢复测试通过：捕获{capture.stats.frames_captured}帧，丢弃{capture.stats.frames_dropped}帧")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_memory_usage_stability(self, mock_dxcam):
        """测试内存使用稳定性"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        # 收集内存使用基线
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        except ImportError:
            self.skipTest("psutil不可用，跳过内存测试")
        
        # 运行捕获一段时间
        capture.start_capture()
        time.sleep(1.0)
        capture.stop_capture()
        
        # 检查内存使用
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # 验证内存增长在合理范围内 (小于50MB)
        self.assertLess(memory_increase, 50)
        
        capture.cleanup()
        print(f"✓ 内存稳定性测试通过：内存增长 {memory_increase:.1f} MB")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_concurrent_operations(self, mock_dxcam):
        """测试并发操作安全性"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        capture.start_capture()
        
        # 并发访问帧数据
        def frame_accessor():
            for _ in range(10):
                frame = capture.get_latest_frame()
                if frame is not None:
                    # 简单处理帧数据
                    _ = frame.shape
                time.sleep(0.05)
        
        # 启动多个并发线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=frame_accessor)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
        
        capture.stop_capture()
        capture.cleanup()
        print("✓ 并发操作安全性测试通过")
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    def test_performance_monitoring_accuracy(self, mock_dxcam):
        """测试性能监控准确性"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.initialize()
        
        # 进行已知次数的捕获
        expected_captures = 5
        for i in range(expected_captures):
            frame = capture.capture_single_frame()
            self.assertIsNotNone(frame)
            time.sleep(0.1)
        
        # 验证统计准确性
        stats = capture.get_capture_stats()
        self.assertEqual(stats.frames_captured, expected_captures)
        self.assertEqual(stats.frames_dropped, 0)
        self.assertEqual(stats.error_count, 0)
        
        # 测试性能报告
        report = capture.get_performance_report()
        self.assertEqual(report['frames_captured'], expected_captures)
        self.assertEqual(report['success_rate_percent'], 100.0)
        self.assertGreaterEqual(report['uptime_seconds'], 0)
        
        capture.cleanup()
        print(f"✓ 性能监控准确性测试通过：预期{expected_captures}帧，实际{stats.frames_captured}帧")


class TestScreenCaptureStressTest(unittest.TestCase):
    """屏幕捕获模块压力测试"""
    
    def setUp(self):
        """测试前置设置"""
        self.config_manager = MockConfigManager()
        
        # Mock dxcam with high frequency frames
        self.dxcam_mock = MagicMock()
        self.dxcam_instance_mock = MagicMock()
        self.dxcam_mock.create.return_value = self.dxcam_instance_mock
        
        # 创建高频测试图像
        def mock_grab(*args, **kwargs):
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        self.dxcam_instance_mock.grab.side_effect = mock_grab
    
    @patch('chess_ai.vision.screen_capture.DXCAM_AVAILABLE', True)
    @patch('chess_ai.vision.screen_capture.dxcam')
    @unittest.skipIf(os.environ.get('SKIP_STRESS_TESTS'), "跳过压力测试")
    def test_high_frequency_capture(self, mock_dxcam):
        """测试高频率捕获"""
        mock_dxcam.create.return_value = self.dxcam_instance_mock
        
        capture = ScreenCaptureModule(self.config_manager)
        capture.set_target_fps(60)  # 高FPS测试
        capture.initialize()
        
        start_time = time.time()
        capture.start_capture()
        
        # 运行较长时间
        time.sleep(2.0)
        
        capture.stop_capture()
        elapsed = time.time() - start_time
        
        # 验证性能
        stats = capture.get_capture_stats()
        actual_fps = stats.frames_captured / elapsed
        
        self.assertGreater(stats.frames_captured, 60)  # 至少捕获60帧
        self.assertLess(stats.error_count / max(stats.frames_captured, 1), 0.05)  # 错误率<5%
        
        capture.cleanup()
        print(f"✓ 高频率捕获测试通过：{actual_fps:.1f} FPS，{stats.frames_captured}帧，{stats.error_count}错误")


if __name__ == '__main__':
    # 设置测试运行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stress', action='store_true', help='运行压力测试')
    args, remaining = parser.parse_known_args()
    
    if not args.stress:
        os.environ['SKIP_STRESS_TESTS'] = '1'
    
    # 运行测试
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
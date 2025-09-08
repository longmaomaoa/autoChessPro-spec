#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_functionality():
    print("Testing screen capture module basic functionality...")
    
    try:
        # 测试数据类导入和创建
        from chess_ai.vision.screen_capture import CaptureRegion, DisplayInfo, CaptureStats, CaptureState, CaptureError
        
        # 测试CaptureRegion
        region = CaptureRegion(100, 200, 800, 600)
        print(f"CaptureRegion created: {region.x}, {region.y}, {region.width}, {region.height}")
        
        tuple_result = region.to_tuple()
        expected = (100, 200, 900, 800)
        assert tuple_result == expected, f"Expected {expected}, got {tuple_result}"
        print("CaptureRegion.to_tuple() test passed")
        
        # 测试从元组创建
        region2 = CaptureRegion.from_tuple((50, 60, 350, 460))
        assert region2.x == 50 and region2.y == 60 and region2.width == 300 and region2.height == 400
        print("CaptureRegion.from_tuple() test passed")
        
        # 测试DisplayInfo
        display = DisplayInfo(0, "Test Monitor", 1920, 1080, True)
        display_str = str(display)
        print(f"DisplayInfo string: {display_str}")
        
        # 测试CaptureStats
        stats = CaptureStats()
        assert stats.frames_captured == 0
        assert stats.error_count == 0
        print("CaptureStats test passed")
        
        # 测试CaptureState枚举
        assert CaptureState.STOPPED.value == "stopped"
        assert CaptureState.RUNNING.value == "running"
        print("CaptureState test passed")
        
        # 测试CaptureError
        try:
            raise CaptureError("Test error", "TEST_CODE")
        except CaptureError as e:
            assert str(e) == "Test error"
            assert e.error_code == "TEST_CODE"
            print("CaptureError test passed")
        
        print("All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_structure():
    print("\nTesting module structure...")
    
    try:
        # 直接测试ScreenCaptureModule而不依赖配置管理器
        from chess_ai.vision.screen_capture import ScreenCaptureModule
        
        # 创建mock配置
        class MockConfig:
            def __init__(self):
                self.capture_fps = 30
        
        class MockConfigManager:
            def __init__(self):
                self.vision_config = MockConfig()
            
            def get_vision_config(self):
                return self.vision_config
        
        # 测试模块创建
        config_manager = MockConfigManager()
        capture = ScreenCaptureModule(config_manager)
        
        # 验证初始状态
        assert capture.state.value == "stopped"
        assert capture.target_fps == 30
        assert len(capture.available_displays) > 0
        print("ScreenCaptureModule initialization test passed")
        
        # 测试基本方法
        assert capture.set_target_fps(60) == True
        assert capture.target_fps == 60
        
        assert capture.set_target_fps(0) == False  # 无效FPS
        assert capture.set_target_fps(300) == False  # 无效FPS
        print("set_target_fps() test passed")
        
        # 测试捕获区域设置
        assert capture.set_capture_region(100, 100, 800, 600) == True
        assert capture.capture_region is not None
        
        assert capture.set_capture_region(100, 100, 0, 600) == False  # 无效宽度
        print("set_capture_region() test passed")
        
        # 测试重试设置
        assert capture.set_retry_settings(3, 0.5) == True
        assert capture.max_retry_attempts == 3
        assert capture.retry_delay == 0.5
        
        assert capture.set_retry_settings(0, 0.5) == False  # 无效重试次数
        print("set_retry_settings() test passed")
        
        # 测试自动恢复
        capture.set_auto_recovery(False)
        assert capture.auto_recovery == False
        capture.set_auto_recovery(True)
        assert capture.auto_recovery == True
        print("auto_recovery test passed")
        
        # 测试统计
        capture.stats.frames_captured = 10
        capture.reset_stats()
        assert capture.stats.frames_captured == 0
        print("reset_stats() test passed")
        
        # 测试健康检查
        is_healthy = capture.is_healthy()
        print(f"Health check result: {is_healthy}")
        
        # 测试诊断信息
        diagnostics = capture.get_diagnostics()
        required_keys = ['system_info', 'module_state', 'settings', 'health_check']
        for key in required_keys:
            assert key in diagnostics, f"Missing key: {key}"
        print("get_diagnostics() test passed")
        
        # 测试性能报告
        report = capture.get_performance_report()
        required_keys = ['state', 'frames_captured', 'target_fps', 'display_count']
        for key in required_keys:
            assert key in report, f"Missing key: {key}"
        print("get_performance_report() test passed")
        
        # 测试显示器列表
        displays = capture.get_available_displays()
        assert len(displays) > 0
        print(f"Available displays: {len(displays)}")
        
        print("Module structure tests passed!")
        return True
        
    except Exception as e:
        print(f"Module structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Screen Capture Module Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 2
    
    # 运行基本功能测试
    if test_basic_functionality():
        tests_passed += 1
    
    # 运行模块结构测试
    if test_module_structure():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("All tests passed successfully!")
        return True
    else:
        print("Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
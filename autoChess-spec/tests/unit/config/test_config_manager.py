#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器单元测试
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from chess_ai.config.config_manager import (
    ConfigManager, ScreenCaptureConfig, VisionConfig, 
    AIEngineConfig, UIConfig, LoggingConfig, 
    PerformanceConfig, DataConfig
)


class TestConfigManager:
    """配置管理器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        assert self.config_manager.config_dir == Path(self.temp_dir)
        assert isinstance(self.config_manager.screen_capture, ScreenCaptureConfig)
        assert isinstance(self.config_manager.vision, VisionConfig)
        assert isinstance(self.config_manager.ai_engine, AIEngineConfig)
        assert isinstance(self.config_manager.ui, UIConfig)
        assert isinstance(self.config_manager.logging, LoggingConfig)
        assert isinstance(self.config_manager.performance, PerformanceConfig)
        assert isinstance(self.config_manager.data, DataConfig)
    
    def test_default_config_values(self):
        """测试默认配置值"""
        # 屏幕捕获默认值
        assert self.config_manager.screen_capture.enabled == True
        assert self.config_manager.screen_capture.fps == 10
        assert self.config_manager.screen_capture.quality == "high"
        
        # 视觉识别默认值
        assert self.config_manager.vision.confidence_threshold == 0.8
        assert self.config_manager.vision.board_detection_enabled == True
        
        # AI引擎默认值
        assert self.config_manager.ai_engine.thinking_time == 3000
        assert self.config_manager.ai_engine.depth == 15
        assert self.config_manager.ai_engine.difficulty_level == "expert"
        
        # UI默认值
        assert self.config_manager.ui.theme == "default"
        assert self.config_manager.ui.language == "zh_CN"
        assert self.config_manager.ui.window_width == 1200
    
    def test_config_loading(self):
        """测试配置加载"""
        success = self.config_manager.load_config()
        assert success
        
        # 默认配置文件应该被创建
        assert self.config_manager.default_config_file.exists()
    
    def test_config_get_set(self):
        """测试配置获取和设置"""
        # 获取配置值
        fps = self.config_manager.get("screen_capture.fps", 0)
        assert fps == 10
        
        # 设置配置值
        self.config_manager.set("screen_capture.fps", 15)
        new_fps = self.config_manager.get("screen_capture.fps", 0)
        assert new_fps == 15
        
        # 验证配置对象也被更新
        assert self.config_manager.screen_capture.fps == 15
        
        # 获取不存在的配置
        non_existent = self.config_manager.get("non.existent.key", "default")
        assert non_existent == "default"
    
    def test_config_section_get(self):
        """测试配置节获取"""
        screen_capture_section = self.config_manager.get_section("screen_capture")
        assert isinstance(screen_capture_section, dict)
        assert "enabled" in screen_capture_section
        assert "fps" in screen_capture_section
    
    def test_config_update(self):
        """测试批量配置更新"""
        updates = {
            "screen_capture": {
                "fps": 20,
                "quality": "medium"
            },
            "ui": {
                "theme": "dark",
                "window_width": 1400
            }
        }
        
        self.config_manager.update_config(updates)
        
        assert self.config_manager.screen_capture.fps == 20
        assert self.config_manager.screen_capture.quality == "medium"
        assert self.config_manager.ui.theme == "dark"
        assert self.config_manager.ui.window_width == 1400
    
    def test_config_save_and_load(self):
        """测试配置保存和加载"""
        # 修改一些配置
        self.config_manager.set("screen_capture.fps", 25)
        self.config_manager.set("ui.theme", "dark")
        
        # 保存配置
        success = self.config_manager.save_user_config()
        assert success
        assert self.config_manager.user_config_file.exists()
        
        # 创建新的配置管理器并加载
        new_config_manager = ConfigManager(self.temp_dir)
        new_config_manager.load_config()
        
        # 验证配置被正确加载
        assert new_config_manager.screen_capture.fps == 25
        assert new_config_manager.ui.theme == "dark"
    
    def test_config_validation(self):
        """测试配置验证"""
        self.config_manager.load_config()
        
        # 初始配置应该没有错误
        errors = self.config_manager.validate_config()
        assert len(errors) == 0
        
        # 设置无效配置
        self.config_manager.set("screen_capture.fps", -1)
        self.config_manager.set("vision.confidence_threshold", 1.5)
        
        # 应该有验证错误
        errors = self.config_manager.validate_config()
        assert len(errors) > 0
        assert any("帧率必须在1-60之间" in error for error in errors)
        assert any("置信度阈值必须在0.0-1.0之间" in error for error in errors)
    
    def test_config_reset_to_defaults(self):
        """测试重置为默认配置"""
        self.config_manager.load_config()
        
        # 修改配置
        self.config_manager.set("screen_capture.fps", 30)
        self.config_manager.set("ui.theme", "dark")
        
        # 重置
        success = self.config_manager.reset_to_defaults()
        assert success
        
        # 验证配置已重置
        assert self.config_manager.screen_capture.fps == 10
        assert self.config_manager.ui.theme == "default"
    
    def test_config_export_import(self):
        """测试配置导出导入"""
        self.config_manager.load_config()
        
        # 修改一些配置
        self.config_manager.set("screen_capture.fps", 35)
        self.config_manager.set("ui.language", "en_US")
        
        # 导出配置
        export_file = Path(self.temp_dir) / "exported_config.yaml"
        success = self.config_manager.export_config(str(export_file))
        assert success
        assert export_file.exists()
        
        # 重置配置
        self.config_manager.reset_to_defaults()
        assert self.config_manager.screen_capture.fps == 10
        assert self.config_manager.ui.language == "zh_CN"
        
        # 导入配置
        success = self.config_manager.import_config(str(export_file))
        assert success
        assert self.config_manager.screen_capture.fps == 35
        assert self.config_manager.ui.language == "en_US"
    
    def test_config_change_callbacks(self):
        """测试配置变化回调"""
        callback_called = False
        callback_config = None
        
        def test_callback(config):
            nonlocal callback_called, callback_config
            callback_called = True
            callback_config = config
        
        # 添加回调
        self.config_manager.add_change_callback(test_callback)
        
        # 保存配置应该触发回调
        self.config_manager.save_user_config()
        
        assert callback_called
        assert callback_config == self.config_manager
        
        # 移除回调
        self.config_manager.remove_change_callback(test_callback)
        
        callback_called = False
        self.config_manager.save_user_config()
        # 回调不应该被调用
        assert not callback_called
    
    def test_config_info(self):
        """测试配置信息获取"""
        self.config_manager.load_config()
        
        info = self.config_manager.get_config_info()
        
        assert "default_config_file" in info
        assert "user_config_file" in info
        assert "config_dir" in info
        assert "validation_errors" in info
        assert "total_config_items" in info
        
        assert isinstance(info["validation_errors"], list)
        assert isinstance(info["total_config_items"], int)
    
    def test_malformed_config_file(self):
        """测试格式错误的配置文件"""
        # 创建格式错误的用户配置文件
        malformed_content = "invalid: yaml: content: ["
        with open(self.config_manager.user_config_file, 'w') as f:
            f.write(malformed_content)
        
        # 加载配置应该成功（回退到默认配置）
        success = self.config_manager.load_config()
        assert success
        
        # 用户配置应该为空字典（默认值）
        assert self.config_manager.screen_capture.fps == 10  # 默认值
    
    def test_missing_config_sections(self):
        """测试缺失配置节"""
        # 创建只有部分配置节的文件
        partial_config = {
            "screen_capture": {
                "fps": 25
            }
            # 缺失其他配置节
        }
        
        self.config_manager.user_config_file.parent.mkdir(exist_ok=True)
        with open(self.config_manager.user_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(partial_config, f)
        
        # 加载配置
        success = self.config_manager.load_config()
        assert success
        
        # 用户配置的值应该被使用
        assert self.config_manager.screen_capture.fps == 25
        
        # 缺失的配置应该使用默认值
        assert self.config_manager.ui.theme == "default"
        assert self.config_manager.ai_engine.depth == 15


class TestConfigDataClasses:
    """配置数据类测试"""
    
    def test_screen_capture_config(self):
        """测试屏幕捕获配置"""
        config = ScreenCaptureConfig(
            fps=30,
            region_width=1024,
            region_height=768,
            quality="medium"
        )
        
        assert config.fps == 30
        assert config.region_width == 1024
        assert config.region_height == 768
        assert config.quality == "medium"
        assert config.enabled == True  # 默认值
    
    def test_vision_config(self):
        """测试视觉识别配置"""
        config = VisionConfig(
            confidence_threshold=0.9,
            model_path="custom/model.pt",
            board_detection_method="yolo"
        )
        
        assert config.confidence_threshold == 0.9
        assert config.model_path == "custom/model.pt"
        assert config.board_detection_method == "yolo"
        assert config.board_detection_enabled == True  # 默认值
    
    def test_ai_engine_config(self):
        """测试AI引擎配置"""
        config = AIEngineConfig(
            thinking_time=5000,
            depth=20,
            difficulty_level="beginner",
            threads=8
        )
        
        assert config.thinking_time == 5000
        assert config.depth == 20
        assert config.difficulty_level == "beginner"
        assert config.threads == 8
        assert config.engine_type == "pikafish"  # 默认值


if __name__ == "__main__":
    pytest.main([__file__])
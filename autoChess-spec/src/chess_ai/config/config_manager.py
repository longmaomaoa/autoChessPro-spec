#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from loguru import logger
import threading
from datetime import datetime


@dataclass
class ScreenCaptureConfig:
    """屏幕捕获配置"""
    enabled: bool = True
    fps: int = 10
    region_x: int = 0
    region_y: int = 0
    region_width: int = 800
    region_height: int = 600
    auto_detect_region: bool = True
    quality: str = "high"  # high, medium, low


@dataclass
class VisionConfig:
    """视觉识别配置"""
    board_detection_enabled: bool = True
    piece_classification_enabled: bool = True
    confidence_threshold: float = 0.8
    model_path: str = "models/chess_piece_detector.pt"
    board_detection_method: str = "opencv"  # opencv, yolo
    calibration_enabled: bool = True
    capture_fps: int = 30  # 屏幕捕获帧率


@dataclass
class AIEngineConfig:
    """AI引擎配置"""
    engine_path: str = "engines/pikafish.exe"
    engine_type: str = "pikafish"
    thinking_time: int = 3000  # 毫秒
    depth: int = 15
    threads: int = 4
    hash_size: int = 256  # MB
    difficulty_level: str = "expert"  # beginner, intermediate, advanced, expert
    multi_pv: int = 3  # 多候选走法数量


@dataclass
class UIConfig:
    """用户界面配置"""
    theme: str = "default"  # default, dark, light
    language: str = "zh_CN"
    window_width: int = 1200
    window_height: int = 800
    always_on_top: bool = False
    show_coordinates: bool = True
    show_move_hints: bool = True
    animation_enabled: bool = True
    font_size: int = 12


@dataclass
class LoggingConfig:
    """日志配置"""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    file_enabled: bool = True
    file_path: str = "logs/chess_ai.log"
    rotation: str = "10 MB"
    retention: str = "30 days"
    format: str = "{time} | {level} | {name}:{function}:{line} - {message}"


@dataclass
class PerformanceConfig:
    """性能配置"""
    max_cpu_usage: int = 80  # 百分比
    max_memory_usage: int = 512  # MB
    parallel_processing: bool = True
    cache_enabled: bool = True
    cache_size: int = 100  # 缓存项目数量


@dataclass
class DataConfig:
    """数据配置"""
    database_path: str = "data/chess_ai.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 秒
    export_format: str = "pgn"  # pgn, json, csv
    privacy_mode: bool = False


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG_PATH = "config/default.yaml"
    USER_CONFIG_PATH = "config/user.yaml"
    
    def __init__(self, config_dir: Optional[str] = None):
        """初始化配置管理器"""
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.default_config_file = self.config_dir / self.DEFAULT_CONFIG_PATH
        self.user_config_file = self.config_dir / self.USER_CONFIG_PATH
        
        # 配置数据
        self._config_data: Dict[str, Any] = {}
        self._default_config: Dict[str, Any] = {}
        self._user_config: Dict[str, Any] = {}
        
        # 配置对象
        self.screen_capture = ScreenCaptureConfig()
        self.vision = VisionConfig()
        self.ai_engine = AIEngineConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        self.data = DataConfig()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 配置变化回调
        self._change_callbacks: List[callable] = []
        
        logger.info(f"配置管理器初始化，配置目录: {self.config_dir}")
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            with self._lock:
                # 加载默认配置
                self._load_default_config()
                
                # 加载用户配置
                self._load_user_config()
                
                # 合并配置
                self._merge_configs()
                
                # 更新配置对象
                self._update_config_objects()
                
                logger.info("配置加载完成")
                return True
                
        except Exception as e:
            logger.exception(f"加载配置失败: {e}")
            return False
    
    def _load_default_config(self) -> None:
        """加载默认配置"""
        # 如果默认配置文件不存在，创建它
        if not self.default_config_file.exists():
            self._create_default_config()
        
        # 读取默认配置
        with open(self.default_config_file, 'r', encoding='utf-8') as f:
            self._default_config = yaml.safe_load(f) or {}
        
        logger.debug("默认配置加载完成")
    
    def _load_user_config(self) -> None:
        """加载用户配置"""
        if self.user_config_file.exists():
            try:
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    self._user_config = yaml.safe_load(f) or {}
                logger.debug("用户配置加载完成")
            except Exception as e:
                logger.warning(f"加载用户配置失败，使用默认配置: {e}")
                self._user_config = {}
        else:
            self._user_config = {}
            logger.debug("用户配置文件不存在，使用默认配置")
    
    def _merge_configs(self) -> None:
        """合并配置"""
        self._config_data = self._deep_merge(self._default_config, self._user_config)
    
    def _update_config_objects(self) -> None:
        """更新配置对象"""
        # 屏幕捕获配置
        screen_capture_data = self._config_data.get('screen_capture', {})
        self.screen_capture = ScreenCaptureConfig(**{
            k: v for k, v in screen_capture_data.items() 
            if k in ScreenCaptureConfig.__dataclass_fields__
        })
        
        # 视觉识别配置
        vision_data = self._config_data.get('vision', {})
        self.vision = VisionConfig(**{
            k: v for k, v in vision_data.items() 
            if k in VisionConfig.__dataclass_fields__
        })
        
        # AI引擎配置
        ai_engine_data = self._config_data.get('ai_engine', {})
        self.ai_engine = AIEngineConfig(**{
            k: v for k, v in ai_engine_data.items() 
            if k in AIEngineConfig.__dataclass_fields__
        })
        
        # 用户界面配置
        ui_data = self._config_data.get('ui', {})
        self.ui = UIConfig(**{
            k: v for k, v in ui_data.items() 
            if k in UIConfig.__dataclass_fields__
        })
        
        # 日志配置
        logging_data = self._config_data.get('logging', {})
        self.logging = LoggingConfig(**{
            k: v for k, v in logging_data.items() 
            if k in LoggingConfig.__dataclass_fields__
        })
        
        # 性能配置
        performance_data = self._config_data.get('performance', {})
        self.performance = PerformanceConfig(**{
            k: v for k, v in performance_data.items() 
            if k in PerformanceConfig.__dataclass_fields__
        })
        
        # 数据配置
        data_data = self._config_data.get('data', {})
        self.data = DataConfig(**{
            k: v for k, v in data_data.items() 
            if k in DataConfig.__dataclass_fields__
        })
    
    def _create_default_config(self) -> None:
        """创建默认配置文件"""
        # 确保配置目录存在
        self.default_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建默认配置
        default_config = {
            'screen_capture': asdict(ScreenCaptureConfig()),
            'vision': asdict(VisionConfig()),
            'ai_engine': asdict(AIEngineConfig()),
            'ui': asdict(UIConfig()),
            'logging': asdict(LoggingConfig()),
            'performance': asdict(PerformanceConfig()),
            'data': asdict(DataConfig()),
            'version': '1.0.0',
            'created_at': datetime.now().isoformat()
        }
        
        # 写入文件
        with open(self.default_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"默认配置文件已创建: {self.default_config_file}")
    
    def save_user_config(self) -> bool:
        """保存用户配置"""
        try:
            with self._lock:
                # 确保用户配置目录存在
                self.user_config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 收集当前配置
                current_config = {
                    'screen_capture': asdict(self.screen_capture),
                    'vision': asdict(self.vision),
                    'ai_engine': asdict(self.ai_engine),
                    'ui': asdict(self.ui),
                    'logging': asdict(self.logging),
                    'performance': asdict(self.performance),
                    'data': asdict(self.data),
                    'updated_at': datetime.now().isoformat()
                }
                
                # 写入用户配置文件
                with open(self.user_config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
                
                logger.info("用户配置保存完成")
                
                # 通知配置变化
                self._notify_config_changed()
                
                return True
                
        except Exception as e:
            logger.exception(f"保存用户配置失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            keys = key.split('.')
            value = self._config_data
            
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        with self._lock:
            keys = key.split('.')
            config = self._config_data
            
            # 导航到父级字典
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # 设置值
            config[keys[-1]] = value
            
            # 更新配置对象
            self._update_config_objects()
            
            logger.debug(f"配置已更新: {key} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节"""
        return self.get(section, {})
    
    def get_vision_config(self) -> 'VisionConfig':
        """获取视觉识别配置对象"""
        return self.vision
    
    def get_screen_capture_config(self) -> 'ScreenCaptureConfig':
        """获取屏幕捕获配置对象"""
        return self.screen_capture
    
    def get_ai_engine_config(self) -> 'AIEngineConfig':
        """获取AI引擎配置对象"""
        return self.ai_engine
    
    def get_ui_config(self) -> 'UIConfig':
        """获取UI配置对象"""
        return self.ui
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """批量更新配置"""
        with self._lock:
            self._config_data = self._deep_merge(self._config_data, updates)
            self._update_config_objects()
            
            logger.info(f"批量更新配置，共更新 {len(updates)} 个配置项")
    
    def reset_to_defaults(self) -> bool:
        """重置为默认配置"""
        try:
            with self._lock:
                # 重新加载默认配置
                self._load_default_config()
                self._config_data = self._default_config.copy()
                self._update_config_objects()
                
                logger.info("配置已重置为默认值")
                
                # 保存到用户配置
                return self.save_user_config()
                
        except Exception as e:
            logger.exception(f"重置配置失败: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        
        try:
            # 验证屏幕捕获配置
            if not (1 <= self.screen_capture.fps <= 60):
                errors.append("屏幕捕获帧率必须在1-60之间")
            
            if self.screen_capture.region_width <= 0 or self.screen_capture.region_height <= 0:
                errors.append("屏幕捕获区域尺寸必须大于0")
            
            # 验证视觉识别配置
            if not (0.0 <= self.vision.confidence_threshold <= 1.0):
                errors.append("置信度阈值必须在0.0-1.0之间")
            
            model_path = Path(self.vision.model_path)
            if not model_path.exists() and not model_path.is_absolute():
                # 尝试相对于配置目录的路径
                full_model_path = self.config_dir / model_path
                if not full_model_path.exists():
                    errors.append(f"视觉识别模型文件不存在: {self.vision.model_path}")
            
            # 验证AI引擎配置
            if self.ai_engine.thinking_time < 100:
                errors.append("AI思考时间不能少于100毫秒")
            
            if not (1 <= self.ai_engine.depth <= 50):
                errors.append("AI搜索深度必须在1-50之间")
            
            engine_path = Path(self.ai_engine.engine_path)
            if not engine_path.exists() and not engine_path.is_absolute():
                # 尝试相对于配置目录的路径
                full_engine_path = self.config_dir / engine_path
                if not full_engine_path.exists():
                    errors.append(f"AI引擎文件不存在: {self.ai_engine.engine_path}")
            
            # 验证性能配置
            if not (1 <= self.performance.max_cpu_usage <= 100):
                errors.append("最大CPU使用率必须在1-100之间")
            
            if self.performance.max_memory_usage < 64:
                errors.append("最大内存使用量不能少于64MB")
            
        except Exception as e:
            errors.append(f"配置验证异常: {e}")
        
        return errors
    
    def add_change_callback(self, callback: callable) -> None:
        """添加配置变化回调"""
        if callback not in self._change_callbacks:
            self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: callable) -> None:
        """移除配置变化回调"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def _notify_config_changed(self) -> None:
        """通知配置变化"""
        for callback in self._change_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.exception(f"配置变化回调异常: {e}")
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def export_config(self, file_path: str) -> bool:
        """导出配置到文件"""
        try:
            config_to_export = {
                'screen_capture': asdict(self.screen_capture),
                'vision': asdict(self.vision),
                'ai_engine': asdict(self.ai_engine),
                'ui': asdict(self.ui),
                'logging': asdict(self.logging),
                'performance': asdict(self.performance),
                'data': asdict(self.data),
                'exported_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_export, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已导出到: {file_path}")
            return True
            
        except Exception as e:
            logger.exception(f"导出配置失败: {e}")
            return False
    
    def import_config(self, file_path: str) -> bool:
        """从文件导入配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = yaml.safe_load(f)
            
            if not isinstance(imported_config, dict):
                logger.error("导入的配置文件格式错误")
                return False
            
            # 验证导入的配置
            temp_config = ConfigManager()
            temp_config._config_data = imported_config
            temp_config._update_config_objects()
            
            errors = temp_config.validate_config()
            if errors:
                logger.error(f"导入的配置验证失败: {errors}")
                return False
            
            # 应用导入的配置
            with self._lock:
                self._config_data = imported_config
                self._update_config_objects()
            
            logger.info(f"配置已从文件导入: {file_path}")
            
            # 保存到用户配置
            return self.save_user_config()
            
        except Exception as e:
            logger.exception(f"导入配置失败: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'default_config_file': str(self.default_config_file),
            'user_config_file': str(self.user_config_file),
            'config_dir': str(self.config_dir),
            'default_config_exists': self.default_config_file.exists(),
            'user_config_exists': self.user_config_file.exists(),
            'validation_errors': self.validate_config(),
            'total_config_items': len(self._config_data)
        }
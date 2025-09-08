#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕捕获模块

该模块使用DXcam实现高性能屏幕捕获，支持实时棋局监控。
基于Desktop Duplication API，可实现240Hz+的高频率捕获，满足实时监控需求。
"""

import time
import threading
from typing import Optional, Tuple, Union, Callable, Any, List, Dict
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging
from datetime import datetime
import platform
import traceback

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    dxcam = None

# Windows API支持（用于多显示器检测）
if platform.system() == "Windows":
    try:
        import win32api
        import win32con
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
else:
    WIN32_AVAILABLE = False

from ..utils.logger import get_logger
from ..config.config_manager import ConfigManager

logger = get_logger(__name__)


class CaptureState(Enum):
    """捕获状态枚举"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class CaptureRegion:
    """捕获区域信息"""
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """转换为元组格式"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @classmethod
    def from_tuple(cls, region: Tuple[int, int, int, int]) -> 'CaptureRegion':
        """从元组创建区域"""
        return cls(region[0], region[1], region[2] - region[0], region[3] - region[1])


@dataclass
class DisplayInfo:
    """显示器信息"""
    index: int
    name: str
    width: int
    height: int
    is_primary: bool = False
    
    def __str__(self) -> str:
        primary_str = " (主显示器)" if self.is_primary else ""
        return f"显示器 {self.index}: {self.name} ({self.width}x{self.height}){primary_str}"


@dataclass
class CaptureStats:
    """捕获统计信息"""
    frames_captured: int = 0
    frames_dropped: int = 0
    average_fps: float = 0.0
    last_capture_time: Optional[datetime] = None
    error_count: int = 0
    retry_count: int = 0
    total_runtime: float = 0.0


class CaptureError(Exception):
    """屏幕捕获相关错误"""
    
    def __init__(self, message: str, error_code: str = None, cause: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.cause = cause
        self.timestamp = datetime.now()


class ScreenCaptureModule:
    """高性能屏幕捕获模块
    
    使用DXcam库实现基于Desktop Duplication API的屏幕捕获，
    支持高频率实时捕获、多显示器、区域设置等功能。
    """
    
    def __init__(self, config_manager: ConfigManager):
        """初始化屏幕捕获模块
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.vision_config = config_manager.get_vision_config()
        
        # 核心组件
        self.dxcam_instance: Optional[Any] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.state = CaptureState.STOPPED
        
        # 捕获设置
        self.target_fps = self.vision_config.capture_fps
        self.capture_region: Optional[CaptureRegion] = None
        self.output_idx = 0  # 默认主显示器
        
        # 状态管理
        self.is_capturing = False
        self.frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.stats = CaptureStats()
        
        # 线程同步
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # 性能监控
        self._frame_times = []
        self._last_fps_update = time.time()
        
        # 错误恢复设置
        self.max_retry_attempts = 5
        self.retry_delay = 0.1
        self.auto_recovery = True
        
        # 多显示器支持
        self.available_displays: List[DisplayInfo] = []
        self._detect_displays()
        
        logger.info(f"屏幕捕获模块初始化完成，目标FPS: {self.target_fps}")
        logger.info(f"检测到 {len(self.available_displays)} 个显示器")
    
    def _detect_displays(self) -> None:
        """检测可用的显示器"""
        self.available_displays.clear()
        
        try:
            if WIN32_AVAILABLE:
                # 使用Windows API检测显示器
                monitors = win32api.EnumDisplayMonitors(None, None)
                for i, (hmon, hdc, rect) in enumerate(monitors):
                    try:
                        monitor_info = win32api.GetMonitorInfo(hmon)
                        device_name = monitor_info.get('Device', f'Monitor{i}')
                        is_primary = monitor_info.get('Flags', 0) == win32con.MONITORINFOF_PRIMARY
                        
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        
                        display = DisplayInfo(
                            index=i,
                            name=device_name,
                            width=width,
                            height=height,
                            is_primary=is_primary
                        )
                        self.available_displays.append(display)
                        logger.info(f"检测到显示器: {display}")
                        
                    except Exception as e:
                        logger.warning(f"获取显示器 {i} 信息失败: {e}")
            
            else:
                # 备用方案：创建默认显示器信息
                logger.warning("Windows API不可用，使用默认显示器配置")
                self.available_displays.append(DisplayInfo(
                    index=0,
                    name="Default Monitor",
                    width=1920,
                    height=1080,
                    is_primary=True
                ))
                
        except Exception as e:
            logger.error(f"检测显示器失败: {e}")
            # 创建默认显示器作为备用
            if not self.available_displays:
                self.available_displays.append(DisplayInfo(
                    index=0,
                    name="Fallback Monitor",
                    width=1920,
                    height=1080,
                    is_primary=True
                ))
    
    def get_available_displays(self) -> List[DisplayInfo]:
        """获取可用显示器列表
        
        Returns:
            List[DisplayInfo]: 显示器信息列表
        """
        return self.available_displays.copy()
    
    def set_output_display(self, display_index: int) -> bool:
        """设置输出显示器
        
        Args:
            display_index: 显示器索引
            
        Returns:
            bool: 设置成功返回True
        """
        if not 0 <= display_index < len(self.available_displays):
            logger.error(f"无效的显示器索引: {display_index}")
            return False
        
        # 停止当前捕获
        was_capturing = self.state == CaptureState.RUNNING
        if was_capturing:
            self.stop_capture()
        
        # 更新显示器设置
        self.output_idx = display_index
        logger.info(f"切换到显示器: {self.available_displays[display_index]}")
        
        # 重新初始化
        self.cleanup()
        success = self.initialize()
        
        # 恢复捕获状态
        if was_capturing and success:
            self.start_capture()
        
        return success
    
    def initialize(self) -> bool:
        """初始化DXcam捕获系统
        
        Returns:
            bool: 初始化成功返回True，否则返回False
        """
        if not DXCAM_AVAILABLE:
            error_msg = "DXcam库不可用，请安装dxcam: pip install dxcam"
            logger.error(error_msg)
            self.state = CaptureState.ERROR
            raise CaptureError(error_msg, "DXCAM_UNAVAILABLE")
        
        retry_count = 0
        while retry_count < self.max_retry_attempts:
            try:
                self.state = CaptureState.INITIALIZING
                logger.info(f"正在初始化DXcam... (尝试 {retry_count + 1}/{self.max_retry_attempts})")
                
                # 验证显示器索引
                if self.output_idx >= len(self.available_displays):
                    logger.warning(f"显示器索引 {self.output_idx} 超出范围，使用主显示器")
                    self.output_idx = 0
                
                # 创建DXcam实例
                self.dxcam_instance = dxcam.create(
                    output_idx=self.output_idx,
                    output_color="RGB"
                )
                
                if self.dxcam_instance is None:
                    raise CaptureError("DXcam实例创建失败", "INSTANCE_CREATION_FAILED")
                
                # 验证DXcam功能
                test_frame = self.dxcam_instance.grab()
                if test_frame is None:
                    logger.warning("DXcam测试捕获失败，可能存在权限或兼容性问题")
                
                logger.info(f"DXcam初始化成功，使用显示器: {self.output_idx}")
                if self.available_displays:
                    logger.info(f"当前显示器: {self.available_displays[self.output_idx]}")
                
                self.state = CaptureState.STOPPED
                self.stats.retry_count += retry_count
                return True
                
            except Exception as e:
                retry_count += 1
                self.stats.error_count += 1
                
                if retry_count < self.max_retry_attempts:
                    logger.warning(f"DXcam初始化失败 (尝试 {retry_count}/{self.max_retry_attempts}): {e}")
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    error_msg = f"DXcam初始化失败，已达到最大重试次数: {e}"
                    logger.error(error_msg)
                    self.state = CaptureState.ERROR
                    raise CaptureError(error_msg, "INITIALIZATION_FAILED", e)
        
        return False
    
    def set_capture_region(self, x: int, y: int, width: int, height: int) -> bool:
        """设置捕获区域
        
        Args:
            x: 区域左上角X坐标
            y: 区域左上角Y坐标  
            width: 区域宽度
            height: 区域高度
            
        Returns:
            bool: 设置成功返回True
        """
        try:
            if width <= 0 or height <= 0:
                logger.error("捕获区域尺寸必须大于0")
                return False
            
            self.capture_region = CaptureRegion(x, y, width, height)
            logger.info(f"设置捕获区域: ({x}, {y}) - {width}x{height}")
            return True
            
        except Exception as e:
            logger.error(f"设置捕获区域失败: {e}")
            return False
    
    def set_target_fps(self, fps: int) -> bool:
        """设置目标帧率
        
        Args:
            fps: 目标帧率 (1-240)
            
        Returns:
            bool: 设置成功返回True
        """
        if not 1 <= fps <= 240:
            logger.error("目标FPS必须在1-240之间")
            return False
        
        self.target_fps = fps
        logger.info(f"设置目标FPS: {fps}")
        return True
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """设置帧回调函数
        
        Args:
            callback: 接收np.ndarray图像的回调函数
        """
        self.frame_callback = callback
        logger.info("设置帧回调函数")
    
    def start_capture(self) -> bool:
        """开始屏幕捕获
        
        Returns:
            bool: 启动成功返回True
        """
        if self.state == CaptureState.RUNNING:
            logger.warning("捕获已在运行中")
            return True
        
        if self.dxcam_instance is None:
            logger.error("DXcam未初始化，请先调用initialize()")
            return False
        
        try:
            self.state = CaptureState.RUNNING
            self.is_capturing = True
            self._stop_event.clear()
            
            # 启动捕获线程
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                name="ScreenCapture"
            )
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info("屏幕捕获已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动屏幕捕获失败: {e}")
            self.state = CaptureState.ERROR
            return False
    
    def stop_capture(self) -> None:
        """停止屏幕捕获"""
        if self.state != CaptureState.RUNNING:
            return
        
        logger.info("正在停止屏幕捕获...")
        self.is_capturing = False
        self._stop_event.set()
        
        # 等待捕获线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("捕获线程未在超时时间内结束")
        
        self.state = CaptureState.STOPPED
        logger.info("屏幕捕获已停止")
    
    def pause_capture(self) -> None:
        """暂停捕获"""
        if self.state == CaptureState.RUNNING:
            self.state = CaptureState.PAUSED
            logger.info("屏幕捕获已暂停")
    
    def resume_capture(self) -> None:
        """恢复捕获"""
        if self.state == CaptureState.PAUSED:
            self.state = CaptureState.RUNNING
            logger.info("屏幕捕获已恢复")
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """捕获单帧图像
        
        Returns:
            Optional[np.ndarray]: 捕获的图像，失败返回None
        """
        if self.dxcam_instance is None:
            logger.error("DXcam未初始化")
            return None
        
        try:
            if self.capture_region:
                region = self.capture_region.to_tuple()
                frame = self.dxcam_instance.grab(region=region)
            else:
                frame = self.dxcam_instance.grab()
            
            if frame is not None:
                self.stats.frames_captured += 1
                self.stats.last_capture_time = datetime.now()
                
            return frame
            
        except Exception as e:
            logger.error(f"单帧捕获失败: {e}")
            self.stats.error_count += 1
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新捕获的帧
        
        Returns:
            Optional[np.ndarray]: 最新帧，无可用帧返回None
        """
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_capture_stats(self) -> CaptureStats:
        """获取捕获统计信息
        
        Returns:
            CaptureStats: 统计信息
        """
        return self.stats
    
    def get_state(self) -> CaptureState:
        """获取当前状态
        
        Returns:
            CaptureState: 当前状态
        """
        return self.state
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.info("正在清理屏幕捕获资源...")
        
        # 停止捕获
        self.stop_capture()
        
        # 释放DXcam资源
        if self.dxcam_instance is not None:
            try:
                self.dxcam_instance.release()
            except Exception as e:
                logger.warning(f"释放DXcam资源失败: {e}")
            finally:
                self.dxcam_instance = None
        
        # 清理状态
        self.latest_frame = None
        self.frame_callback = None
        self._frame_times.clear()
        
        logger.info("屏幕捕获资源清理完成")
    
    def _capture_loop(self) -> None:
        """捕获循环（在独立线程中运行）"""
        frame_interval = 1.0 / self.target_fps
        retry_count = 0
        consecutive_failures = 0
        last_success_time = time.time()
        start_time = time.time()
        
        logger.info(f"捕获循环启动，帧间隔: {frame_interval:.3f}s")
        
        while self.is_capturing and not self._stop_event.is_set():
            try:
                # 检查暂停状态
                if self.state == CaptureState.PAUSED:
                    time.sleep(0.1)
                    continue
                
                loop_start = time.time()
                
                # 检查是否需要自动恢复
                if consecutive_failures > 0 and self.auto_recovery:
                    if time.time() - last_success_time > 5.0:  # 5秒无成功捕获
                        logger.warning("检测到连续失败，尝试自动恢复...")
                        if self._attempt_recovery():
                            consecutive_failures = 0
                            last_success_time = time.time()
                
                # 捕获帧
                try:
                    frame = self.capture_single_frame()
                except CaptureError as ce:
                    logger.error(f"捕获异常: {ce}")
                    frame = None
                except Exception as e:
                    logger.error(f"未知捕获错误: {e}")
                    frame = None
                
                if frame is not None:
                    # 成功捕获
                    with self._lock:
                        self.latest_frame = frame
                    
                    # 调用回调函数
                    if self.frame_callback:
                        try:
                            self.frame_callback(frame)
                        except Exception as e:
                            logger.error(f"帧回调函数执行失败: {e}")
                            logger.debug(f"回调函数异常详情: {traceback.format_exc()}")
                    
                    # 重置失败计数
                    consecutive_failures = 0
                    retry_count = 0
                    last_success_time = time.time()
                    
                    # 更新性能统计
                    self._update_fps_stats(loop_start)
                    
                else:
                    # 处理捕获失败
                    consecutive_failures += 1
                    retry_count += 1
                    self.stats.frames_dropped += 1
                    
                    if consecutive_failures >= self.max_retry_attempts:
                        if self.auto_recovery:
                            logger.error(f"连续{self.max_retry_attempts}次捕获失败，尝试自动恢复...")
                            if self._attempt_recovery():
                                consecutive_failures = 0
                                continue
                        
                        logger.error(f"连续{self.max_retry_attempts}次捕获失败，停止捕获")
                        self.state = CaptureState.ERROR
                        break
                    
                    logger.warning(f"帧捕获失败，连续失败次数: {consecutive_failures}")
                    time.sleep(self.retry_delay)
                    continue
                
                # 控制帧率
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"捕获循环异常: {e}")
                logger.debug(f"异常详情: {traceback.format_exc()}")
                self.stats.error_count += 1
                consecutive_failures += 1
                time.sleep(self.retry_delay)
                
                # 如果异常过多，尝试恢复
                if consecutive_failures >= self.max_retry_attempts and self.auto_recovery:
                    if self._attempt_recovery():
                        consecutive_failures = 0
        
        # 更新总运行时间
        self.stats.total_runtime += time.time() - start_time
        logger.info(f"捕获循环结束，总运行时间: {self.stats.total_runtime:.2f}秒")
    
    def _attempt_recovery(self) -> bool:
        """尝试自动恢复捕获系统
        
        Returns:
            bool: 恢复成功返回True
        """
        logger.info("开始自动恢复流程...")
        
        try:
            # 清理当前实例
            if self.dxcam_instance is not None:
                try:
                    self.dxcam_instance.release()
                except:
                    pass
                self.dxcam_instance = None
            
            # 短暂等待
            time.sleep(1.0)
            
            # 重新检测显示器
            self._detect_displays()
            
            # 重新初始化
            if self.initialize():
                logger.info("自动恢复成功")
                return True
            else:
                logger.error("自动恢复失败：重新初始化失败")
                return False
                
        except Exception as e:
            logger.error(f"自动恢复过程中发生异常: {e}")
            return False
    
    def _update_fps_stats(self, frame_time: float) -> None:
        """更新FPS统计"""
        current_time = time.time()
        self._frame_times.append(current_time)
        
        # 保持最近1秒的帧时间记录
        cutoff_time = current_time - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff_time]
        
        # 每秒更新一次FPS
        if current_time - self._last_fps_update >= 1.0:
            self.stats.average_fps = len(self._frame_times)
            self._last_fps_update = current_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取详细的性能报告
        
        Returns:
            Dict[str, Any]: 性能报告数据
        """
        current_time = time.time()
        
        # 计算运行时间
        if hasattr(self, '_start_time'):
            uptime = current_time - self._start_time
        else:
            uptime = self.stats.total_runtime
        
        # 计算成功率
        total_attempts = self.stats.frames_captured + self.stats.frames_dropped
        success_rate = (self.stats.frames_captured / total_attempts * 100) if total_attempts > 0 else 0
        
        # 内存使用情况（如果可用的话）
        memory_info = {}
        try:
            import psutil
            process = psutil.Process()
            memory_info = {
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024
            }
        except ImportError:
            memory_info = {'memory_percent': -1, 'memory_mb': -1}
        
        return {
            'state': self.state.value,
            'uptime_seconds': uptime,
            'frames_captured': self.stats.frames_captured,
            'frames_dropped': self.stats.frames_dropped,
            'success_rate_percent': success_rate,
            'average_fps': self.stats.average_fps,
            'target_fps': self.target_fps,
            'error_count': self.stats.error_count,
            'retry_count': self.stats.retry_count,
            'last_capture': self.stats.last_capture_time.isoformat() if self.stats.last_capture_time else None,
            'display_count': len(self.available_displays),
            'current_display': self.output_idx,
            'capture_region': {
                'x': self.capture_region.x,
                'y': self.capture_region.y,
                'width': self.capture_region.width,
                'height': self.capture_region.height
            } if self.capture_region else None,
            'memory_usage': memory_info,
            'auto_recovery_enabled': self.auto_recovery
        }
    
    def set_auto_recovery(self, enabled: bool) -> None:
        """设置自动恢复功能
        
        Args:
            enabled: 是否启用自动恢复
        """
        self.auto_recovery = enabled
        logger.info(f"自动恢复功能已{'启用' if enabled else '禁用'}")
    
    def set_retry_settings(self, max_attempts: int, delay: float) -> bool:
        """设置重试参数
        
        Args:
            max_attempts: 最大重试次数 (1-10)
            delay: 重试延迟 (0.05-2.0秒)
            
        Returns:
            bool: 设置成功返回True
        """
        if not 1 <= max_attempts <= 10:
            logger.error("最大重试次数必须在1-10之间")
            return False
        
        if not 0.05 <= delay <= 2.0:
            logger.error("重试延迟必须在0.05-2.0秒之间")
            return False
        
        self.max_retry_attempts = max_attempts
        self.retry_delay = delay
        logger.info(f"重试设置已更新：最大重试次数={max_attempts}，延迟={delay}s")
        return True
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = CaptureStats()
        self._frame_times.clear()
        self._last_fps_update = time.time()
        logger.info("统计信息已重置")
    
    def is_healthy(self) -> bool:
        """检查捕获系统健康状态
        
        Returns:
            bool: 系统健康返回True
        """
        if self.state == CaptureState.ERROR:
            return False
        
        # 检查是否长时间无成功捕获
        if self.stats.last_capture_time:
            time_since_last = (datetime.now() - self.stats.last_capture_time).total_seconds()
            if time_since_last > 10.0:  # 超过10秒无捕获
                return False
        
        # 检查错误率
        if self.stats.frames_captured > 0:
            error_rate = self.stats.error_count / self.stats.frames_captured
            if error_rate > 0.1:  # 错误率超过10%
                return False
        
        return True
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息
        
        Returns:
            Dict[str, Any]: 诊断信息
        """
        diagnostics = {
            'system_info': {
                'platform': platform.system(),
                'dxcam_available': DXCAM_AVAILABLE,
                'win32_available': WIN32_AVAILABLE
            },
            'module_state': {
                'state': self.state.value,
                'is_capturing': self.is_capturing,
                'dxcam_instance': self.dxcam_instance is not None,
                'capture_thread_alive': self.capture_thread.is_alive() if self.capture_thread else False
            },
            'settings': {
                'target_fps': self.target_fps,
                'output_display': self.output_idx,
                'max_retry_attempts': self.max_retry_attempts,
                'retry_delay': self.retry_delay,
                'auto_recovery': self.auto_recovery
            },
            'health_check': {
                'is_healthy': self.is_healthy(),
                'last_error': getattr(self, '_last_error', None)
            }
        }
        
        return diagnostics
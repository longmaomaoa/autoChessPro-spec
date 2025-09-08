#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主应用程序类
"""

import sys
import signal
from typing import Optional

from chess_ai.config import ConfigManager
from chess_ai.core.board_state import BoardState
from chess_ai.ui import MainWindow
from chess_ai.vision import ScreenCaptureModule, BoardRecognitionModule
from chess_ai.ai_engine import AIEngineInterface
from chess_ai.data import DataManager
from chess_ai.utils import PerformanceManager


class Application:
    """主应用程序类"""
    
    def __init__(self, config_manager: ConfigManager):
        """初始化应用程序"""
        self.config = config_manager
        self.qt_app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        self.board_state = BoardState()
        
        # 核心模块
        self.screen_capture: Optional[ScreenCaptureModule] = None
        self.board_recognition: Optional[BoardRecognitionModule] = None
        self.ai_engine: Optional[AIEngineInterface] = None
        self.data_manager: Optional[DataManager] = None
        self.performance_manager = PerformanceManager()
        
        # 应用状态
        self.is_running = False
        self.is_analysis_enabled = False
        
        # 定时器
        self.capture_timer: Optional[QTimer] = None
        
        print("应用程序初始化完成")
    
    def initialize_modules(self) -> bool:
        """初始化所有模块"""
        try:
            logger.info("初始化应用模块...")
            
            # 初始化数据管理器
            self.data_manager = DataManager(self.config)
            if not self.data_manager.initialize():
                logger.error("数据管理器初始化失败")
                return False
            
            # 初始化屏幕捕获模块
            self.screen_capture = ScreenCaptureModule(self.config)
            if not self.screen_capture.initialize():
                logger.error("屏幕捕获模块初始化失败")
                return False
            
            # 初始化棋盘识别模块
            self.board_recognition = BoardRecognitionModule(self.config)
            if not self.board_recognition.initialize():
                logger.error("棋盘识别模块初始化失败")
                return False
            
            # 初始化AI引擎
            self.ai_engine = AIEngineInterface(self.config)
            if not self.ai_engine.initialize():
                logger.error("AI引擎初始化失败")
                return False
            
            logger.info("所有模块初始化成功")
            return True
            
        except Exception as e:
            logger.exception(f"模块初始化异常: {e}")
            return False
    
    def initialize_ui(self) -> bool:
        """初始化用户界面"""
        try:
            logger.info("初始化用户界面...")
            
            # 创建Qt应用
            self.qt_app = QApplication(sys.argv)
            self.qt_app.setApplicationName("中国象棋智能对弈助手")
            self.qt_app.setApplicationVersion("1.0.0")
            
            # 创建主窗口
            self.main_window = MainWindow(self)
            
            # 设置信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # 连接窗口信号
            self._connect_window_signals()
            
            # 设置定时器
            self._setup_timers()
            
            logger.info("用户界面初始化成功")
            return True
            
        except Exception as e:
            logger.exception(f"用户界面初始化异常: {e}")
            return False
    
    def _connect_window_signals(self) -> None:
        """连接窗口信号"""
        if self.main_window is None:
            return
        
        # 连接控制信号
        self.main_window.start_analysis_requested.connect(self.start_analysis)
        self.main_window.stop_analysis_requested.connect(self.stop_analysis)
        self.main_window.reset_board_requested.connect(self.reset_board)
        
        # 连接设置信号
        self.main_window.settings_changed.connect(self.on_settings_changed)
        
        logger.debug("窗口信号连接完成")
    
    def _setup_timers(self) -> None:
        """设置定时器"""
        if self.qt_app is None:
            return
        
        # 屏幕捕获定时器
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._on_capture_timer)
        
        # 性能监控定时器
        performance_timer = QTimer()
        performance_timer.timeout.connect(self._on_performance_timer)
        performance_timer.start(5000)  # 每5秒更新一次性能数据
        
        logger.debug("定时器设置完成")
    
    def run(self) -> int:
        """运行应用程序"""
        try:
            logger.info("启动应用程序")
            
            # 初始化模块
            if not self.initialize_modules():
                logger.error("模块初始化失败")
                return 1
            
            # 初始化用户界面
            if not self.initialize_ui():
                logger.error("用户界面初始化失败")
                return 1
            
            # 显示主窗口
            self.main_window.show()
            self.is_running = True
            
            logger.info("应用程序启动成功")
            
            # 运行Qt事件循环
            return self.qt_app.exec()
            
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            return 0
        except Exception as e:
            logger.exception(f"应用程序运行异常: {e}")
            return 1
        finally:
            self.cleanup()
    
    def start_analysis(self) -> None:
        """开始分析"""
        if self.is_analysis_enabled:
            return
        
        logger.info("开始棋局分析")
        
        try:
            # 启动屏幕捕获
            if self.screen_capture and not self.screen_capture.start_capture():
                logger.error("启动屏幕捕获失败")
                return
            
            # 启动定时器
            capture_interval = self.config.get("screen_capture.interval", 100)
            self.capture_timer.start(capture_interval)
            
            self.is_analysis_enabled = True
            
            # 通知UI状态变化
            if self.main_window:
                self.main_window.on_analysis_started()
            
            logger.info("棋局分析已启动")
            
        except Exception as e:
            logger.exception(f"启动分析异常: {e}")
    
    def stop_analysis(self) -> None:
        """停止分析"""
        if not self.is_analysis_enabled:
            return
        
        logger.info("停止棋局分析")
        
        try:
            # 停止定时器
            if self.capture_timer:
                self.capture_timer.stop()
            
            # 停止屏幕捕获
            if self.screen_capture:
                self.screen_capture.stop_capture()
            
            self.is_analysis_enabled = False
            
            # 通知UI状态变化
            if self.main_window:
                self.main_window.on_analysis_stopped()
            
            logger.info("棋局分析已停止")
            
        except Exception as e:
            logger.exception(f"停止分析异常: {e}")
    
    def reset_board(self) -> None:
        """重置棋盘"""
        logger.info("重置棋盘")
        
        try:
            # 创建新的棋局状态
            self.board_state = BoardState()
            
            # 清除历史数据
            if self.data_manager:
                self.data_manager.clear_current_game()
            
            # 通知UI更新
            if self.main_window:
                self.main_window.on_board_reset(self.board_state)
            
            logger.info("棋盘重置完成")
            
        except Exception as e:
            logger.exception(f"重置棋盘异常: {e}")
    
    def on_settings_changed(self, settings: dict) -> None:
        """设置变化处理"""
        logger.info("处理设置变化")
        
        try:
            # 更新配置
            self.config.update_config(settings)
            
            # 重新初始化需要更新的模块
            if "screen_capture" in settings and self.screen_capture:
                self.screen_capture.update_config(settings["screen_capture"])
            
            if "ai_engine" in settings and self.ai_engine:
                self.ai_engine.update_config(settings["ai_engine"])
            
            logger.info("设置更新完成")
            
        except Exception as e:
            logger.exception(f"设置更新异常: {e}")
    
    def _on_capture_timer(self) -> None:
        """屏幕捕获定时器处理"""
        if not self.is_analysis_enabled:
            return
        
        try:
            # 获取屏幕截图
            if self.screen_capture:
                frame = self.screen_capture.capture_frame()
                if frame is not None:
                    self._process_frame(frame)
                    
        except Exception as e:
            logger.exception(f"屏幕捕获处理异常: {e}")
    
    def _process_frame(self, frame) -> None:
        """处理捕获的帧"""
        try:
            # 识别棋盘状态
            if self.board_recognition:
                new_board_state = self.board_recognition.recognize_board(frame)
                
                if new_board_state:
                    # 检查是否有变化
                    if self._board_state_changed(new_board_state):
                        self._on_board_state_changed(new_board_state)
                        
        except Exception as e:
            logger.exception(f"帧处理异常: {e}")
    
    def _board_state_changed(self, new_state: BoardState) -> bool:
        """检查棋盘状态是否有变化"""
        # 简单比较FEN字符串
        current_fen = self.board_state.to_fen().split()[0]  # 只比较棋盘部分
        new_fen = new_state.to_fen().split()[0]
        
        return current_fen != new_fen
    
    def _on_board_state_changed(self, new_state: BoardState) -> None:
        """棋盘状态变化处理"""
        logger.info("检测到棋盘状态变化")
        
        try:
            # 更新当前状态
            self.board_state = new_state
            
            # 获取AI建议
            if self.ai_engine:
                suggestions = self.ai_engine.get_move_suggestions(new_state)
                
                # 更新UI
                if self.main_window:
                    self.main_window.on_board_state_updated(new_state, suggestions)
            
            # 保存到历史
            if self.data_manager:
                self.data_manager.save_board_state(new_state)
                
        except Exception as e:
            logger.exception(f"棋盘状态变化处理异常: {e}")
    
    def _on_performance_timer(self) -> None:
        """性能监控定时器处理"""
        try:
            performance_data = self.performance_manager.get_performance_stats()
            
            # 更新UI中的性能指示器
            if self.main_window:
                self.main_window.update_performance_stats(performance_data)
                
        except Exception as e:
            logger.exception(f"性能监控异常: {e}")
    
    def _signal_handler(self, signum, frame) -> None:
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，准备退出")
        self.quit()
    
    def quit(self) -> None:
        """退出应用程序"""
        logger.info("退出应用程序")
        
        try:
            # 停止分析
            self.stop_analysis()
            
            # 关闭主窗口
            if self.main_window:
                self.main_window.close()
            
            # 退出Qt应用
            if self.qt_app:
                self.qt_app.quit()
            
            self.is_running = False
            
        except Exception as e:
            logger.exception(f"退出应用程序异常: {e}")
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理应用程序资源")
        
        try:
            # 停止所有模块
            if self.screen_capture:
                self.screen_capture.cleanup()
            
            if self.board_recognition:
                self.board_recognition.cleanup()
            
            if self.ai_engine:
                self.ai_engine.cleanup()
            
            if self.data_manager:
                self.data_manager.cleanup()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.exception(f"资源清理异常: {e}")
    
    @property
    def current_board_state(self) -> BoardState:
        """获取当前棋盘状态"""
        return self.board_state
    
    def get_config(self) -> ConfigManager:
        """获取配置管理器"""
        return self.config
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        return self.performance_manager.get_performance_stats()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋智能对弈助手主窗口

提供完整的PyQt6图形用户界面，包括棋盘显示、AI分析面板、控制面板等
"""

import sys
import os
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QMenuBar, QMenu, QStatusBar, QToolBar, QLabel,
    QFrame, QPushButton, QMessageBox, QApplication,
    QFileDialog, QDialog, QTabWidget, QScrollArea, QGroupBox
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QTimer, QSize, QRect, QSettings, 
    QThread, QObject, pyqtSlot
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QFont, QPalette, QColor, QAction,
    QKeySequence, QCloseEvent, QResizeEvent
)

from chess_ai.config.config_manager import ConfigManager
from chess_ai.core.board_state import BoardState
from chess_ai.ai_engine.ai_engine_interface import MoveSuggestion, GameSituation
from chess_ai.ai_engine.position_evaluation import WinProbability


class WindowLayout:
    """窗口布局配置"""
    
    # 默认窗口尺寸
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 800
    MIN_WIDTH = 900
    MIN_HEIGHT = 600
    
    # 分割器比例
    MAIN_SPLITTER_RATIO = [2, 1]  # 左侧棋盘区域 : 右侧分析区域
    RIGHT_SPLITTER_RATIO = [3, 2]  # 上部AI分析 : 下部控制面板
    
    # 间距和边距
    MAIN_MARGIN = 8
    WIDGET_SPACING = 6
    GROUP_MARGIN = 10


class ThemeManager:
    """主题管理器"""
    
    def __init__(self):
        self.current_theme = "default"
        self.themes = {
            "default": {
                "background": "#f0f0f0",
                "text": "#333333",
                "accent": "#4a90e2",
                "success": "#5cb85c",
                "warning": "#f0ad4e", 
                "error": "#d9534f",
                "border": "#cccccc"
            },
            "dark": {
                "background": "#2b2b2b",
                "text": "#ffffff", 
                "accent": "#4a90e2",
                "success": "#5cb85c",
                "warning": "#f0ad4e",
                "error": "#d9534f",
                "border": "#555555"
            },
            "chess": {
                "background": "#8B4513",  # 棋盘色调
                "text": "#ffffff",
                "accent": "#FFD700",      # 金色
                "success": "#32CD32",
                "warning": "#FF8C00",
                "error": "#DC143C",
                "border": "#A0522D"
            }
        }
    
    def apply_theme(self, widget: QWidget, theme_name: str = None):
        """应用主题到组件"""
        if theme_name:
            self.current_theme = theme_name
        
        theme = self.themes.get(self.current_theme, self.themes["default"])
        
        style_sheet = f"""
        QMainWindow {{
            background-color: {theme['background']};
            color: {theme['text']};
        }}
        QFrame {{
            background-color: {theme['background']};
            border: 1px solid {theme['border']};
        }}
        QPushButton {{
            background-color: {theme['accent']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {self._lighten_color(theme['accent'])};
        }}
        QPushButton:pressed {{
            background-color: {self._darken_color(theme['accent'])};
        }}
        QPushButton:disabled {{
            background-color: {theme['border']};
            color: #999999;
        }}
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {theme['border']};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        QLabel {{
            color: {theme['text']};
        }}
        """
        
        widget.setStyleSheet(style_sheet)
    
    def _lighten_color(self, color: str) -> str:
        """颜色变亮"""
        # 简化实现
        return color.replace("#4a90e2", "#5ba0f2")
    
    def _darken_color(self, color: str) -> str:
        """颜色变暗"""  
        # 简化实现
        return color.replace("#4a90e2", "#3a80d2")


class MainWindow(QMainWindow):
    """中国象棋智能对弈助手主窗口
    
    负责整个应用程序的图形界面布局和交互逻辑
    """
    
    # 信号定义
    start_analysis_requested = pyqtSignal()
    stop_analysis_requested = pyqtSignal()
    reset_board_requested = pyqtSignal()
    settings_changed = pyqtSignal(dict)
    capture_area_selected = pyqtSignal(QRect)
    
    # 内部状态变化信号
    analysis_state_changed = pyqtSignal(bool)  # True=运行中, False=已停止
    board_state_changed = pyqtSignal(BoardState)
    ai_suggestion_received = pyqtSignal(list)  # List[MoveSuggestion]
    win_probability_updated = pyqtSignal(WinProbability)
    
    def __init__(self, config: Optional[ConfigManager] = None):
        super().__init__()
        
        self.config = config or ConfigManager()
        self.theme_manager = ThemeManager()
        self.settings = QSettings("ChessAI", "Assistant")
        
        # 应用状态
        self.is_analysis_running = False
        self.current_board_state: Optional[BoardState] = None
        self.current_suggestions: list = []
        self.current_win_probability: Optional[WinProbability] = None
        
        # 子组件引用
        self.central_widget: Optional[QWidget] = None
        self.main_splitter: Optional[QSplitter] = None
        self.right_splitter: Optional[QSplitter] = None
        
        # 导入子面板类
        from chess_ai.ui.board_display import BoardDisplayWidget
        from chess_ai.ui.analysis_panel import AnalysisPanelWidget
        from chess_ai.ui.control_panel import ControlPanelWidget
        from chess_ai.ui.status_bar import StatusBarWidget
        
        self.board_display: Optional[BoardDisplayWidget] = None
        self.analysis_panel: Optional[AnalysisPanelWidget] = None
        self.control_panel: Optional[ControlPanelWidget] = None
        self.status_bar_widget: Optional[StatusBarWidget] = None
        
        # 性能监控计时器
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_display)
        self.performance_timer.setInterval(1000)  # 每秒更新
        
        # 初始化界面
        self.init_ui()
        self.init_connections()
        self.load_settings()
        
        # 应用主题
        self.theme_manager.apply_theme(self)
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("中国象棋智能对弈助手 v1.0")
        self.setMinimumSize(WindowLayout.MIN_WIDTH, WindowLayout.MIN_HEIGHT)
        self.resize(WindowLayout.DEFAULT_WIDTH, WindowLayout.DEFAULT_HEIGHT)
        
        # 设置窗口图标
        icon_path = self._get_resource_path("icons/chess_ai.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建工具栏
        self._create_toolbar()
        
        # 创建中央组件
        self._create_central_widget()
        
        # 创建状态栏
        self._create_status_bar()
        
        # 设置初始状态
        self._update_ui_state()
    
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 新建分析
        new_action = QAction("新建分析(&N)", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setStatusTip("开始新的棋局分析")
        new_action.triggered.connect(self.reset_board_requested.emit)
        file_menu.addAction(new_action)
        
        # 保存分析
        save_action = QAction("保存分析(&S)", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("保存当前分析结果")
        save_action.triggered.connect(self._save_analysis)
        file_menu.addAction(save_action)
        
        # 加载分析
        load_action = QAction("加载分析(&O)", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.setStatusTip("加载历史分析结果")
        load_action.triggered.connect(self._load_analysis)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("退出应用程序")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 分析菜单
        analysis_menu = menubar.addMenu("分析(&A)")
        
        # 开始分析
        self.start_action = QAction("开始分析(&S)", self)
        self.start_action.setShortcut(QKeySequence("F5"))
        self.start_action.setStatusTip("开始实时棋局分析")
        self.start_action.triggered.connect(self.start_analysis_requested.emit)
        analysis_menu.addAction(self.start_action)
        
        # 停止分析
        self.stop_action = QAction("停止分析(&T)", self)
        self.stop_action.setShortcut(QKeySequence("F6"))
        self.stop_action.setStatusTip("停止实时分析")
        self.stop_action.triggered.connect(self.stop_analysis_requested.emit)
        analysis_menu.addAction(self.stop_action)
        
        # 设置菜单
        settings_menu = menubar.addMenu("设置(&S)")
        
        # 偏好设置
        preferences_action = QAction("偏好设置(&P)", self)
        preferences_action.setStatusTip("打开偏好设置")
        preferences_action.triggered.connect(self._show_preferences)
        settings_menu.addAction(preferences_action)
        
        # 主题选择
        theme_menu = settings_menu.addMenu("主题(&T)")
        
        default_theme = QAction("默认主题", self)
        default_theme.triggered.connect(lambda: self._change_theme("default"))
        theme_menu.addAction(default_theme)
        
        dark_theme = QAction("深色主题", self)
        dark_theme.triggered.connect(lambda: self._change_theme("dark"))
        theme_menu.addAction(dark_theme)
        
        chess_theme = QAction("棋盘主题", self)
        chess_theme.triggered.connect(lambda: self._change_theme("chess"))
        theme_menu.addAction(chess_theme)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.setStatusTip("关于象棋智能助手")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar("主工具栏")
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # 开始分析按钮
        self.start_button = QPushButton("开始分析")
        self.start_button.setIcon(self._get_icon("play.png"))
        self.start_button.clicked.connect(self.start_analysis_requested.emit)
        toolbar.addWidget(self.start_button)
        
        # 停止分析按钮
        self.stop_button = QPushButton("停止分析")
        self.stop_button.setIcon(self._get_icon("stop.png"))
        self.stop_button.clicked.connect(self.stop_analysis_requested.emit)
        toolbar.addWidget(self.stop_button)
        
        toolbar.addSeparator()
        
        # 重置棋盘按钮
        reset_button = QPushButton("重置棋盘")
        reset_button.setIcon(self._get_icon("reset.png"))
        reset_button.clicked.connect(self.reset_board_requested.emit)
        toolbar.addWidget(reset_button)
        
        toolbar.addSeparator()
        
        # 区域选择按钮
        area_button = QPushButton("选择区域")
        area_button.setIcon(self._get_icon("select.png"))
        area_button.clicked.connect(self._select_capture_area)
        toolbar.addWidget(area_button)
    
    def _create_central_widget(self):
        """创建中央组件"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(WindowLayout.MAIN_MARGIN, WindowLayout.MAIN_MARGIN,
                                      WindowLayout.MAIN_MARGIN, WindowLayout.MAIN_MARGIN)
        main_layout.setSpacing(WindowLayout.WIDGET_SPACING)
        
        # 主分割器 - 水平分割
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # 左侧：棋盘显示区域
        self._create_board_area()
        
        # 右侧分割器 - 垂直分割
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.right_splitter)
        
        # 右上：AI分析面板
        self._create_analysis_area()
        
        # 右下：控制面板
        self._create_control_area()
        
        # 设置分割器比例
        self.main_splitter.setSizes([
            WindowLayout.DEFAULT_WIDTH * WindowLayout.MAIN_SPLITTER_RATIO[0] // 3,
            WindowLayout.DEFAULT_WIDTH * WindowLayout.MAIN_SPLITTER_RATIO[1] // 3
        ])
        
        self.right_splitter.setSizes([
            WindowLayout.DEFAULT_HEIGHT * WindowLayout.RIGHT_SPLITTER_RATIO[0] // 5,
            WindowLayout.DEFAULT_HEIGHT * WindowLayout.RIGHT_SPLITTER_RATIO[1] // 5
        ])
    
    def _create_board_area(self):
        """创建棋盘显示区域"""
        from chess_ai.ui.board_display import BoardDisplayWidget
        
        board_frame = QFrame()
        board_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(WindowLayout.GROUP_MARGIN, WindowLayout.GROUP_MARGIN,
                                       WindowLayout.GROUP_MARGIN, WindowLayout.GROUP_MARGIN)
        
        # 标题
        board_title = QLabel("棋盘监控")
        board_title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        board_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        board_layout.addWidget(board_title)
        
        # 棋盘显示组件
        self.board_display = BoardDisplayWidget()
        board_layout.addWidget(self.board_display)
        
        self.main_splitter.addWidget(board_frame)
    
    def _create_analysis_area(self):
        """创建AI分析区域"""
        from chess_ai.ui.analysis_panel import AnalysisPanelWidget
        
        analysis_frame = QFrame()
        analysis_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        analysis_layout = QVBoxLayout(analysis_frame)
        analysis_layout.setContentsMargins(WindowLayout.GROUP_MARGIN, WindowLayout.GROUP_MARGIN,
                                          WindowLayout.GROUP_MARGIN, WindowLayout.GROUP_MARGIN)
        
        # 标题
        analysis_title = QLabel("AI智能分析")
        analysis_title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        analysis_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        analysis_layout.addWidget(analysis_title)
        
        # AI分析面板
        self.analysis_panel = AnalysisPanelWidget()
        analysis_layout.addWidget(self.analysis_panel)
        
        self.right_splitter.addWidget(analysis_frame)
    
    def _create_control_area(self):
        """创建控制面板区域"""
        from chess_ai.ui.control_panel import ControlPanelWidget
        
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(WindowLayout.GROUP_MARGIN, WindowLayout.GROUP_MARGIN,
                                         WindowLayout.GROUP_MARGIN, WindowLayout.GROUP_MARGIN)
        
        # 标题
        control_title = QLabel("控制面板")
        control_title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        control_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(control_title)
        
        # 控制面板
        self.control_panel = ControlPanelWidget()
        control_layout.addWidget(self.control_panel)
        
        self.right_splitter.addWidget(control_frame)
    
    def _create_status_bar(self):
        """创建状态栏"""
        from chess_ai.ui.status_bar import StatusBarWidget
        
        self.status_bar_widget = StatusBarWidget()
        self.setStatusBar(self.status_bar_widget)
    
    def init_connections(self):
        """初始化信号连接"""
        # 内部状态变化
        self.analysis_state_changed.connect(self._update_ui_state)
        
        # 子组件连接 (延迟连接，等待子组件加载)
        QTimer.singleShot(100, self._connect_child_widgets)
    
    def _connect_child_widgets(self):
        """连接子组件信号"""
        if self.board_display:
            self.board_state_changed.connect(self.board_display.update_board_state)
        
        if self.analysis_panel:
            self.ai_suggestion_received.connect(self.analysis_panel.update_suggestions)
            self.win_probability_updated.connect(self.analysis_panel.update_win_probability)
        
        if self.control_panel:
            # 控制面板信号连接
            pass
        
        if self.status_bar_widget:
            self.analysis_state_changed.connect(self.status_bar_widget.update_analysis_status)
    
    def load_settings(self):
        """加载设置"""
        # 窗口几何
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # 窗口状态
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # 主题设置
        theme = self.settings.value("theme", "default")
        self.theme_manager.apply_theme(self, theme)
    
    def save_settings(self):
        """保存设置"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("theme", self.theme_manager.current_theme)
    
    # 公共方法 - 外部状态更新
    @pyqtSlot()
    def on_analysis_started(self):
        """分析开始回调"""
        self.is_analysis_running = True
        self.analysis_state_changed.emit(True)
        self.performance_timer.start()
        
        if self.status_bar_widget:
            self.status_bar_widget.show_message("AI分析已启动", 2000)
    
    @pyqtSlot()
    def on_analysis_stopped(self):
        """分析停止回调"""
        self.is_analysis_running = False
        self.analysis_state_changed.emit(False)
        self.performance_timer.stop()
        
        if self.status_bar_widget:
            self.status_bar_widget.show_message("AI分析已停止", 2000)
    
    @pyqtSlot(BoardState)
    def on_board_reset(self, board_state: BoardState):
        """棋盘重置回调"""
        self.current_board_state = board_state
        self.current_suggestions = []
        self.current_win_probability = None
        
        self.board_state_changed.emit(board_state)
        
        if self.status_bar_widget:
            self.status_bar_widget.show_message("棋盘已重置", 2000)
    
    @pyqtSlot(BoardState, list)
    def on_board_state_updated(self, board_state: BoardState, suggestions: list):
        """棋盘状态更新回调"""
        self.current_board_state = board_state
        self.current_suggestions = suggestions
        
        self.board_state_changed.emit(board_state)
        self.ai_suggestion_received.emit(suggestions)
    
    @pyqtSlot(WinProbability)
    def on_win_probability_updated(self, win_probability: WinProbability):
        """胜率更新回调"""
        self.current_win_probability = win_probability
        self.win_probability_updated.emit(win_probability)
    
    @pyqtSlot(dict)
    def update_performance_stats(self, stats: dict):
        """更新性能统计"""
        if self.status_bar_widget:
            self.status_bar_widget.update_performance_stats(stats)
    
    # 内部方法
    def _update_ui_state(self):
        """更新UI状态"""
        # 更新按钮状态
        if hasattr(self, 'start_button'):
            self.start_button.setEnabled(not self.is_analysis_running)
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(self.is_analysis_running)
        
        # 更新菜单动作状态
        if hasattr(self, 'start_action'):
            self.start_action.setEnabled(not self.is_analysis_running)
        if hasattr(self, 'stop_action'):
            self.stop_action.setEnabled(self.is_analysis_running)
    
    def _get_resource_path(self, relative_path: str) -> str:
        """获取资源文件路径"""
        # 简化实现 - 实际需要更复杂的资源管理
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, "..", "..", "..", "resources", relative_path)
    
    def _get_icon(self, icon_name: str) -> QIcon:
        """获取图标"""
        icon_path = self._get_resource_path(f"icons/{icon_name}")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        else:
            # 返回默认图标
            return QIcon()
    
    @pyqtSlot()
    def _save_analysis(self):
        """保存分析结果"""
        if not self.current_board_state:
            QMessageBox.warning(self, "警告", "没有可保存的分析数据")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分析结果", "", "JSON文件 (*.json)"
        )
        
        if file_path:
            # TODO: 实现保存逻辑
            QMessageBox.information(self, "信息", f"分析结果已保存到: {file_path}")
    
    @pyqtSlot()
    def _load_analysis(self):
        """加载分析结果"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载分析结果", "", "JSON文件 (*.json)"
        )
        
        if file_path:
            # TODO: 实现加载逻辑
            QMessageBox.information(self, "信息", f"已加载分析结果: {file_path}")
    
    @pyqtSlot()
    def _show_preferences(self):
        """显示偏好设置"""
        # TODO: 实现设置对话框
        QMessageBox.information(self, "信息", "偏好设置功能将在后续版本中实现")
    
    @pyqtSlot()
    def _select_capture_area(self):
        """选择捕获区域"""
        # TODO: 实现区域选择
        QMessageBox.information(self, "信息", "区域选择功能将在后续版本中实现")
    
    def _change_theme(self, theme_name: str):
        """更改主题"""
        self.theme_manager.apply_theme(self, theme_name)
        if self.status_bar_widget:
            self.status_bar_widget.show_message(f"已切换到{theme_name}主题", 2000)
    
    @pyqtSlot()
    def _show_about(self):
        """显示关于信息"""
        QMessageBox.about(self, "关于象棋智能助手", 
                         "中国象棋智能对弈助手 v1.0\n\n"
                         "基于计算机视觉和AI引擎的智能象棋分析工具\n\n"
                         "主要功能:\n"
                         "• 实时棋局识别\n"
                         "• AI走法建议\n"
                         "• 胜率分析\n"
                         "• 局面评估\n\n"
                         "Copyright © 2025 Chess AI Team")
    
    @pyqtSlot()
    def update_performance_display(self):
        """更新性能显示"""
        # 定期更新性能统计显示
        if self.is_analysis_running and self.status_bar_widget:
            # TODO: 获取真实性能数据
            fake_stats = {
                "fps": 30,
                "memory_mb": 256,
                "cpu_percent": 45
            }
            self.status_bar_widget.update_performance_stats(fake_stats)
    
    # 事件处理
    def closeEvent(self, event: QCloseEvent):
        """窗口关闭事件"""
        # 保存设置
        self.save_settings()
        
        # 如果分析正在运行，询问是否确认关闭
        if self.is_analysis_running:
            reply = QMessageBox.question(
                self, "确认退出",
                "AI分析正在运行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            else:
                # 发出停止分析信号
                self.stop_analysis_requested.emit()
        
        # 停止计时器
        if self.performance_timer.isActive():
            self.performance_timer.stop()
        
        event.accept()
    
    def resizeEvent(self, event: QResizeEvent):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 调整分割器比例
        if self.main_splitter:
            total_width = self.width() - 2 * WindowLayout.MAIN_MARGIN
            left_width = total_width * WindowLayout.MAIN_SPLITTER_RATIO[0] // 3
            right_width = total_width * WindowLayout.MAIN_SPLITTER_RATIO[1] // 3
            self.main_splitter.setSizes([left_width, right_width])
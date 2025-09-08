#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置界面和用户偏好管理

该模块提供完整的设置界面和用户偏好管理功能，包括：
- 综合设置界面（引擎设置、界面设置、高级选项）
- 用户偏好配置和持久化存储
- 实时设置预览和应用
- 设置导入导出功能
- 主题和界面个性化设置
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QPushButton, QGroupBox, QGridLayout,
    QScrollArea, QTextEdit, QLineEdit, QFileDialog,
    QColorDialog, QFontDialog, QMessageBox, QSplitter,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QButtonGroup, QRadioButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QTimer, QRect
from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
import sys
from pathlib import Path

class SettingsCategory(Enum):
    """设置类别枚举"""
    ENGINE = "engine"
    INTERFACE = "interface"
    ADVANCED = "advanced"
    THEME = "theme"
    HOTKEYS = "hotkeys"
    EXPORT = "export"

@dataclass
class EngineSettings:
    """引擎设置数据类"""
    engine_path: str = ""
    hash_size_mb: int = 128
    threads: int = 1
    analysis_depth: int = 15
    analysis_time: float = 5.0
    multipv: int = 3
    contempt: int = 0
    book_enabled: bool = True
    book_path: str = ""
    log_enabled: bool = False
    log_level: str = "INFO"

@dataclass
class InterfaceSettings:
    """界面设置数据类"""
    language: str = "zh_CN"
    auto_save: bool = True
    save_interval: int = 300
    show_coordinates: bool = True
    show_move_hints: bool = True
    show_last_move: bool = True
    animation_enabled: bool = True
    animation_speed: float = 1.0
    sound_enabled: bool = True
    sound_volume: float = 0.5
    confirm_moves: bool = False
    auto_rotate_board: bool = False

@dataclass
class ThemeSettings:
    """主题设置数据类"""
    theme_name: str = "default"
    board_style: str = "wood"
    piece_style: str = "traditional"
    font_family: str = "Microsoft YaHei"
    font_size: int = 12
    window_opacity: float = 1.0
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        "background": "#f0f0f0",
        "board_light": "#f4d03f",
        "board_dark": "#d4ac0d",
        "text": "#2c3e50",
        "accent": "#3498db"
    })

@dataclass
class HotkeySettings:
    """热键设置数据类"""
    new_game: str = "Ctrl+N"
    open_file: str = "Ctrl+O"
    save_file: str = "Ctrl+S"
    undo_move: str = "Ctrl+Z"
    redo_move: str = "Ctrl+Y"
    flip_board: str = "F"
    start_engine: str = "Space"
    stop_engine: str = "Esc"
    settings: str = "Ctrl+,"
    fullscreen: str = "F11"

@dataclass
class UserPreferences:
    """用户偏好设置综合数据类"""
    engine: EngineSettings = field(default_factory=EngineSettings)
    interface: InterfaceSettings = field(default_factory=InterfaceSettings)
    theme: ThemeSettings = field(default_factory=ThemeSettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    version: str = "1.0.0"
    last_updated: float = field(default_factory=lambda: __import__('time').time())

class SettingsValidator:
    """设置验证器"""
    
    @staticmethod
    def validate_engine_settings(settings: EngineSettings) -> List[str]:
        """验证引擎设置"""
        errors = []
        
        if settings.hash_size_mb < 1 or settings.hash_size_mb > 2048:
            errors.append("哈希表大小必须在1-2048MB之间")
            
        if settings.threads < 1 or settings.threads > 64:
            errors.append("线程数必须在1-64之间")
            
        if settings.analysis_depth < 1 or settings.analysis_depth > 50:
            errors.append("分析深度必须在1-50之间")
            
        if settings.analysis_time < 0.1 or settings.analysis_time > 300:
            errors.append("分析时间必须在0.1-300秒之间")
            
        if settings.multipv < 1 or settings.multipv > 10:
            errors.append("变化数量必须在1-10之间")
            
        return errors
    
    @staticmethod
    def validate_interface_settings(settings: InterfaceSettings) -> List[str]:
        """验证界面设置"""
        errors = []
        
        if settings.save_interval < 10 or settings.save_interval > 3600:
            errors.append("保存间隔必须在10-3600秒之间")
            
        if settings.animation_speed < 0.1 or settings.animation_speed > 5.0:
            errors.append("动画速度必须在0.1-5.0倍速之间")
            
        if settings.sound_volume < 0.0 or settings.sound_volume > 1.0:
            errors.append("音量必须在0.0-1.0之间")
            
        return errors

class ColorPickerButton(QPushButton):
    """颜色选择按钮组件"""
    
    color_changed = pyqtSignal(str, QColor)
    
    def __init__(self, color_name: str, initial_color: str, parent=None):
        super().__init__(parent)
        self.color_name = color_name
        self.current_color = QColor(initial_color)
        
        self.setText(f"  {color_name}  ")
        self.update_button_style()
        self.clicked.connect(self._select_color)
    
    def update_button_style(self):
        """更新按钮样式"""
        rgb = self.current_color.name()
        style = f"""
        QPushButton {{
            background-color: {rgb};
            border: 2px solid #666;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
            color: {'white' if self.current_color.lightness() < 128 else 'black'};
        }}
        QPushButton:hover {{
            border: 2px solid #333;
        }}
        """
        self.setStyleSheet(style)
    
    def _select_color(self):
        """选择颜色"""
        color = QColorDialog.getColor(self.current_color, self)
        if color.isValid():
            self.current_color = color
            self.update_button_style()
            self.color_changed.emit(self.color_name, color)
    
    def set_color(self, color: Union[str, QColor]):
        """设置颜色"""
        if isinstance(color, str):
            self.current_color = QColor(color)
        else:
            self.current_color = color
        self.update_button_style()

class FontPickerButton(QPushButton):
    """字体选择按钮组件"""
    
    font_changed = pyqtSignal(QFont)
    
    def __init__(self, initial_font: QFont, parent=None):
        super().__init__(parent)
        self.current_font = initial_font
        
        self.update_button_text()
        self.clicked.connect(self._select_font)
    
    def update_button_text(self):
        """更新按钮文本"""
        text = f"{self.current_font.family()}, {self.current_font.pointSize()}pt"
        if self.current_font.bold():
            text += ", 粗体"
        if self.current_font.italic():
            text += ", 斜体"
        self.setText(text)
        self.setFont(self.current_font)
    
    def _select_font(self):
        """选择字体"""
        font, ok = QFontDialog.getFont(self.current_font, self)
        if ok:
            self.current_font = font
            self.update_button_text()
            self.font_changed.emit(font)
    
    def set_font(self, font: QFont):
        """设置字体"""
        self.current_font = font
        self.update_button_text()

class SettingsPanel(QWidget):
    """设置面板主界面"""
    
    # 信号定义
    settings_changed = pyqtSignal(UserPreferences)
    settings_applied = pyqtSignal(UserPreferences)
    settings_reset = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._preferences: UserPreferences = UserPreferences()
        self._temp_preferences: UserPreferences = UserPreferences()
        self._settings_file = Path("config/user_preferences.json")
        self._setup_ui()
        self._load_settings()
        self._setup_connections()
    
    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页控件
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 创建各个设置页面
        self._create_engine_tab()
        self._create_interface_tab()
        self._create_theme_tab()
        self._create_hotkeys_tab()
        self._create_advanced_tab()
        
        # 底部按钮区域
        button_layout = self._create_button_area()
        layout.addLayout(button_layout)
    
    def _create_engine_tab(self) -> QWidget:
        """创建引擎设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 引擎路径设置
        engine_group = QGroupBox("引擎配置")
        engine_layout = QGridLayout(engine_group)
        
        engine_layout.addWidget(QLabel("引擎路径:"), 0, 0)
        self.engine_path_edit = QLineEdit()
        engine_layout.addWidget(self.engine_path_edit, 0, 1)
        self.engine_browse_btn = QPushButton("浏览...")
        self.engine_browse_btn.clicked.connect(self._browse_engine_path)
        engine_layout.addWidget(self.engine_browse_btn, 0, 2)
        
        scroll_layout.addWidget(engine_group)
        
        # 引擎参数设置
        params_group = QGroupBox("引擎参数")
        params_layout = QGridLayout(params_group)
        
        # 哈希表大小
        params_layout.addWidget(QLabel("哈希表大小:"), 0, 0)
        self.hash_size_spin = QSpinBox()
        self.hash_size_spin.setRange(1, 2048)
        self.hash_size_spin.setSuffix(" MB")
        params_layout.addWidget(self.hash_size_spin, 0, 1)
        
        # 线程数
        params_layout.addWidget(QLabel("线程数:"), 1, 0)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        params_layout.addWidget(self.threads_spin, 1, 1)
        
        # 分析深度
        params_layout.addWidget(QLabel("分析深度:"), 2, 0)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 50)
        params_layout.addWidget(self.depth_spin, 2, 1)
        
        # 分析时间
        params_layout.addWidget(QLabel("分析时间:"), 3, 0)
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.1, 300.0)
        self.time_spin.setSuffix(" 秒")
        params_layout.addWidget(self.time_spin, 3, 1)
        
        # 变化数量
        params_layout.addWidget(QLabel("变化数量:"), 4, 0)
        self.multipv_spin = QSpinBox()
        self.multipv_spin.setRange(1, 10)
        params_layout.addWidget(self.multipv_spin, 4, 1)
        
        # 藐视值
        params_layout.addWidget(QLabel("藐视值:"), 5, 0)
        self.contempt_spin = QSpinBox()
        self.contempt_spin.setRange(-200, 200)
        params_layout.addWidget(self.contempt_spin, 5, 1)
        
        scroll_layout.addWidget(params_group)
        
        # 开局库设置
        book_group = QGroupBox("开局库")
        book_layout = QGridLayout(book_group)
        
        self.book_enabled_check = QCheckBox("启用开局库")
        book_layout.addWidget(self.book_enabled_check, 0, 0)
        
        book_layout.addWidget(QLabel("开局库路径:"), 1, 0)
        self.book_path_edit = QLineEdit()
        book_layout.addWidget(self.book_path_edit, 1, 1)
        self.book_browse_btn = QPushButton("浏览...")
        self.book_browse_btn.clicked.connect(self._browse_book_path)
        book_layout.addWidget(self.book_browse_btn, 1, 2)
        
        scroll_layout.addWidget(book_group)
        
        # 日志设置
        log_group = QGroupBox("日志设置")
        log_layout = QGridLayout(log_group)
        
        self.log_enabled_check = QCheckBox("启用日志")
        log_layout.addWidget(self.log_enabled_check, 0, 0)
        
        log_layout.addWidget(QLabel("日志级别:"), 1, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        log_layout.addWidget(self.log_level_combo, 1, 1)
        
        scroll_layout.addWidget(log_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(widget, "引擎设置")
        return widget
    
    def _create_interface_tab(self) -> QWidget:
        """创建界面设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 语言和区域设置
        locale_group = QGroupBox("语言和区域")
        locale_layout = QGridLayout(locale_group)
        
        locale_layout.addWidget(QLabel("界面语言:"), 0, 0)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["简体中文", "繁体中文", "English", "日本語"])
        locale_layout.addWidget(self.language_combo, 0, 1)
        
        scroll_layout.addWidget(locale_group)
        
        # 自动保存设置
        save_group = QGroupBox("自动保存")
        save_layout = QGridLayout(save_group)
        
        self.auto_save_check = QCheckBox("启用自动保存")
        save_layout.addWidget(self.auto_save_check, 0, 0)
        
        save_layout.addWidget(QLabel("保存间隔:"), 1, 0)
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(10, 3600)
        self.save_interval_spin.setSuffix(" 秒")
        save_layout.addWidget(self.save_interval_spin, 1, 1)
        
        scroll_layout.addWidget(save_group)
        
        # 棋盘显示设置
        board_group = QGroupBox("棋盘显示")
        board_layout = QGridLayout(board_group)
        
        self.show_coordinates_check = QCheckBox("显示坐标")
        board_layout.addWidget(self.show_coordinates_check, 0, 0)
        
        self.show_move_hints_check = QCheckBox("显示着法提示")
        board_layout.addWidget(self.show_move_hints_check, 1, 0)
        
        self.show_last_move_check = QCheckBox("显示上一步着法")
        board_layout.addWidget(self.show_last_move_check, 2, 0)
        
        self.auto_rotate_check = QCheckBox("自动旋转棋盘")
        board_layout.addWidget(self.auto_rotate_check, 3, 0)
        
        scroll_layout.addWidget(board_group)
        
        # 动画设置
        animation_group = QGroupBox("动画效果")
        animation_layout = QGridLayout(animation_group)
        
        self.animation_enabled_check = QCheckBox("启用动画")
        animation_layout.addWidget(self.animation_enabled_check, 0, 0)
        
        animation_layout.addWidget(QLabel("动画速度:"), 1, 0)
        self.animation_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.animation_speed_slider.setRange(1, 50)
        animation_layout.addWidget(self.animation_speed_slider, 1, 1)
        self.animation_speed_label = QLabel("1.0x")
        animation_layout.addWidget(self.animation_speed_label, 1, 2)
        
        scroll_layout.addWidget(animation_group)
        
        # 音效设置
        sound_group = QGroupBox("音效")
        sound_layout = QGridLayout(sound_group)
        
        self.sound_enabled_check = QCheckBox("启用音效")
        sound_layout.addWidget(self.sound_enabled_check, 0, 0)
        
        sound_layout.addWidget(QLabel("音量:"), 1, 0)
        self.sound_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.sound_volume_slider.setRange(0, 100)
        sound_layout.addWidget(self.sound_volume_slider, 1, 1)
        self.sound_volume_label = QLabel("50%")
        sound_layout.addWidget(self.sound_volume_label, 1, 2)
        
        scroll_layout.addWidget(sound_group)
        
        # 其他设置
        misc_group = QGroupBox("其他")
        misc_layout = QGridLayout(misc_group)
        
        self.confirm_moves_check = QCheckBox("着法确认")
        misc_layout.addWidget(self.confirm_moves_check, 0, 0)
        
        scroll_layout.addWidget(misc_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(widget, "界面设置")
        return widget
    
    def _create_theme_tab(self) -> QWidget:
        """创建主题设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 主题选择
        theme_group = QGroupBox("主题风格")
        theme_layout = QGridLayout(theme_group)
        
        theme_layout.addWidget(QLabel("主题:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["默认", "暗色", "象棋风"])
        theme_layout.addWidget(self.theme_combo, 0, 1)
        
        theme_layout.addWidget(QLabel("棋盘风格:"), 1, 0)
        self.board_style_combo = QComboBox()
        self.board_style_combo.addItems(["木质", "石材", "金属", "简约"])
        theme_layout.addWidget(self.board_style_combo, 1, 1)
        
        theme_layout.addWidget(QLabel("棋子风格:"), 2, 0)
        self.piece_style_combo = QComboBox()
        self.piece_style_combo.addItems(["传统", "现代", "艺术", "简约"])
        theme_layout.addWidget(self.piece_style_combo, 2, 1)
        
        scroll_layout.addWidget(theme_group)
        
        # 字体设置
        font_group = QGroupBox("字体设置")
        font_layout = QGridLayout(font_group)
        
        font_layout.addWidget(QLabel("界面字体:"), 0, 0)
        initial_font = QFont("Microsoft YaHei", 12)
        self.font_picker = FontPickerButton(initial_font)
        font_layout.addWidget(self.font_picker, 0, 1)
        
        scroll_layout.addWidget(font_group)
        
        # 颜色设置
        color_group = QGroupBox("颜色方案")
        color_layout = QGridLayout(color_group)
        
        self.color_buttons = {}
        colors = [
            ("背景色", "background", "#f0f0f0"),
            ("棋盘浅色", "board_light", "#f4d03f"),
            ("棋盘深色", "board_dark", "#d4ac0d"),
            ("文字颜色", "text", "#2c3e50"),
            ("强调色", "accent", "#3498db")
        ]
        
        for i, (name, key, default_color) in enumerate(colors):
            color_layout.addWidget(QLabel(name + ":"), i, 0)
            button = ColorPickerButton(name, default_color)
            button.color_changed.connect(self._on_color_changed)
            self.color_buttons[key] = button
            color_layout.addWidget(button, i, 1)
        
        scroll_layout.addWidget(color_group)
        
        # 窗口设置
        window_group = QGroupBox("窗口设置")
        window_layout = QGridLayout(window_group)
        
        window_layout.addWidget(QLabel("窗口透明度:"), 0, 0)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(100)
        window_layout.addWidget(self.opacity_slider, 0, 1)
        self.opacity_label = QLabel("100%")
        window_layout.addWidget(self.opacity_label, 0, 2)
        
        scroll_layout.addWidget(window_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(widget, "主题设置")
        return widget
    
    def _create_hotkeys_tab(self) -> QWidget:
        """创建热键设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 热键列表
        self.hotkey_tree = QTreeWidget()
        self.hotkey_tree.setHeaderLabels(["功能", "热键", "描述"])
        layout.addWidget(self.hotkey_tree)
        
        # 热键设置
        hotkeys_data = [
            ("新建游戏", "new_game", "Ctrl+N", "开始新的对局"),
            ("打开文件", "open_file", "Ctrl+O", "打开棋谱文件"),
            ("保存文件", "save_file", "Ctrl+S", "保存当前棋谱"),
            ("撤销着法", "undo_move", "Ctrl+Z", "撤销上一步着法"),
            ("重做着法", "redo_move", "Ctrl+Y", "重做下一步着法"),
            ("翻转棋盘", "flip_board", "F", "翻转棋盘视角"),
            ("启动引擎", "start_engine", "Space", "开始AI分析"),
            ("停止引擎", "stop_engine", "Esc", "停止AI分析"),
            ("打开设置", "settings", "Ctrl+,", "打开设置界面"),
            ("全屏模式", "fullscreen", "F11", "切换全屏模式")
        ]
        
        for func_name, key, hotkey, desc in hotkeys_data:
            item = QTreeWidgetItem(self.hotkey_tree)
            item.setText(0, func_name)
            item.setText(1, hotkey)
            item.setText(2, desc)
            item.setData(0, Qt.ItemDataRole.UserRole, key)
        
        # 重置按钮
        reset_hotkeys_btn = QPushButton("重置为默认")
        reset_hotkeys_btn.clicked.connect(self._reset_hotkeys)
        layout.addWidget(reset_hotkeys_btn)
        
        self.tab_widget.addTab(widget, "热键设置")
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        """创建高级设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 导入导出设置
        import_export_group = QGroupBox("导入/导出")
        import_export_layout = QVBoxLayout(import_export_group)
        
        export_btn = QPushButton("导出设置...")
        export_btn.clicked.connect(self._export_settings)
        import_export_layout.addWidget(export_btn)
        
        import_btn = QPushButton("导入设置...")
        import_btn.clicked.connect(self._import_settings)
        import_export_layout.addWidget(import_btn)
        
        scroll_layout.addWidget(import_export_group)
        
        # 重置设置
        reset_group = QGroupBox("重置")
        reset_layout = QVBoxLayout(reset_group)
        
        reset_all_btn = QPushButton("重置所有设置")
        reset_all_btn.clicked.connect(self._reset_all_settings)
        reset_all_btn.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        reset_layout.addWidget(reset_all_btn)
        
        scroll_layout.addWidget(reset_group)
        
        # 调试信息
        debug_group = QGroupBox("调试信息")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMaximumHeight(150)
        debug_layout.addWidget(self.debug_text)
        
        refresh_debug_btn = QPushButton("刷新调试信息")
        refresh_debug_btn.clicked.connect(self._refresh_debug_info)
        debug_layout.addWidget(refresh_debug_btn)
        
        scroll_layout.addWidget(debug_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(widget, "高级设置")
        return widget
    
    def _create_button_area(self) -> QHBoxLayout:
        """创建底部按钮区域"""
        layout = QHBoxLayout()
        
        layout.addStretch()
        
        # 预览按钮
        self.preview_btn = QPushButton("预览")
        self.preview_btn.clicked.connect(self._preview_settings)
        layout.addWidget(self.preview_btn)
        
        # 应用按钮
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self._apply_settings)
        layout.addWidget(self.apply_btn)
        
        # 确定按钮
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self._ok_settings)
        layout.addWidget(self.ok_btn)
        
        # 取消按钮
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self._cancel_settings)
        layout.addWidget(self.cancel_btn)
        
        return layout
    
    def _setup_connections(self):
        """设置信号连接"""
        # 动画速度滑块
        self.animation_speed_slider.valueChanged.connect(self._update_animation_speed_label)
        
        # 音量滑块
        self.sound_volume_slider.valueChanged.connect(self._update_sound_volume_label)
        
        # 透明度滑块
        self.opacity_slider.valueChanged.connect(self._update_opacity_label)
        
        # 字体选择
        self.font_picker.font_changed.connect(self._on_font_changed)
    
    def _browse_engine_path(self):
        """浏览引擎路径"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择引擎文件", "", "可执行文件 (*.exe);;所有文件 (*.*)"
        )
        if path:
            self.engine_path_edit.setText(path)
    
    def _browse_book_path(self):
        """浏览开局库路径"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择开局库文件", "", "开局库文件 (*.book *.bin);;所有文件 (*.*)"
        )
        if path:
            self.book_path_edit.setText(path)
    
    def _update_animation_speed_label(self, value: int):
        """更新动画速度标签"""
        speed = value / 10.0
        self.animation_speed_label.setText(f"{speed:.1f}x")
    
    def _update_sound_volume_label(self, value: int):
        """更新音量标签"""
        self.sound_volume_label.setText(f"{value}%")
    
    def _update_opacity_label(self, value: int):
        """更新透明度标签"""
        self.opacity_label.setText(f"{value}%")
    
    def _on_color_changed(self, name: str, color: QColor):
        """处理颜色变化"""
        # 实时预览颜色变化
        pass
    
    def _on_font_changed(self, font: QFont):
        """处理字体变化"""
        # 实时预览字体变化
        pass
    
    def _reset_hotkeys(self):
        """重置热键为默认值"""
        default_hotkeys = HotkeySettings()
        # TODO: 更新热键显示
        
    def _export_settings(self):
        """导出设置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出设置", "user_preferences.json", "JSON文件 (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(self._preferences), f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "导出成功", f"设置已导出到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法导出设置: {e}")
    
    def _import_settings(self):
        """导入设置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入设置", "", "JSON文件 (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证导入的数据
                # TODO: 添加数据验证逻辑
                
                # 应用导入的设置
                self._load_from_dict(data)
                self._update_ui_from_preferences()
                
                QMessageBox.information(self, "导入成功", f"设置已从文件导入: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导入失败", f"无法导入设置: {e}")
    
    def _reset_all_settings(self):
        """重置所有设置"""
        reply = QMessageBox.question(
            self, "确认重置", 
            "这将重置所有设置到默认值，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._preferences = UserPreferences()
            self._update_ui_from_preferences()
            QMessageBox.information(self, "重置完成", "所有设置已重置为默认值")
    
    def _refresh_debug_info(self):
        """刷新调试信息"""
        debug_info = []
        debug_info.append(f"Python版本: {sys.version}")
        debug_info.append(f"设置文件: {self._settings_file}")
        debug_info.append(f"配置版本: {self._preferences.version}")
        debug_info.append(f"上次更新: {self._preferences.last_updated}")
        
        self.debug_text.setText("\n".join(debug_info))
    
    def _preview_settings(self):
        """预览设置"""
        self._collect_settings()
        self.settings_changed.emit(self._temp_preferences)
    
    def _apply_settings(self):
        """应用设置"""
        self._collect_settings()
        errors = self._validate_settings()
        
        if errors:
            QMessageBox.warning(self, "设置错误", "\n".join(errors))
            return
        
        self._preferences = self._temp_preferences
        self._save_settings()
        self.settings_applied.emit(self._preferences)
        
    def _ok_settings(self):
        """确定设置"""
        self._apply_settings()
        self.close()
    
    def _cancel_settings(self):
        """取消设置"""
        self.close()
    
    def _collect_settings(self):
        """收集UI中的设置"""
        # 引擎设置
        engine = EngineSettings(
            engine_path=self.engine_path_edit.text(),
            hash_size_mb=self.hash_size_spin.value(),
            threads=self.threads_spin.value(),
            analysis_depth=self.depth_spin.value(),
            analysis_time=self.time_spin.value(),
            multipv=self.multipv_spin.value(),
            contempt=self.contempt_spin.value(),
            book_enabled=self.book_enabled_check.isChecked(),
            book_path=self.book_path_edit.text(),
            log_enabled=self.log_enabled_check.isChecked(),
            log_level=self.log_level_combo.currentText()
        )
        
        # 界面设置
        interface = InterfaceSettings(
            language=["zh_CN", "zh_TW", "en_US", "ja_JP"][self.language_combo.currentIndex()],
            auto_save=self.auto_save_check.isChecked(),
            save_interval=self.save_interval_spin.value(),
            show_coordinates=self.show_coordinates_check.isChecked(),
            show_move_hints=self.show_move_hints_check.isChecked(),
            show_last_move=self.show_last_move_check.isChecked(),
            animation_enabled=self.animation_enabled_check.isChecked(),
            animation_speed=self.animation_speed_slider.value() / 10.0,
            sound_enabled=self.sound_enabled_check.isChecked(),
            sound_volume=self.sound_volume_slider.value() / 100.0,
            confirm_moves=self.confirm_moves_check.isChecked(),
            auto_rotate_board=self.auto_rotate_check.isChecked()
        )
        
        # 主题设置
        color_scheme = {}
        for key, button in self.color_buttons.items():
            color_scheme[key] = button.current_color.name()
        
        theme = ThemeSettings(
            theme_name=["default", "dark", "chess"][self.theme_combo.currentIndex()],
            board_style=["wood", "stone", "metal", "minimal"][self.board_style_combo.currentIndex()],
            piece_style=["traditional", "modern", "artistic", "minimal"][self.piece_style_combo.currentIndex()],
            font_family=self.font_picker.current_font.family(),
            font_size=self.font_picker.current_font.pointSize(),
            window_opacity=self.opacity_slider.value() / 100.0,
            color_scheme=color_scheme
        )
        
        # TODO: 收集热键设置
        hotkeys = HotkeySettings()
        
        self._temp_preferences = UserPreferences(
            engine=engine,
            interface=interface,
            theme=theme,
            hotkeys=hotkeys
        )
    
    def _validate_settings(self) -> List[str]:
        """验证设置"""
        errors = []
        
        # 验证引擎设置
        engine_errors = SettingsValidator.validate_engine_settings(self._temp_preferences.engine)
        errors.extend(engine_errors)
        
        # 验证界面设置
        interface_errors = SettingsValidator.validate_interface_settings(self._temp_preferences.interface)
        errors.extend(interface_errors)
        
        return errors
    
    def _load_settings(self):
        """加载设置"""
        try:
            if self._settings_file.exists():
                with open(self._settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._load_from_dict(data)
        except Exception as e:
            print(f"加载设置失败: {e}")
        
        self._update_ui_from_preferences()
    
    def _load_from_dict(self, data: Dict[str, Any]):
        """从字典加载设置"""
        try:
            # TODO: 完善数据加载逻辑
            pass
        except Exception as e:
            print(f"从字典加载设置失败: {e}")
    
    def _save_settings(self):
        """保存设置"""
        try:
            self._settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self._preferences), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存设置失败: {e}")
    
    def _update_ui_from_preferences(self):
        """从偏好设置更新UI"""
        # 引擎设置
        self.engine_path_edit.setText(self._preferences.engine.engine_path)
        self.hash_size_spin.setValue(self._preferences.engine.hash_size_mb)
        self.threads_spin.setValue(self._preferences.engine.threads)
        self.depth_spin.setValue(self._preferences.engine.analysis_depth)
        self.time_spin.setValue(self._preferences.engine.analysis_time)
        self.multipv_spin.setValue(self._preferences.engine.multipv)
        self.contempt_spin.setValue(self._preferences.engine.contempt)
        self.book_enabled_check.setChecked(self._preferences.engine.book_enabled)
        self.book_path_edit.setText(self._preferences.engine.book_path)
        self.log_enabled_check.setChecked(self._preferences.engine.log_enabled)
        
        # 界面设置
        self.auto_save_check.setChecked(self._preferences.interface.auto_save)
        self.save_interval_spin.setValue(self._preferences.interface.save_interval)
        self.show_coordinates_check.setChecked(self._preferences.interface.show_coordinates)
        self.show_move_hints_check.setChecked(self._preferences.interface.show_move_hints)
        self.show_last_move_check.setChecked(self._preferences.interface.show_last_move)
        self.animation_enabled_check.setChecked(self._preferences.interface.animation_enabled)
        self.animation_speed_slider.setValue(int(self._preferences.interface.animation_speed * 10))
        self.sound_enabled_check.setChecked(self._preferences.interface.sound_enabled)
        self.sound_volume_slider.setValue(int(self._preferences.interface.sound_volume * 100))
        self.confirm_moves_check.setChecked(self._preferences.interface.confirm_moves)
        self.auto_rotate_check.setChecked(self._preferences.interface.auto_rotate_board)
        
        # 主题设置
        self.opacity_slider.setValue(int(self._preferences.theme.window_opacity * 100))
        
        # 更新颜色按钮
        for key, color in self._preferences.theme.color_scheme.items():
            if key in self.color_buttons:
                self.color_buttons[key].set_color(color)
        
        # 更新字体
        font = QFont(self._preferences.theme.font_family, self._preferences.theme.font_size)
        self.font_picker.set_font(font)
    
    def get_preferences(self) -> UserPreferences:
        """获取用户偏好设置"""
        return self._preferences
    
    def set_preferences(self, preferences: UserPreferences):
        """设置用户偏好"""
        self._preferences = preferences
        self._update_ui_from_preferences()

# 向后兼容的别名
ControlPanel = SettingsPanel
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置界面测试
"""

import sys
import json
import tempfile
from typing import Dict, Any
from pathlib import Path

# Mock PyQt6 for testing without installation
class MockQWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self._text = ""
        self._value = 0
        self._checked = False
        self._current_index = 0
        self._items = []
        
    def setText(self, text):
        self._text = text
        
    def text(self):
        return self._text
        
    def setValue(self, value):
        self._value = value
        
    def value(self):
        return self._value
        
    def setChecked(self, checked):
        self._checked = checked
        
    def isChecked(self):
        return self._checked
        
    def setCurrentIndex(self, index):
        self._current_index = index
        
    def currentIndex(self):
        return self._current_index
        
    def currentText(self):
        if self._current_index < len(self._items):
            return self._items[self._current_index]
        return ""
        
    def addItems(self, items):
        self._items = items

class MockQt:
    class AlignmentFlag:
        AlignCenter = 1
    class ItemDataRole:
        UserRole = 1
    class Orientation:
        Horizontal = 1
    class StandardButton:
        Yes = 1
        No = 2

class MockQColor:
    def __init__(self, color_str="#ffffff"):
        self._color = color_str
        
    def name(self):
        return self._color
        
    def lightness(self):
        return 128

class MockQFont:
    def __init__(self, family="Arial", size=12):
        self._family = family
        self._size = size
        self._bold = False
        self._italic = False
        
    def family(self):
        return self._family
        
    def pointSize(self):
        return self._size
        
    def bold(self):
        return self._bold
        
    def italic(self):
        return self._italic

# Mock all PyQt6 imports
sys.modules['PyQt6'] = type(sys)('mock_pyqt6')
sys.modules['PyQt6.QtWidgets'] = type(sys)('mock_widgets')
sys.modules['PyQt6.QtCore'] = type(sys)('mock_core')
sys.modules['PyQt6.QtGui'] = type(sys)('mock_gui')

# Set up mock attributes
for widget_name in ['QWidget', 'QVBoxLayout', 'QHBoxLayout', 'QTabWidget',
                   'QLabel', 'QSlider', 'QSpinBox', 'QDoubleSpinBox', 'QCheckBox',
                   'QComboBox', 'QPushButton', 'QGroupBox', 'QGridLayout',
                   'QScrollArea', 'QTextEdit', 'QLineEdit', 'QFileDialog',
                   'QColorDialog', 'QFontDialog', 'QMessageBox', 'QSplitter',
                   'QListWidget', 'QListWidgetItem', 'QTreeWidget', 'QTreeWidgetItem',
                   'QProgressBar', 'QButtonGroup', 'QRadioButton', 'QFrame',
                   'QMainWindow', 'QMenuBar', 'QMenu', 'QStatusBar', 'QToolBar',
                   'QApplication', 'QDialog']:
    setattr(sys.modules['PyQt6.QtWidgets'], widget_name, MockQWidget)

setattr(sys.modules['PyQt6.QtCore'], 'Qt', MockQt)
setattr(sys.modules['PyQt6.QtCore'], 'pyqtSignal', lambda *args: lambda x: x)
setattr(sys.modules['PyQt6.QtCore'], 'QSettings', MockQWidget)
setattr(sys.modules['PyQt6.QtCore'], 'QTimer', MockQWidget)
setattr(sys.modules['PyQt6.QtCore'], 'QRect', MockQWidget)
setattr(sys.modules['PyQt6.QtCore'], 'QSize', MockQWidget)
setattr(sys.modules['PyQt6.QtCore'], 'QThread', MockQWidget)
setattr(sys.modules['PyQt6.QtCore'], 'QObject', MockQWidget)
setattr(sys.modules['PyQt6.QtCore'], 'pyqtSlot', lambda *args: lambda x: x)

setattr(sys.modules['PyQt6.QtGui'], 'QFont', MockQFont)
setattr(sys.modules['PyQt6.QtGui'], 'QColor', MockQColor)
setattr(sys.modules['PyQt6.QtGui'], 'QPalette', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QPixmap', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QIcon', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QPainter', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QPen', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QBrush', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QAction', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QKeySequence', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QCloseEvent', MockQWidget)
setattr(sys.modules['PyQt6.QtGui'], 'QResizeEvent', MockQWidget)

# Now import the actual modules after mocking
sys.path.insert(0, 'src')

from chess_ai.ui.control_panel import (
    SettingsCategory, EngineSettings, InterfaceSettings, 
    ThemeSettings, HotkeySettings, UserPreferences,
    SettingsValidator
)

def test_settings_category_enum():
    """测试设置类别枚举"""
    print("测试设置类别枚举...")
    
    # 验证枚举值
    assert SettingsCategory.ENGINE.value == "engine"
    assert SettingsCategory.INTERFACE.value == "interface"
    assert SettingsCategory.ADVANCED.value == "advanced"
    assert SettingsCategory.THEME.value == "theme"
    assert SettingsCategory.HOTKEYS.value == "hotkeys"
    assert SettingsCategory.EXPORT.value == "export"
    
    # 验证枚举数量
    assert len([c for c in SettingsCategory]) == 6
    
    print("PASS: 设置类别枚举测试")

def test_engine_settings():
    """测试引擎设置数据类"""
    print("测试引擎设置数据类...")
    
    # 测试默认值
    engine = EngineSettings()
    assert engine.engine_path == ""
    assert engine.hash_size_mb == 128
    assert engine.threads == 1
    assert engine.analysis_depth == 15
    assert engine.analysis_time == 5.0
    assert engine.multipv == 3
    assert engine.contempt == 0
    assert engine.book_enabled == True
    assert engine.book_path == ""
    assert engine.log_enabled == False
    assert engine.log_level == "INFO"
    
    # 测试赋值
    engine = EngineSettings(
        engine_path="/path/to/engine",
        hash_size_mb=256,
        threads=4,
        analysis_depth=20,
        analysis_time=10.0,
        multipv=5,
        contempt=50,
        book_enabled=False,
        book_path="/path/to/book",
        log_enabled=True,
        log_level="DEBUG"
    )
    
    assert engine.engine_path == "/path/to/engine"
    assert engine.hash_size_mb == 256
    assert engine.threads == 4
    assert engine.analysis_depth == 20
    assert engine.analysis_time == 10.0
    assert engine.multipv == 5
    assert engine.contempt == 50
    assert engine.book_enabled == False
    assert engine.book_path == "/path/to/book"
    assert engine.log_enabled == True
    assert engine.log_level == "DEBUG"
    
    print("PASS: 引擎设置数据类测试")

def test_interface_settings():
    """测试界面设置数据类"""
    print("测试界面设置数据类...")
    
    # 测试默认值
    interface = InterfaceSettings()
    assert interface.language == "zh_CN"
    assert interface.auto_save == True
    assert interface.save_interval == 300
    assert interface.show_coordinates == True
    assert interface.show_move_hints == True
    assert interface.show_last_move == True
    assert interface.animation_enabled == True
    assert interface.animation_speed == 1.0
    assert interface.sound_enabled == True
    assert interface.sound_volume == 0.5
    assert interface.confirm_moves == False
    assert interface.auto_rotate_board == False
    
    # 测试赋值
    interface = InterfaceSettings(
        language="en_US",
        auto_save=False,
        save_interval=600,
        show_coordinates=False,
        show_move_hints=False,
        show_last_move=False,
        animation_enabled=False,
        animation_speed=2.0,
        sound_enabled=False,
        sound_volume=0.8,
        confirm_moves=True,
        auto_rotate_board=True
    )
    
    assert interface.language == "en_US"
    assert interface.auto_save == False
    assert interface.save_interval == 600
    assert interface.show_coordinates == False
    assert interface.show_move_hints == False
    assert interface.show_last_move == False
    assert interface.animation_enabled == False
    assert interface.animation_speed == 2.0
    assert interface.sound_enabled == False
    assert interface.sound_volume == 0.8
    assert interface.confirm_moves == True
    assert interface.auto_rotate_board == True
    
    print("PASS: 界面设置数据类测试")

def test_theme_settings():
    """测试主题设置数据类"""
    print("测试主题设置数据类...")
    
    # 测试默认值
    theme = ThemeSettings()
    assert theme.theme_name == "default"
    assert theme.board_style == "wood"
    assert theme.piece_style == "traditional"
    assert theme.font_family == "Microsoft YaHei"
    assert theme.font_size == 12
    assert theme.window_opacity == 1.0
    assert isinstance(theme.color_scheme, dict)
    assert len(theme.color_scheme) == 5
    assert "background" in theme.color_scheme
    assert "board_light" in theme.color_scheme
    assert "board_dark" in theme.color_scheme
    assert "text" in theme.color_scheme
    assert "accent" in theme.color_scheme
    
    # 测试自定义值
    custom_colors = {
        "background": "#ffffff",
        "board_light": "#eeeeee",
        "board_dark": "#cccccc",
        "text": "#000000",
        "accent": "#ff0000"
    }
    
    theme = ThemeSettings(
        theme_name="dark",
        board_style="stone",
        piece_style="modern",
        font_family="Arial",
        font_size=14,
        window_opacity=0.8,
        color_scheme=custom_colors
    )
    
    assert theme.theme_name == "dark"
    assert theme.board_style == "stone"
    assert theme.piece_style == "modern"
    assert theme.font_family == "Arial"
    assert theme.font_size == 14
    assert theme.window_opacity == 0.8
    assert theme.color_scheme == custom_colors
    
    print("PASS: 主题设置数据类测试")

def test_hotkey_settings():
    """测试热键设置数据类"""
    print("测试热键设置数据类...")
    
    # 测试默认值
    hotkeys = HotkeySettings()
    assert hotkeys.new_game == "Ctrl+N"
    assert hotkeys.open_file == "Ctrl+O"
    assert hotkeys.save_file == "Ctrl+S"
    assert hotkeys.undo_move == "Ctrl+Z"
    assert hotkeys.redo_move == "Ctrl+Y"
    assert hotkeys.flip_board == "F"
    assert hotkeys.start_engine == "Space"
    assert hotkeys.stop_engine == "Esc"
    assert hotkeys.settings == "Ctrl+,"
    assert hotkeys.fullscreen == "F11"
    
    # 测试自定义值
    hotkeys = HotkeySettings(
        new_game="Ctrl+Shift+N",
        open_file="Ctrl+Alt+O",
        save_file="Ctrl+Shift+S",
        undo_move="Alt+Z",
        redo_move="Alt+Y",
        flip_board="Ctrl+F",
        start_engine="F5",
        stop_engine="F6",
        settings="Ctrl+P",
        fullscreen="F10"
    )
    
    assert hotkeys.new_game == "Ctrl+Shift+N"
    assert hotkeys.open_file == "Ctrl+Alt+O"
    assert hotkeys.save_file == "Ctrl+Shift+S"
    assert hotkeys.undo_move == "Alt+Z"
    assert hotkeys.redo_move == "Alt+Y"
    assert hotkeys.flip_board == "Ctrl+F"
    assert hotkeys.start_engine == "F5"
    assert hotkeys.stop_engine == "F6"
    assert hotkeys.settings == "Ctrl+P"
    assert hotkeys.fullscreen == "F10"
    
    print("PASS: 热键设置数据类测试")

def test_user_preferences():
    """测试用户偏好设置综合数据类"""
    print("测试用户偏好设置综合数据类...")
    
    # 测试默认值
    prefs = UserPreferences()
    assert isinstance(prefs.engine, EngineSettings)
    assert isinstance(prefs.interface, InterfaceSettings)
    assert isinstance(prefs.theme, ThemeSettings)
    assert isinstance(prefs.hotkeys, HotkeySettings)
    assert prefs.version == "1.0.0"
    assert isinstance(prefs.last_updated, float)
    
    # 测试自定义值
    custom_engine = EngineSettings(engine_path="/custom/engine", threads=8)
    custom_interface = InterfaceSettings(language="en_US", sound_enabled=False)
    custom_theme = ThemeSettings(theme_name="dark", font_size=16)
    custom_hotkeys = HotkeySettings(new_game="F2", save_file="F3")
    
    prefs = UserPreferences(
        engine=custom_engine,
        interface=custom_interface,
        theme=custom_theme,
        hotkeys=custom_hotkeys,
        version="2.0.0"
    )
    
    assert prefs.engine.engine_path == "/custom/engine"
    assert prefs.engine.threads == 8
    assert prefs.interface.language == "en_US"
    assert prefs.interface.sound_enabled == False
    assert prefs.theme.theme_name == "dark"
    assert prefs.theme.font_size == 16
    assert prefs.hotkeys.new_game == "F2"
    assert prefs.hotkeys.save_file == "F3"
    assert prefs.version == "2.0.0"
    
    print("PASS: 用户偏好设置综合数据类测试")

def test_settings_validator():
    """测试设置验证器"""
    print("测试设置验证器...")
    
    # 测试引擎设置验证
    # 有效设置
    valid_engine = EngineSettings(
        hash_size_mb=128,
        threads=4,
        analysis_depth=15,
        analysis_time=5.0,
        multipv=3
    )
    errors = SettingsValidator.validate_engine_settings(valid_engine)
    assert len(errors) == 0
    
    # 无效设置
    invalid_engine = EngineSettings(
        hash_size_mb=3000,  # 超出范围
        threads=100,        # 超出范围
        analysis_depth=0,   # 小于最小值
        analysis_time=-1.0, # 负数
        multipv=20          # 超出范围
    )
    errors = SettingsValidator.validate_engine_settings(invalid_engine)
    assert len(errors) == 5
    
    # 测试界面设置验证
    # 有效设置
    valid_interface = InterfaceSettings(
        save_interval=300,
        animation_speed=1.0,
        sound_volume=0.5
    )
    errors = SettingsValidator.validate_interface_settings(valid_interface)
    assert len(errors) == 0
    
    # 无效设置
    invalid_interface = InterfaceSettings(
        save_interval=5,      # 小于最小值
        animation_speed=10.0, # 超出范围
        sound_volume=2.0      # 超出范围
    )
    errors = SettingsValidator.validate_interface_settings(invalid_interface)
    assert len(errors) == 3
    
    print("PASS: 设置验证器测试")

def test_data_serialization():
    """测试数据序列化"""
    print("测试数据序列化...")
    
    # 创建测试数据
    prefs = UserPreferences(
        engine=EngineSettings(engine_path="/test/engine", threads=2),
        interface=InterfaceSettings(language="en_US", auto_save=False),
        theme=ThemeSettings(theme_name="dark", font_size=14),
        hotkeys=HotkeySettings(new_game="F2")
    )
    
    # 测试序列化
    from dataclasses import asdict
    data = asdict(prefs)
    
    # 验证序列化结果
    assert isinstance(data, dict)
    assert "engine" in data
    assert "interface" in data
    assert "theme" in data
    assert "hotkeys" in data
    assert "version" in data
    assert "last_updated" in data
    
    # 验证嵌套数据
    assert data["engine"]["engine_path"] == "/test/engine"
    assert data["engine"]["threads"] == 2
    assert data["interface"]["language"] == "en_US"
    assert data["interface"]["auto_save"] == False
    assert data["theme"]["theme_name"] == "dark"
    assert data["theme"]["font_size"] == 14
    assert data["hotkeys"]["new_game"] == "F2"
    
    # 测试JSON序列化
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    assert isinstance(json_str, str)
    assert "/test/engine" in json_str
    assert "en_US" in json_str
    assert "dark" in json_str
    
    # 测试反序列化
    loaded_data = json.loads(json_str)
    assert loaded_data["engine"]["engine_path"] == "/test/engine"
    assert loaded_data["interface"]["language"] == "en_US"
    assert loaded_data["theme"]["theme_name"] == "dark"
    
    print("PASS: 数据序列化测试")

def test_settings_persistence():
    """测试设置持久化"""
    print("测试设置持久化...")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    try:
        # 创建测试数据
        prefs = UserPreferences(
            engine=EngineSettings(hash_size_mb=256, threads=4),
            interface=InterfaceSettings(language="zh_TW", sound_volume=0.8),
            theme=ThemeSettings(font_family="Arial", window_opacity=0.9)
        )
        
        # 保存设置
        from dataclasses import asdict
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(prefs), f, indent=2, ensure_ascii=False)
        
        # 验证文件存在
        assert temp_path.exists()
        
        # 加载设置
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        # 验证加载的数据
        assert loaded_data["engine"]["hash_size_mb"] == 256
        assert loaded_data["engine"]["threads"] == 4
        assert loaded_data["interface"]["language"] == "zh_TW"
        assert loaded_data["interface"]["sound_volume"] == 0.8
        assert loaded_data["theme"]["font_family"] == "Arial"
        assert loaded_data["theme"]["window_opacity"] == 0.9
        
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
    
    print("PASS: 设置持久化测试")

def test_edge_cases():
    """测试边界情况"""
    print("测试边界情况...")
    
    # 空字符串和None值
    engine = EngineSettings(engine_path="", book_path="")
    assert engine.engine_path == ""
    assert engine.book_path == ""
    
    # 边界值
    engine = EngineSettings(hash_size_mb=1, threads=1, analysis_depth=1)
    assert engine.hash_size_mb == 1
    assert engine.threads == 1
    assert engine.analysis_depth == 1
    
    # 最大值
    engine = EngineSettings(hash_size_mb=2048, threads=64, analysis_depth=50)
    assert engine.hash_size_mb == 2048
    assert engine.threads == 64
    assert engine.analysis_depth == 50
    
    # 浮点数边界
    interface = InterfaceSettings(animation_speed=0.1, sound_volume=0.0)
    assert interface.animation_speed == 0.1
    assert interface.sound_volume == 0.0
    
    interface = InterfaceSettings(animation_speed=5.0, sound_volume=1.0)
    assert interface.animation_speed == 5.0
    assert interface.sound_volume == 1.0
    
    # 空颜色方案
    theme = ThemeSettings(color_scheme={})
    assert len(theme.color_scheme) == 0
    
    print("PASS: 边界情况测试")

def test_complex_scenarios():
    """测试复杂场景"""
    print("测试复杂场景...")
    
    # 多语言设置场景
    languages = ["zh_CN", "zh_TW", "en_US", "ja_JP"]
    for lang in languages:
        interface = InterfaceSettings(language=lang)
        assert interface.language == lang
    
    # 多主题组合场景
    themes = ["default", "dark", "chess"]
    board_styles = ["wood", "stone", "metal", "minimal"]
    piece_styles = ["traditional", "modern", "artistic", "minimal"]
    
    for theme in themes:
        for board_style in board_styles:
            for piece_style in piece_styles:
                theme_settings = ThemeSettings(
                    theme_name=theme,
                    board_style=board_style,
                    piece_style=piece_style
                )
                assert theme_settings.theme_name == theme
                assert theme_settings.board_style == board_style
                assert theme_settings.piece_style == piece_style
    
    # 复杂引擎配置场景
    complex_engine = EngineSettings(
        engine_path="/very/long/path/to/some/chess/engine/executable.exe",
        hash_size_mb=1024,
        threads=16,
        analysis_depth=25,
        analysis_time=60.0,
        multipv=5,
        contempt=-50,
        book_enabled=True,
        book_path="/another/very/long/path/to/opening/book.book",
        log_enabled=True,
        log_level="DEBUG"
    )
    
    # 验证复杂配置
    assert len(complex_engine.engine_path) > 50
    assert complex_engine.hash_size_mb == 1024
    assert complex_engine.threads == 16
    assert complex_engine.analysis_depth == 25
    assert complex_engine.analysis_time == 60.0
    assert complex_engine.multipv == 5
    assert complex_engine.contempt == -50
    assert complex_engine.book_enabled == True
    assert len(complex_engine.book_path) > 40
    assert complex_engine.log_enabled == True
    assert complex_engine.log_level == "DEBUG"
    
    # 验证设置有效性
    errors = SettingsValidator.validate_engine_settings(complex_engine)
    assert len(errors) == 0
    
    print("PASS: 复杂场景测试")

def run_all_tests():
    """运行所有测试"""
    print("开始设置界面测试...")
    print("=" * 60)
    
    try:
        test_settings_category_enum()
        test_engine_settings()
        test_interface_settings()
        test_theme_settings()
        test_hotkey_settings()
        test_user_preferences()
        test_settings_validator()
        test_data_serialization()
        test_settings_persistence()
        test_edge_cases()
        test_complex_scenarios()
        
        print("=" * 60)
        print("测试结果: 11/11 通过")
        print("成功率: 100%")
        print("设置界面和用户偏好实现完成!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
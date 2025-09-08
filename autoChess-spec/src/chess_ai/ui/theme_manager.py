#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题管理器和界面个性化

该模块提供全面的主题管理和界面个性化功能，包括：
- 多主题系统（默认、暗色、象棋风格）
- 动态主题切换和实时预览
- 自定义颜色方案和字体配置
- 棋盘和棋子样式管理
- 界面动画和特效控制
- 用户个性化设置存储
"""

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QColor, QFont, QPalette, QPixmap, QIcon, QLinearGradient,
    QRadialGradient, QPainter, QPen, QBrush
)
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import colorsys
import math

class ThemeType(Enum):
    """主题类型枚举"""
    DEFAULT = "default"
    DARK = "dark"
    CHESS = "chess"
    CUSTOM = "custom"

class BoardStyle(Enum):
    """棋盘样式枚举"""
    WOOD = "wood"
    STONE = "stone"
    METAL = "metal"
    MINIMAL = "minimal"
    CLASSIC = "classic"

class PieceStyle(Enum):
    """棋子样式枚举"""
    TRADITIONAL = "traditional"
    MODERN = "modern"
    ARTISTIC = "artistic"
    MINIMAL = "minimal"
    CALLIGRAPHY = "calligraphy"

class AnimationStyle(Enum):
    """动画样式枚举"""
    NONE = "none"
    SIMPLE = "simple"
    SMOOTH = "smooth"
    ELASTIC = "elastic"
    BOUNCE = "bounce"

@dataclass
class ColorScheme:
    """颜色方案数据类"""
    name: str = "default"
    background: str = "#f0f0f0"
    foreground: str = "#2c3e50"
    primary: str = "#3498db"
    secondary: str = "#e74c3c"
    accent: str = "#f39c12"
    board_light: str = "#f4d03f"
    board_dark: str = "#d4ac0d"
    text: str = "#2c3e50"
    text_secondary: str = "#7f8c8d"
    border: str = "#bdc3c7"
    hover: str = "#ecf0f1"
    selected: str = "#3498db"
    highlight: str = "#e8f6fd"
    success: str = "#27ae60"
    warning: str = "#f39c12"
    error: str = "#e74c3c"
    info: str = "#3498db"

@dataclass
class FontScheme:
    """字体方案数据类"""
    name: str = "default"
    family: str = "Microsoft YaHei"
    size: int = 12
    weight: int = 400  # 400=normal, 700=bold
    ui_font: QFont = field(init=False)
    title_font: QFont = field(init=False)
    button_font: QFont = field(init=False)
    code_font: QFont = field(init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        self.ui_font = QFont(self.family, self.size, self.weight)
        self.title_font = QFont(self.family, self.size + 2, 600)
        self.button_font = QFont(self.family, self.size, 500)
        self.code_font = QFont("Consolas", self.size, 400)

@dataclass
class BoardTheme:
    """棋盘主题数据类"""
    name: str = "wood"
    style: BoardStyle = BoardStyle.WOOD
    light_color: str = "#f4d03f"
    dark_color: str = "#d4ac0d"
    border_color: str = "#8b7355"
    coord_color: str = "#5d4037"
    grid_color: str = "#795548"
    background_texture: Optional[str] = None
    border_width: int = 2
    grid_width: int = 1
    corner_radius: int = 8
    shadow_enabled: bool = True
    gradient_enabled: bool = True

@dataclass
class PieceTheme:
    """棋子主题数据类"""
    name: str = "traditional"
    style: PieceStyle = PieceStyle.TRADITIONAL
    red_color: str = "#c0392b"
    black_color: str = "#2c3e50"
    text_color: str = "#ffffff"
    border_color: str = "#34495e"
    shadow_color: str = "#7f8c8d"
    font_family: str = "华文楷体"
    font_size: int = 16
    font_weight: int = 700
    border_width: int = 2
    shadow_enabled: bool = True
    glow_enabled: bool = False
    texture_enabled: bool = False

@dataclass
class AnimationSettings:
    """动画设置数据类"""
    style: AnimationStyle = AnimationStyle.SMOOTH
    duration: int = 300  # 毫秒
    easing: str = "OutCubic"
    move_animation: bool = True
    capture_animation: bool = True
    selection_animation: bool = True
    hover_animation: bool = True
    fade_in_duration: int = 200
    fade_out_duration: int = 150
    bounce_amplitude: float = 0.1
    elastic_amplitude: float = 0.8

@dataclass
class Theme:
    """完整主题数据类"""
    name: str = "default"
    type: ThemeType = ThemeType.DEFAULT
    display_name: str = "默认主题"
    description: str = "默认的明亮主题"
    colors: ColorScheme = field(default_factory=ColorScheme)
    fonts: FontScheme = field(default_factory=FontScheme)
    board: BoardTheme = field(default_factory=BoardTheme)
    pieces: PieceTheme = field(default_factory=PieceTheme)
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    window_opacity: float = 1.0
    blur_enabled: bool = False
    effects_enabled: bool = True
    version: str = "1.0.0"
    author: str = "系统"
    created_at: float = field(default_factory=lambda: __import__('time').time())

class ColorUtils:
    """颜色工具类"""
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """十六进制颜色转RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """RGB颜色转十六进制"""
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def lighten_color(hex_color: str, factor: float) -> str:
        """变亮颜色"""
        r, g, b = ColorUtils.hex_to_rgb(hex_color)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return ColorUtils.rgb_to_hex(r, g, b)
    
    @staticmethod
    def darken_color(hex_color: str, factor: float) -> str:
        """变暗颜色"""
        r, g, b = ColorUtils.hex_to_rgb(hex_color)
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return ColorUtils.rgb_to_hex(r, g, b)
    
    @staticmethod
    def blend_colors(color1: str, color2: str, ratio: float) -> str:
        """混合两种颜色"""
        r1, g1, b1 = ColorUtils.hex_to_rgb(color1)
        r2, g2, b2 = ColorUtils.hex_to_rgb(color2)
        
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        
        return ColorUtils.rgb_to_hex(r, g, b)
    
    @staticmethod
    def get_contrast_color(hex_color: str) -> str:
        """获取对比色（黑或白）"""
        r, g, b = ColorUtils.hex_to_rgb(hex_color)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return "#000000" if luminance > 0.5 else "#ffffff"
    
    @staticmethod
    def generate_palette(base_color: str, count: int = 5) -> List[str]:
        """根据基础颜色生成调色板"""
        r, g, b = ColorUtils.hex_to_rgb(base_color)
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        
        palette = []
        for i in range(count):
            # 调整亮度
            new_l = max(0.1, min(0.9, l + (i - count//2) * 0.15))
            new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
            hex_color = ColorUtils.rgb_to_hex(
                int(new_r * 255), int(new_g * 255), int(new_b * 255)
            )
            palette.append(hex_color)
        
        return palette

class ThemePresets:
    """预设主题"""
    
    @staticmethod
    def get_default_theme() -> Theme:
        """获取默认主题"""
        colors = ColorScheme(
            name="default",
            background="#f8f9fa",
            foreground="#212529",
            primary="#007bff",
            secondary="#6c757d",
            accent="#28a745",
            board_light="#f4d03f",
            board_dark="#d4ac0d",
            text="#212529",
            text_secondary="#6c757d",
            border="#dee2e6",
            hover="#e9ecef",
            selected="#007bff",
            highlight="#cce5ff"
        )
        
        fonts = FontScheme(
            name="default",
            family="Microsoft YaHei",
            size=12,
            weight=400
        )
        
        board = BoardTheme(
            name="wood",
            style=BoardStyle.WOOD,
            light_color="#f4d03f",
            dark_color="#d4ac0d",
            border_color="#8b7355",
            shadow_enabled=True,
            gradient_enabled=True
        )
        
        pieces = PieceTheme(
            name="traditional",
            style=PieceStyle.TRADITIONAL,
            red_color="#c0392b",
            black_color="#2c3e50",
            font_family="华文楷体",
            shadow_enabled=True
        )
        
        return Theme(
            name="default",
            type=ThemeType.DEFAULT,
            display_name="默认主题",
            description="清爽明亮的默认主题",
            colors=colors,
            fonts=fonts,
            board=board,
            pieces=pieces
        )
    
    @staticmethod
    def get_dark_theme() -> Theme:
        """获取暗色主题"""
        colors = ColorScheme(
            name="dark",
            background="#2c3e50",
            foreground="#ecf0f1",
            primary="#3498db",
            secondary="#95a5a6",
            accent="#e74c3c",
            board_light="#34495e",
            board_dark="#2c3e50",
            text="#ecf0f1",
            text_secondary="#bdc3c7",
            border="#34495e",
            hover="#34495e",
            selected="#3498db",
            highlight="#2980b9"
        )
        
        fonts = FontScheme(
            name="dark",
            family="Microsoft YaHei",
            size=12,
            weight=400
        )
        
        board = BoardTheme(
            name="dark_wood",
            style=BoardStyle.WOOD,
            light_color="#34495e",
            dark_color="#2c3e50",
            border_color="#1a252f",
            shadow_enabled=True,
            gradient_enabled=True
        )
        
        pieces = PieceTheme(
            name="dark_traditional",
            style=PieceStyle.TRADITIONAL,
            red_color="#e74c3c",
            black_color="#ecf0f1",
            font_family="华文楷体",
            shadow_enabled=True,
            glow_enabled=True
        )
        
        return Theme(
            name="dark",
            type=ThemeType.DARK,
            display_name="暗色主题",
            description="护眼的暗色主题",
            colors=colors,
            fonts=fonts,
            board=board,
            pieces=pieces
        )
    
    @staticmethod
    def get_chess_theme() -> Theme:
        """获取象棋风格主题"""
        colors = ColorScheme(
            name="chess",
            background="#faf3e0",
            foreground="#3d2914",
            primary="#8b4513",
            secondary="#cd853f",
            accent="#ff6347",
            board_light="#deb887",
            board_dark="#d2b48c",
            text="#3d2914",
            text_secondary="#8b7355",
            border="#8b4513",
            hover="#f5deb3",
            selected="#ff6347",
            highlight="#ffe4b5"
        )
        
        fonts = FontScheme(
            name="chess",
            family="华文楷体",
            size=13,
            weight=500
        )
        
        board = BoardTheme(
            name="chess_traditional",
            style=BoardStyle.CLASSIC,
            light_color="#deb887",
            dark_color="#d2b48c",
            border_color="#8b4513",
            coord_color="#3d2914",
            shadow_enabled=True,
            gradient_enabled=True,
            corner_radius=4
        )
        
        pieces = PieceTheme(
            name="chess_calligraphy",
            style=PieceStyle.CALLIGRAPHY,
            red_color="#8b0000",
            black_color="#2f4f4f",
            font_family="华文楷体",
            font_size=18,
            font_weight=700,
            shadow_enabled=True,
            glow_enabled=False
        )
        
        return Theme(
            name="chess",
            type=ThemeType.CHESS,
            display_name="象棋风格",
            description="传统的中国象棋风格主题",
            colors=colors,
            fonts=fonts,
            board=board,
            pieces=pieces
        )

class StyleSheetGenerator:
    """样式表生成器"""
    
    @staticmethod
    def generate_main_stylesheet(theme: Theme) -> str:
        """生成主界面样式表"""
        colors = theme.colors
        fonts = theme.fonts
        
        stylesheet = f"""
        QMainWindow {{
            background-color: {colors.background};
            color: {colors.text};
            font-family: "{fonts.family}";
            font-size: {fonts.size}px;
        }}
        
        QWidget {{
            background-color: {colors.background};
            color: {colors.text};
            font-family: "{fonts.family}";
        }}
        
        QMenuBar {{
            background-color: {colors.background};
            color: {colors.text};
            border-bottom: 1px solid {colors.border};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 12px;
            margin: 2px 0px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {colors.hover};
            border-radius: 4px;
        }}
        
        QMenu {{
            background-color: {colors.background};
            border: 1px solid {colors.border};
            border-radius: 6px;
            padding: 4px;
        }}
        
        QMenu::item {{
            background-color: transparent;
            padding: 8px 16px;
            border-radius: 4px;
            margin: 1px;
        }}
        
        QMenu::item:selected {{
            background-color: {colors.hover};
        }}
        
        QToolBar {{
            background-color: {colors.background};
            border: none;
            spacing: 3px;
        }}
        
        QPushButton {{
            background-color: {colors.primary};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
        }}
        
        QPushButton:hover {{
            background-color: {ColorUtils.darken_color(colors.primary, 0.1)};
        }}
        
        QPushButton:pressed {{
            background-color: {ColorUtils.darken_color(colors.primary, 0.2)};
        }}
        
        QPushButton:disabled {{
            background-color: {colors.border};
            color: {colors.text_secondary};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {colors.border};
            border-radius: 6px;
        }}
        
        QTabBar::tab {{
            background-color: {colors.hover};
            color: {colors.text};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors.background};
            border-bottom: 2px solid {colors.primary};
        }}
        
        QGroupBox {{
            font-weight: 600;
            border: 2px solid {colors.border};
            border-radius: 6px;
            margin: 8px 0px;
            padding-top: 12px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
            background-color: {colors.background};
        }}
        
        QLineEdit {{
            background-color: {colors.background};
            border: 2px solid {colors.border};
            border-radius: 6px;
            padding: 6px 12px;
        }}
        
        QLineEdit:focus {{
            border-color: {colors.primary};
        }}
        
        QComboBox {{
            background-color: {colors.background};
            border: 2px solid {colors.border};
            border-radius: 6px;
            padding: 6px 12px;
            min-width: 100px;
        }}
        
        QComboBox:focus {{
            border-color: {colors.primary};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {colors.text};
            margin-right: 5px;
        }}
        
        QCheckBox {{
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 2px solid {colors.border};
            background-color: {colors.background};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {colors.primary};
            border-color: {colors.primary};
        }}
        
        QSlider::groove:horizontal {{
            background-color: {colors.border};
            height: 6px;
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background-color: {colors.primary};
            width: 18px;
            height: 18px;
            border-radius: 9px;
            margin-top: -6px;
            margin-bottom: -6px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background-color: {ColorUtils.darken_color(colors.primary, 0.1)};
        }}
        
        QStatusBar {{
            background-color: {colors.background};
            border-top: 1px solid {colors.border};
        }}
        
        QScrollArea {{
            border: none;
            background-color: {colors.background};
        }}
        
        QScrollBar:vertical {{
            background-color: {colors.hover};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors.border};
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {ColorUtils.darken_color(colors.border, 0.1)};
        }}
        """
        
        return stylesheet
    
    @staticmethod
    def generate_board_stylesheet(theme: Theme) -> str:
        """生成棋盘样式表"""
        board = theme.board
        colors = theme.colors
        
        stylesheet = f"""
        .board-widget {{
            background-color: {board.light_color};
            border: {board.border_width}px solid {board.border_color};
            border-radius: {board.corner_radius}px;
        }}
        
        .board-coordinate {{
            color: {board.coord_color};
            font-weight: 600;
            font-size: 11px;
        }}
        
        .board-grid {{
            color: {board.grid_color};
            stroke-width: {board.grid_width};
        }}
        
        .board-highlight {{
            background-color: {colors.highlight};
            border: 2px solid {colors.selected};
            border-radius: 4px;
        }}
        
        .board-suggestion {{
            border: 2px dashed {colors.accent};
            border-radius: 50%;
            background-color: transparent;
        }}
        """
        
        return stylesheet

class ThemeManager(QObject):
    """主题管理器"""
    
    # 信号定义
    theme_changed = pyqtSignal(Theme)
    theme_preview = pyqtSignal(Theme)
    theme_applied = pyqtSignal(Theme)
    
    def __init__(self):
        super().__init__()
        self._current_theme: Optional[Theme] = None
        self._themes: Dict[str, Theme] = {}
        self._custom_themes: Dict[str, Theme] = {}
        self._theme_file = Path("config/themes.json")
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._update_animation)
        self._setup_default_themes()
        self._load_custom_themes()
    
    def _setup_default_themes(self):
        """设置默认主题"""
        self._themes.update({
            "default": ThemePresets.get_default_theme(),
            "dark": ThemePresets.get_dark_theme(),
            "chess": ThemePresets.get_chess_theme()
        })
        
        # 设置默认主题
        self._current_theme = self._themes["default"]
    
    def _load_custom_themes(self):
        """加载自定义主题"""
        try:
            if self._theme_file.exists():
                with open(self._theme_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # TODO: 实现自定义主题反序列化
                    pass
        except Exception as e:
            print(f"加载自定义主题失败: {e}")
    
    def _save_custom_themes(self):
        """保存自定义主题"""
        try:
            self._theme_file.parent.mkdir(parents=True, exist_ok=True)
            # TODO: 实现自定义主题序列化
            pass
        except Exception as e:
            print(f"保存自定义主题失败: {e}")
    
    def get_available_themes(self) -> List[str]:
        """获取可用主题列表"""
        return list(self._themes.keys()) + list(self._custom_themes.keys())
    
    def get_theme(self, name: str) -> Optional[Theme]:
        """获取指定主题"""
        if name in self._themes:
            return self._themes[name]
        elif name in self._custom_themes:
            return self._custom_themes[name]
        return None
    
    def get_current_theme(self) -> Theme:
        """获取当前主题"""
        return self._current_theme
    
    def set_theme(self, name: str, apply: bool = True):
        """设置主题"""
        theme = self.get_theme(name)
        if not theme:
            return False
        
        self._current_theme = theme
        
        if apply:
            self._apply_theme(theme)
            self.theme_applied.emit(theme)
        else:
            self.theme_changed.emit(theme)
        
        return True
    
    def preview_theme(self, name: str):
        """预览主题"""
        theme = self.get_theme(name)
        if theme:
            self.theme_preview.emit(theme)
    
    def _apply_theme(self, theme: Theme):
        """应用主题"""
        app = QApplication.instance()
        if not app:
            return
        
        # 生成样式表
        main_stylesheet = StyleSheetGenerator.generate_main_stylesheet(theme)
        
        # 应用到应用程序
        app.setStyleSheet(main_stylesheet)
        
        # 设置窗口透明度
        for widget in app.allWidgets():
            if isinstance(widget, QWidget) and widget.isWindow():
                widget.setWindowOpacity(theme.window_opacity)
    
    def create_custom_theme(self, name: str, base_theme: str = "default") -> Theme:
        """创建自定义主题"""
        base = self.get_theme(base_theme)
        if not base:
            base = self._themes["default"]
        
        # 深拷贝基础主题
        import copy
        custom_theme = copy.deepcopy(base)
        custom_theme.name = name
        custom_theme.type = ThemeType.CUSTOM
        custom_theme.display_name = f"自定义 - {name}"
        custom_theme.author = "用户"
        custom_theme.created_at = __import__('time').time()
        
        self._custom_themes[name] = custom_theme
        self._save_custom_themes()
        
        return custom_theme
    
    def modify_theme_colors(self, theme_name: str, colors: Dict[str, str]):
        """修改主题颜色"""
        theme = self.get_theme(theme_name)
        if not theme or theme.type != ThemeType.CUSTOM:
            return False
        
        for key, value in colors.items():
            if hasattr(theme.colors, key):
                setattr(theme.colors, key, value)
        
        self._save_custom_themes()
        return True
    
    def modify_theme_fonts(self, theme_name: str, fonts: Dict[str, Any]):
        """修改主题字体"""
        theme = self.get_theme(theme_name)
        if not theme or theme.type != ThemeType.CUSTOM:
            return False
        
        for key, value in fonts.items():
            if hasattr(theme.fonts, key):
                setattr(theme.fonts, key, value)
        
        # 重新初始化字体
        theme.fonts.__post_init__()
        
        self._save_custom_themes()
        return True
    
    def delete_custom_theme(self, name: str) -> bool:
        """删除自定义主题"""
        if name in self._custom_themes:
            del self._custom_themes[name]
            self._save_custom_themes()
            return True
        return False
    
    def export_theme(self, name: str, file_path: str) -> bool:
        """导出主题"""
        theme = self.get_theme(name)
        if not theme:
            return False
        
        try:
            # TODO: 实现主题导出
            return True
        except Exception as e:
            print(f"导出主题失败: {e}")
            return False
    
    def import_theme(self, file_path: str) -> Optional[str]:
        """导入主题"""
        try:
            # TODO: 实现主题导入
            return "imported_theme"
        except Exception as e:
            print(f"导入主题失败: {e}")
            return None
    
    def generate_color_palette(self, base_color: str) -> List[str]:
        """生成调色板"""
        return ColorUtils.generate_palette(base_color)
    
    def get_main_stylesheet(self) -> str:
        """获取主样式表"""
        return StyleSheetGenerator.generate_main_stylesheet(self._current_theme)
    
    def get_board_stylesheet(self) -> str:
        """获取棋盘样式表"""
        return StyleSheetGenerator.generate_board_stylesheet(self._current_theme)
    
    def _update_animation(self):
        """更新动画"""
        # TODO: 实现动画更新逻辑
        pass
    
    def start_animations(self):
        """开始动画"""
        if self._current_theme.animation.style != AnimationStyle.NONE:
            self._animation_timer.start(16)  # 60 FPS
    
    def stop_animations(self):
        """停止动画"""
        self._animation_timer.stop()
    
    def get_theme_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取主题信息"""
        theme = self.get_theme(name)
        if not theme:
            return None
        
        return {
            "name": theme.name,
            "display_name": theme.display_name,
            "description": theme.description,
            "type": theme.type.value,
            "author": theme.author,
            "version": theme.version,
            "created_at": theme.created_at,
            "colors_count": len(theme.colors.__dict__),
            "has_animations": theme.animation.style != AnimationStyle.NONE,
            "window_opacity": theme.window_opacity,
            "effects_enabled": theme.effects_enabled
        }

# 单例模式的主题管理器实例
_theme_manager_instance: Optional[ThemeManager] = None

def get_theme_manager() -> ThemeManager:
    """获取主题管理器单例"""
    global _theme_manager_instance
    if _theme_manager_instance is None:
        _theme_manager_instance = ThemeManager()
    return _theme_manager_instance
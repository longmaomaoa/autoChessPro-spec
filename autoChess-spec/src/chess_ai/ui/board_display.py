#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋棋盘显示组件

提供完整的象棋棋盘可视化、用户交互和AI建议显示功能
"""

import math
import time
from typing import Optional, List, Dict, Tuple, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSplitter, QGroupBox, QCheckBox,
    QSlider, QSpinBox, QComboBox, QTextEdit, QProgressBar
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QTimer, QPoint, QRect, QSize, QRectF,
    pyqtSlot, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QPainterPath,
    QLinearGradient, QRadialGradient, QPolygonF, QTransform, 
    QMouseEvent, QPaintEvent, QResizeEvent
)

from chess_ai.core.board_state import BoardState, PieceType, PieceColor, Position
from chess_ai.ai_engine.ai_engine_interface import MoveSuggestion, MoveType


class ChessPieceRenderer:
    """象棋棋子渲染器"""
    
    def __init__(self):
        # 棋子中文字符映射
        self.piece_chars = {
            PieceType.SHUAI: "帥", PieceType.JIANG: "將",
            PieceType.SHI: "仕", PieceType.SHI_BLACK: "士", 
            PieceType.XIANG: "相", PieceType.XIANG_BLACK: "象",
            PieceType.MA: "馬", PieceType.MA_BLACK: "马",
            PieceType.CHE: "車", PieceType.CHE_BLACK: "车",
            PieceType.PAO: "炮", PieceType.PAO_BLACK: "砲",
            PieceType.BING: "兵", PieceType.ZU: "卒"
        }
        
        # 棋子颜色
        self.piece_colors = {
            PieceColor.RED: QColor(220, 20, 20),     # 红色
            PieceColor.BLACK: QColor(20, 20, 20)     # 黑色
        }
        
        # 棋子背景色
        self.piece_backgrounds = {
            PieceColor.RED: QColor(255, 245, 200),   # 淡黄色
            PieceColor.BLACK: QColor(255, 245, 200)  # 淡黄色
        }
    
    def draw_piece(self, painter: QPainter, piece_type: PieceType, 
                   piece_color: PieceColor, center: QPoint, radius: float):
        """绘制棋子"""
        # 获取棋子字符
        char = self.piece_chars.get(piece_type, "？")
        
        # 绘制棋子背景圆
        painter.setBrush(QBrush(self.piece_backgrounds[piece_color]))
        painter.setPen(QPen(QColor(139, 69, 19), 2))  # 棕色边框
        painter.drawEllipse(center, radius, radius)
        
        # 绘制棋子文字
        painter.setPen(QPen(self.piece_colors[piece_color]))
        font = QFont("Microsoft YaHei", int(radius * 0.6), QFont.Weight.Bold)
        painter.setFont(font)
        
        # 计算文字位置
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(char)
        text_pos = QPoint(
            center.x() - text_rect.width() // 2,
            center.y() + text_rect.height() // 4
        )
        
        painter.drawText(text_pos, char)


class BoardCoordinates:
    """棋盘坐标系统"""
    
    def __init__(self, board_rect: QRect):
        self.board_rect = board_rect
        self.margin = 40
        
        # 计算网格大小
        self.grid_width = (board_rect.width() - 2 * self.margin) / 8
        self.grid_height = (board_rect.height() - 2 * self.margin) / 9
        
        # 起始坐标
        self.start_x = board_rect.left() + self.margin
        self.start_y = board_rect.top() + self.margin
    
    def position_to_point(self, pos: Position) -> QPoint:
        """象棋位置转换为屏幕坐标"""
        x = self.start_x + pos.file * self.grid_width
        y = self.start_y + pos.rank * self.grid_height
        return QPoint(int(x), int(y))
    
    def point_to_position(self, point: QPoint) -> Optional[Position]:
        """屏幕坐标转换为象棋位置"""
        if not self.board_rect.contains(point):
            return None
        
        # 计算相对位置
        rel_x = point.x() - self.start_x
        rel_y = point.y() - self.start_y
        
        # 计算网格位置
        file = round(rel_x / self.grid_width)
        rank = round(rel_y / self.grid_height)
        
        # 边界检查
        if 0 <= file <= 8 and 0 <= rank <= 9:
            return Position(file, rank)
        
        return None
    
    def get_piece_radius(self) -> float:
        """获取棋子半径"""
        return min(self.grid_width, self.grid_height) * 0.35


class MoveAnimation:
    """走法动画"""
    
    def __init__(self, from_pos: Position, to_pos: Position, 
                 piece_type: PieceType, piece_color: PieceColor):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_type = piece_type
        self.piece_color = piece_color
        self.progress = 0.0
        self.duration = 500  # 毫秒


class BoardDisplayWidget(QWidget):
    """象棋棋盘显示组件
    
    提供完整的棋盘可视化、用户交互和走法建议显示
    """
    
    # 信号定义
    piece_clicked = pyqtSignal(Position, PieceType, PieceColor)
    position_clicked = pyqtSignal(Position)
    move_requested = pyqtSignal(Position, Position)  # 从位置, 到位置
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 棋盘状态
        self.board_state: Optional[BoardState] = None
        self.coordinates: Optional[BoardCoordinates] = None
        self.piece_renderer = ChessPieceRenderer()
        
        # 交互状态
        self.selected_position: Optional[Position] = None
        self.highlighted_positions: List[Position] = []
        self.suggested_moves: List[MoveSuggestion] = []
        
        # 显示设置
        self.show_coordinates = True
        self.show_suggestions = True
        self.show_last_move = True
        self.animation_enabled = True
        
        # 动画系统
        self.current_animation: Optional[MoveAnimation] = None
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        
        # 颜色主题
        self.colors = {
            'board_bg': QColor(139, 69, 19),          # 棋盘背景
            'line': QColor(0, 0, 0),                  # 网格线
            'coordinate': QColor(50, 50, 50),         # 坐标文字
            'selected': QColor(255, 215, 0, 100),     # 选中高亮
            'suggestion': QColor(0, 150, 255, 80),    # 建议高亮
            'last_move': QColor(255, 100, 100, 60),   # 上步走法
            'river': QColor(100, 149, 237, 30)        # 楚河汉界
        }
        
        # 界面设置
        self.setMinimumSize(400, 450)
        self.setMouseTracking(True)
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 创建控制面板
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # 棋盘区域占据剩余空间
        layout.addStretch(1)
        
        # 创建信息面板
        info_panel = self._create_info_panel()
        layout.addWidget(info_panel)
    
    def _create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumHeight(80)
        
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 显示选项
        options_group = QGroupBox("显示选项")
        options_layout = QHBoxLayout(options_group)
        
        self.coord_checkbox = QCheckBox("坐标")
        self.coord_checkbox.setChecked(self.show_coordinates)
        self.coord_checkbox.toggled.connect(self._toggle_coordinates)
        options_layout.addWidget(self.coord_checkbox)
        
        self.suggestion_checkbox = QCheckBox("AI建议")
        self.suggestion_checkbox.setChecked(self.show_suggestions)
        self.suggestion_checkbox.toggled.connect(self._toggle_suggestions)
        options_layout.addWidget(self.suggestion_checkbox)
        
        self.animation_checkbox = QCheckBox("动画")
        self.animation_checkbox.setChecked(self.animation_enabled)
        self.animation_checkbox.toggled.connect(self._toggle_animation)
        options_layout.addWidget(self.animation_checkbox)
        
        layout.addWidget(options_group)
        
        # 控制按钮
        buttons_group = QGroupBox("操作")
        buttons_layout = QHBoxLayout(buttons_group)
        
        self.reset_button = QPushButton("重置选择")
        self.reset_button.clicked.connect(self._reset_selection)
        buttons_layout.addWidget(self.reset_button)
        
        self.flip_button = QPushButton("翻转棋盘")
        self.flip_button.clicked.connect(self._flip_board)
        buttons_layout.addWidget(self.flip_button)
        
        layout.addWidget(buttons_group)
        
        layout.addStretch(1)
        
        return panel
    
    def _create_info_panel(self) -> QWidget:
        """创建信息面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumHeight(60)
        
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 当前选中信息
        self.selection_label = QLabel("未选中棋子")
        layout.addWidget(self.selection_label)
        
        layout.addStretch(1)
        
        # 建议数量
        self.suggestion_label = QLabel("建议: 0")
        layout.addWidget(self.suggestion_label)
        
        return panel
    
    @pyqtSlot(BoardState)
    def update_board_state(self, board_state: BoardState):
        """更新棋盘状态"""
        old_state = self.board_state
        self.board_state = board_state
        
        # 如果有动画且状态发生变化，启动动画
        if (self.animation_enabled and old_state and 
            self._has_state_changed(old_state, board_state)):
            self._start_move_animation(old_state, board_state)
        
        self.update()
    
    @pyqtSlot(list)
    def update_suggestions(self, suggestions: List[MoveSuggestion]):
        """更新AI建议"""
        self.suggested_moves = suggestions
        self.suggestion_label.setText(f"建议: {len(suggestions)}")
        self.update()
    
    def paintEvent(self, event: QPaintEvent):
        """绘制棋盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 计算棋盘区域
        board_rect = self._calculate_board_rect()
        self.coordinates = BoardCoordinates(board_rect)
        
        # 绘制棋盘背景
        self._draw_board_background(painter, board_rect)
        
        # 绘制网格线
        self._draw_grid_lines(painter, board_rect)
        
        # 绘制楚河汉界
        self._draw_river(painter, board_rect)
        
        # 绘制特殊标记
        self._draw_special_marks(painter, board_rect)
        
        # 绘制坐标
        if self.show_coordinates:
            self._draw_coordinates(painter, board_rect)
        
        # 绘制高亮区域
        self._draw_highlights(painter)
        
        # 绘制AI建议
        if self.show_suggestions:
            self._draw_suggestions(painter)
        
        # 绘制棋子
        self._draw_pieces(painter)
        
        # 绘制动画
        if self.current_animation:
            self._draw_animation(painter)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击事件"""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if not self.coordinates:
            return
        
        # 转换坐标
        click_pos = self.coordinates.point_to_position(event.pos())
        if not click_pos:
            return
        
        self.position_clicked.emit(click_pos)
        
        # 处理棋子选择
        if self.board_state:
            piece = self.board_state.get_piece_at(click_pos)
            
            if piece:
                # 点击了棋子
                self.piece_clicked.emit(click_pos, piece.piece_type, piece.color)
                
                if self.selected_position == click_pos:
                    # 取消选择
                    self.selected_position = None
                    self.highlighted_positions = []
                else:
                    # 选择棋子
                    self.selected_position = click_pos
                    self.highlighted_positions = self._get_possible_moves(click_pos, piece)
                    
                    # 更新选择信息
                    piece_name = self._get_piece_name(piece.piece_type, piece.color)
                    self.selection_label.setText(f"已选中: {piece_name} ({click_pos})")
            
            elif self.selected_position:
                # 点击了空位置，尝试移动
                self.move_requested.emit(self.selected_position, click_pos)
                
                # 重置选择
                self.selected_position = None
                self.highlighted_positions = []
                self.selection_label.setText("未选中棋子")
        
        self.update()
    
    def _calculate_board_rect(self) -> QRect:
        """计算棋盘绘制区域"""
        widget_rect = self.rect()
        
        # 为控制面板和信息面板留出空间
        available_height = widget_rect.height() - 140  # 减去面板高度
        available_width = widget_rect.width() - 20     # 减去边距
        
        # 保持棋盘比例 (约为 8:9)
        ideal_width = available_height * 8 / 9
        ideal_height = available_width * 9 / 8
        
        if ideal_width <= available_width:
            # 高度受限
            board_width = int(ideal_width)
            board_height = available_height
        else:
            # 宽度受限
            board_width = available_width
            board_height = int(ideal_height)
        
        # 居中显示
        x = (widget_rect.width() - board_width) // 2
        y = 80 + (available_height - board_height) // 2  # 80是控制面板高度
        
        return QRect(x, y, board_width, board_height)
    
    def _draw_board_background(self, painter: QPainter, rect: QRect):
        """绘制棋盘背景"""
        painter.setBrush(QBrush(self.colors['board_bg']))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(rect)
    
    def _draw_grid_lines(self, painter: QPainter, rect: QRect):
        """绘制网格线"""
        if not self.coordinates:
            return
        
        painter.setPen(QPen(self.colors['line'], 1))
        
        # 绘制垂直线
        for i in range(9):
            x = self.coordinates.start_x + i * self.coordinates.grid_width
            y_start = self.coordinates.start_y
            y_end = self.coordinates.start_y + 9 * self.coordinates.grid_height
            painter.drawLine(int(x), int(y_start), int(x), int(y_end))
        
        # 绘制水平线
        for i in range(10):
            y = self.coordinates.start_y + i * self.coordinates.grid_height
            x_start = self.coordinates.start_x
            x_end = self.coordinates.start_x + 8 * self.coordinates.grid_width
            
            # 中间的楚河汉界线段分开绘制
            if i == 4 or i == 5:
                # 左侧
                painter.drawLine(int(x_start), int(y), 
                               int(x_start + 3 * self.coordinates.grid_width), int(y))
                # 右侧
                painter.drawLine(int(x_start + 5 * self.coordinates.grid_width), int(y),
                               int(x_end), int(y))
            else:
                painter.drawLine(int(x_start), int(y), int(x_end), int(y))
        
        # 绘制九宫格对角线
        self._draw_palace_lines(painter)
    
    def _draw_palace_lines(self, painter: QPainter):
        """绘制九宫格对角线"""
        if not self.coordinates:
            return
        
        painter.setPen(QPen(self.colors['line'], 1))
        
        # 上方九宫格
        top_left = QPoint(int(self.coordinates.start_x + 3 * self.coordinates.grid_width),
                         int(self.coordinates.start_y))
        top_right = QPoint(int(self.coordinates.start_x + 5 * self.coordinates.grid_width),
                          int(self.coordinates.start_y))
        bottom_left = QPoint(int(self.coordinates.start_x + 3 * self.coordinates.grid_width),
                            int(self.coordinates.start_y + 2 * self.coordinates.grid_height))
        bottom_right = QPoint(int(self.coordinates.start_x + 5 * self.coordinates.grid_width),
                             int(self.coordinates.start_y + 2 * self.coordinates.grid_height))
        
        painter.drawLine(top_left, bottom_right)
        painter.drawLine(top_right, bottom_left)
        
        # 下方九宫格
        top_left_b = QPoint(int(self.coordinates.start_x + 3 * self.coordinates.grid_width),
                           int(self.coordinates.start_y + 7 * self.coordinates.grid_height))
        top_right_b = QPoint(int(self.coordinates.start_x + 5 * self.coordinates.grid_width),
                            int(self.coordinates.start_y + 7 * self.coordinates.grid_height))
        bottom_left_b = QPoint(int(self.coordinates.start_x + 3 * self.coordinates.grid_width),
                              int(self.coordinates.start_y + 9 * self.coordinates.grid_height))
        bottom_right_b = QPoint(int(self.coordinates.start_x + 5 * self.coordinates.grid_width),
                               int(self.coordinates.start_y + 9 * self.coordinates.grid_height))
        
        painter.drawLine(top_left_b, bottom_right_b)
        painter.drawLine(top_right_b, bottom_left_b)
    
    def _draw_river(self, painter: QPainter, rect: QRect):
        """绘制楚河汉界"""
        if not self.coordinates:
            return
        
        # 绘制河界背景色
        river_y = self.coordinates.start_y + 4 * self.coordinates.grid_height
        river_height = self.coordinates.grid_height
        river_rect = QRect(int(self.coordinates.start_x), int(river_y),
                          int(8 * self.coordinates.grid_width), int(river_height))
        
        painter.setBrush(QBrush(self.colors['river']))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(river_rect)
        
        # 绘制"楚河汉界"文字
        painter.setPen(QPen(QColor(80, 80, 80)))
        font = QFont("Microsoft YaHei", 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        chu_text = "楚河"
        han_text = "汉界"
        
        # 楚河位置（左侧）
        chu_x = int(self.coordinates.start_x + self.coordinates.grid_width)
        chu_y = int(river_y + river_height // 2 + 6)
        painter.drawText(QPoint(chu_x, chu_y), chu_text)
        
        # 汉界位置（右侧）
        han_x = int(self.coordinates.start_x + 5 * self.coordinates.grid_width)
        han_y = int(river_y + river_height // 2 + 6)
        painter.drawText(QPoint(han_x, han_y), han_text)
    
    def _draw_special_marks(self, painter: QPainter, rect: QRect):
        """绘制特殊标记点"""
        if not self.coordinates:
            return
        
        painter.setPen(QPen(self.colors['line'], 2))
        painter.setBrush(QBrush(self.colors['line']))
        
        # 标记点位置（兵和炮的位置）
        mark_positions = [
            (1, 2), (7, 2),  # 上方炮位
            (0, 3), (2, 3), (4, 3), (6, 3), (8, 3),  # 上方兵位
            (0, 6), (2, 6), (4, 6), (6, 6), (8, 6),  # 下方兵位
            (1, 7), (7, 7)   # 下方炮位
        ]
        
        for file, rank in mark_positions:
            center = self.coordinates.position_to_point(Position(file, rank))
            self._draw_position_mark(painter, center)
    
    def _draw_position_mark(self, painter: QPainter, center: QPoint):
        """绘制位置标记"""
        size = 3
        painter.drawEllipse(center.x() - size, center.y() - size, size * 2, size * 2)
    
    def _draw_coordinates(self, painter: QPainter, rect: QRect):
        """绘制坐标"""
        if not self.coordinates:
            return
        
        painter.setPen(QPen(self.colors['coordinate']))
        font = QFont("Arial", 10)
        painter.setFont(font)
        
        # 绘制文件坐标 (a-i)
        for i in range(9):
            x = self.coordinates.start_x + i * self.coordinates.grid_width
            y = self.coordinates.start_y - 10
            painter.drawText(QPoint(int(x - 5), int(y)), chr(ord('a') + i))
        
        # 绘制等级坐标 (0-9)
        for i in range(10):
            x = self.coordinates.start_x - 15
            y = self.coordinates.start_y + i * self.coordinates.grid_height
            painter.drawText(QPoint(int(x), int(y + 5)), str(9 - i))
    
    def _draw_highlights(self, painter: QPainter):
        """绘制高亮区域"""
        if not self.coordinates:
            return
        
        radius = self.coordinates.get_piece_radius() * 1.2
        
        # 绘制选中位置
        if self.selected_position:
            center = self.coordinates.position_to_point(self.selected_position)
            painter.setBrush(QBrush(self.colors['selected']))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, radius, radius)
        
        # 绘制可能的移动位置
        painter.setBrush(QBrush(self.colors['suggestion']))
        for pos in self.highlighted_positions:
            center = self.coordinates.position_to_point(pos)
            painter.drawEllipse(center, radius * 0.8, radius * 0.8)
    
    def _draw_suggestions(self, painter: QPainter):
        """绘制AI建议"""
        if not self.coordinates or not self.suggested_moves:
            return
        
        # 绘制建议走法的箭头
        for i, suggestion in enumerate(self.suggested_moves[:3]):  # 最多显示3个建议
            # 解析走法
            move_parts = suggestion.move.split('-') if '-' in suggestion.move else [suggestion.move[:2], suggestion.move[2:]]
            if len(move_parts) != 2:
                continue
            
            # 转换为Position对象 (简化实现)
            try:
                from_pos = Position(ord(move_parts[0][0]) - ord('a'), int(move_parts[0][1]))
                to_pos = Position(ord(move_parts[1][0]) - ord('a'), int(move_parts[1][1]))
            except:
                continue
            
            self._draw_move_arrow(painter, from_pos, to_pos, i)
    
    def _draw_move_arrow(self, painter: QPainter, from_pos: Position, 
                        to_pos: Position, priority: int):
        """绘制走法箭头"""
        if not self.coordinates:
            return
        
        start_point = self.coordinates.position_to_point(from_pos)
        end_point = self.coordinates.position_to_point(to_pos)
        
        # 根据优先级设置颜色
        colors = [QColor(255, 100, 100, 150), QColor(100, 255, 100, 150), QColor(100, 100, 255, 150)]
        color = colors[priority % len(colors)]
        
        painter.setPen(QPen(color, 3))
        painter.setBrush(QBrush(color))
        
        # 绘制箭头线
        painter.drawLine(start_point, end_point)
        
        # 绘制箭头头部
        self._draw_arrow_head(painter, start_point, end_point, color)
    
    def _draw_arrow_head(self, painter: QPainter, start: QPoint, end: QPoint, color: QColor):
        """绘制箭头头部"""
        # 计算箭头方向
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.sqrt(dx * dx + dy * dy)
        
        if length == 0:
            return
        
        # 单位向量
        ux = dx / length
        uy = dy / length
        
        # 箭头参数
        arrow_length = 15
        arrow_width = 8
        
        # 箭头顶点
        tip = end
        left = QPoint(int(end.x() - arrow_length * ux - arrow_width * uy),
                     int(end.y() - arrow_length * uy + arrow_width * ux))
        right = QPoint(int(end.x() - arrow_length * ux + arrow_width * uy),
                      int(end.y() - arrow_length * uy - arrow_width * ux))
        
        # 绘制箭头三角形
        triangle = QPolygonF([tip, left, right])
        painter.setPen(QPen(color))
        painter.setBrush(QBrush(color))
        painter.drawPolygon(triangle)
    
    def _draw_pieces(self, painter: QPainter):
        """绘制棋子"""
        if not self.board_state or not self.coordinates:
            return
        
        radius = self.coordinates.get_piece_radius()
        
        for rank in range(10):
            for file in range(9):
                pos = Position(file, rank)
                piece = self.board_state.get_piece_at(pos)
                
                if piece:
                    center = self.coordinates.position_to_point(pos)
                    self.piece_renderer.draw_piece(
                        painter, piece.piece_type, piece.color, center, radius
                    )
    
    def _draw_animation(self, painter: QPainter):
        """绘制移动动画"""
        if not self.current_animation or not self.coordinates:
            return
        
        # 计算动画位置
        from_point = self.coordinates.position_to_point(self.current_animation.from_pos)
        to_point = self.coordinates.position_to_point(self.current_animation.to_pos)
        
        # 插值计算当前位置
        progress = self.current_animation.progress
        current_x = from_point.x() + (to_point.x() - from_point.x()) * progress
        current_y = from_point.y() + (to_point.y() - from_point.y()) * progress
        current_point = QPoint(int(current_x), int(current_y))
        
        # 绘制动画棋子
        radius = self.coordinates.get_piece_radius()
        self.piece_renderer.draw_piece(
            painter, self.current_animation.piece_type,
            self.current_animation.piece_color, current_point, radius
        )
    
    def _has_state_changed(self, old_state: BoardState, new_state: BoardState) -> bool:
        """检查棋盘状态是否发生变化"""
        # 简化实现 - 比较FEN字符串
        return old_state.fen != new_state.fen
    
    def _start_move_animation(self, old_state: BoardState, new_state: BoardState):
        """启动走法动画"""
        # 简化实现 - 找到发生变化的位置
        # 实际需要更复杂的差异检测算法
        
        # 这里假设我们能够检测到移动
        # 在实际实现中，需要比较两个状态找出移动的棋子
        pass
    
    def _update_animation(self):
        """更新动画"""
        if not self.current_animation:
            return
        
        self.current_animation.progress += 0.05  # 每次增加5%
        
        if self.current_animation.progress >= 1.0:
            # 动画完成
            self.current_animation = None
            self.animation_timer.stop()
        
        self.update()
    
    def _get_possible_moves(self, pos: Position, piece) -> List[Position]:
        """获取可能的移动位置"""
        # 简化实现 - 返回一些示例位置
        # 实际需要根据象棋规则计算
        moves = []
        
        # 示例: 为选中的棋子周围添加几个可能位置
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_file = pos.file + dx
                new_rank = pos.rank + dy
                
                if 0 <= new_file <= 8 and 0 <= new_rank <= 9:
                    moves.append(Position(new_file, new_rank))
        
        return moves[:4]  # 最多返回4个位置
    
    def _get_piece_name(self, piece_type: PieceType, color: PieceColor) -> str:
        """获取棋子名称"""
        color_name = "红" if color == PieceColor.RED else "黑"
        piece_char = self.piece_renderer.piece_chars.get(piece_type, "未知")
        return f"{color_name}{piece_char}"
    
    # 控制面板回调
    @pyqtSlot(bool)
    def _toggle_coordinates(self, checked: bool):
        """切换坐标显示"""
        self.show_coordinates = checked
        self.update()
    
    @pyqtSlot(bool)
    def _toggle_suggestions(self, checked: bool):
        """切换建议显示"""
        self.show_suggestions = checked
        self.update()
    
    @pyqtSlot(bool)
    def _toggle_animation(self, checked: bool):
        """切换动画效果"""
        self.animation_enabled = checked
    
    @pyqtSlot()
    def _reset_selection(self):
        """重置选择"""
        self.selected_position = None
        self.highlighted_positions = []
        self.selection_label.setText("未选中棋子")
        self.update()
    
    @pyqtSlot()
    def _flip_board(self):
        """翻转棋盘"""
        # TODO: 实现棋盘翻转
        pass
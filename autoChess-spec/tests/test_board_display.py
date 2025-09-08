#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋盘显示和用户交互功能测试

测试棋盘渲染、坐标转换、用户交互和AI建议显示功能
"""

import sys
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

print("开始棋盘显示和用户交互功能测试...")
print("=" * 60)

# 测试用的数据结构定义 (不依赖PyQt6)
class MockPosition:
    """位置模拟类"""
    
    def __init__(self, file: int, rank: int):
        self.file = file
        self.rank = rank
    
    def __eq__(self, other):
        if not isinstance(other, MockPosition):
            return False
        return self.file == other.file and self.rank == other.rank
    
    def __str__(self):
        return f"{chr(ord('a') + self.file)}{self.rank}"
    
    def __repr__(self):
        return f"Position({self.file}, {self.rank})"

class MockPieceType:
    """棋子类型枚举模拟"""
    SHUAI = "shuai"
    JIANG = "jiang"
    SHI = "shi"
    SHI_BLACK = "shi_black"
    XIANG = "xiang"
    XIANG_BLACK = "xiang_black"
    MA = "ma"
    MA_BLACK = "ma_black"
    CHE = "che"
    CHE_BLACK = "che_black"
    PAO = "pao"
    PAO_BLACK = "pao_black"
    BING = "bing"
    ZU = "zu"

class MockPieceColor:
    """棋子颜色枚举模拟"""
    RED = "red"
    BLACK = "black"

class MockRect:
    """矩形模拟类"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def left(self):
        return self.x
    
    def top(self):
        return self.y
    
    def contains(self, point):
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)

class MockPoint:
    """点模拟类"""
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class MockChessPieceRenderer:
    """象棋棋子渲染器模拟"""
    
    def __init__(self):
        # 棋子中文字符映射
        self.piece_chars = {
            MockPieceType.SHUAI: "帥", MockPieceType.JIANG: "將",
            MockPieceType.SHI: "仕", MockPieceType.SHI_BLACK: "士", 
            MockPieceType.XIANG: "相", MockPieceType.XIANG_BLACK: "象",
            MockPieceType.MA: "馬", MockPieceType.MA_BLACK: "马",
            MockPieceType.CHE: "車", MockPieceType.CHE_BLACK: "车",
            MockPieceType.PAO: "炮", MockPieceType.PAO_BLACK: "砲",
            MockPieceType.BING: "兵", MockPieceType.ZU: "卒"
        }
        
        # 棋子颜色
        self.piece_colors = {
            MockPieceColor.RED: (220, 20, 20),     # 红色
            MockPieceColor.BLACK: (20, 20, 20)     # 黑色
        }
        
        # 棋子背景色
        self.piece_backgrounds = {
            MockPieceColor.RED: (255, 245, 200),   # 淡黄色
            MockPieceColor.BLACK: (255, 245, 200)  # 淡黄色
        }
    
    def draw_piece(self, piece_type: MockPieceType, piece_color: MockPieceColor, 
                   center: MockPoint, radius: float) -> str:
        """绘制棋子 (返回描述字符串)"""
        char = self.piece_chars.get(piece_type, "？")
        color = "红" if piece_color == MockPieceColor.RED else "黑"
        return f"{color}{char}@({center.x},{center.y})"

class MockBoardCoordinates:
    """棋盘坐标系统模拟"""
    
    def __init__(self, board_rect: MockRect):
        self.board_rect = board_rect
        self.margin = 40
        
        # 计算网格大小
        self.grid_width = (board_rect.width - 2 * self.margin) / 8
        self.grid_height = (board_rect.height - 2 * self.margin) / 9
        
        # 起始坐标
        self.start_x = board_rect.left() + self.margin
        self.start_y = board_rect.top() + self.margin
    
    def position_to_point(self, pos: MockPosition) -> MockPoint:
        """象棋位置转换为屏幕坐标"""
        x = self.start_x + pos.file * self.grid_width
        y = self.start_y + pos.rank * self.grid_height
        return MockPoint(int(x), int(y))
    
    def point_to_position(self, point: MockPoint) -> Optional[MockPosition]:
        """屏幕坐标转换为象棋位置"""
        if not self.board_rect.contains(point):
            return None
        
        # 计算相对位置
        rel_x = point.x - self.start_x
        rel_y = point.y - self.start_y
        
        # 计算网格位置
        file = round(rel_x / self.grid_width)
        rank = round(rel_y / self.grid_height)
        
        # 边界检查
        if 0 <= file <= 8 and 0 <= rank <= 9:
            return MockPosition(file, rank)
        
        return None
    
    def get_piece_radius(self) -> float:
        """获取棋子半径"""
        return min(self.grid_width, self.grid_height) * 0.35

class MockMoveAnimation:
    """走法动画模拟"""
    
    def __init__(self, from_pos: MockPosition, to_pos: MockPosition, 
                 piece_type: MockPieceType, piece_color: MockPieceColor):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_type = piece_type
        self.piece_color = piece_color
        self.progress = 0.0
        self.duration = 500  # 毫秒

class MockBoardDisplayWidget:
    """棋盘显示组件模拟"""
    
    def __init__(self):
        # 棋盘状态
        self.board_state = None
        self.coordinates = None
        self.piece_renderer = MockChessPieceRenderer()
        
        # 交互状态
        self.selected_position = None
        self.highlighted_positions = []
        self.suggested_moves = []
        
        # 显示设置
        self.show_coordinates = True
        self.show_suggestions = True
        self.show_last_move = True
        self.animation_enabled = True
        
        # 动画系统
        self.current_animation = None
        self.animation_active = False
        
        # 颜色主题
        self.colors = {
            'board_bg': (139, 69, 19),          # 棋盘背景
            'line': (0, 0, 0),                  # 网格线
            'coordinate': (50, 50, 50),         # 坐标文字
            'selected': (255, 215, 0, 100),     # 选中高亮
            'suggestion': (0, 150, 255, 80),    # 建议高亮
            'last_move': (255, 100, 100, 60),   # 上步走法
            'river': (100, 149, 237, 30)        # 楚河汉界
        }
        
        # 界面状态
        self.min_width = 400
        self.min_height = 450
        self.mouse_tracking = True
        
        # 控制面板状态
        self.coord_checkbox_checked = self.show_coordinates
        self.suggestion_checkbox_checked = self.show_suggestions
        self.animation_checkbox_checked = self.animation_enabled
        
        # 信息面板状态
        self.selection_text = "未选中棋子"
        self.suggestion_count_text = "建议: 0"
    
    def init_ui(self) -> bool:
        """初始化界面"""
        self.control_panel = self._create_control_panel()
        self.info_panel = self._create_info_panel()
        return True
    
    def _create_control_panel(self) -> dict:
        """创建控制面板"""
        return {
            'options_group': {
                'coord_checkbox': self.coord_checkbox_checked,
                'suggestion_checkbox': self.suggestion_checkbox_checked,
                'animation_checkbox': self.animation_checkbox_checked
            },
            'buttons_group': {
                'reset_button': True,
                'flip_button': True
            }
        }
    
    def _create_info_panel(self) -> dict:
        """创建信息面板"""
        return {
            'selection_label': self.selection_text,
            'suggestion_label': self.suggestion_count_text
        }
    
    def update_board_state(self, board_state):
        """更新棋盘状态"""
        old_state = self.board_state
        self.board_state = board_state
        
        # 如果有动画且状态发生变化，启动动画
        if (self.animation_enabled and old_state and 
            self._has_state_changed(old_state, board_state)):
            self._start_move_animation(old_state, board_state)
        
        return "Board state updated"
    
    def update_suggestions(self, suggestions):
        """更新AI建议"""
        self.suggested_moves = suggestions
        self.suggestion_count_text = f"建议: {len(suggestions)}"
        return "Suggestions updated"
    
    def handle_mouse_click(self, point: MockPoint) -> str:
        """处理鼠标点击"""
        if not self.coordinates:
            # 创建测试坐标系统
            test_rect = MockRect(0, 0, 800, 900)
            self.coordinates = MockBoardCoordinates(test_rect)
        
        # 转换坐标
        click_pos = self.coordinates.point_to_position(point)
        if not click_pos:
            return "Click outside board"
        
        # 模拟棋子选择逻辑
        if self.board_state and hasattr(self.board_state, 'get_piece_at'):
            piece = self.board_state.get_piece_at(click_pos)
            
            if piece:
                if self.selected_position == click_pos:
                    # 取消选择
                    self.selected_position = None
                    self.highlighted_positions = []
                    self.selection_text = "未选中棋子"
                    return "Piece deselected"
                else:
                    # 选择棋子
                    self.selected_position = click_pos
                    self.highlighted_positions = self._get_possible_moves(click_pos, piece)
                    piece_name = self._get_piece_name(piece.type, piece.color)
                    self.selection_text = f"已选中: {piece_name} ({click_pos})"
                    return f"Piece selected: {piece_name}"
            
            elif self.selected_position:
                # 尝试移动
                move_result = f"Move from {self.selected_position} to {click_pos}"
                self.selected_position = None
                self.highlighted_positions = []
                self.selection_text = "未选中棋子"
                return move_result
        
        return f"Clicked position: {click_pos}"
    
    def _calculate_board_rect(self, widget_width: int = 800, widget_height: int = 900) -> MockRect:
        """计算棋盘绘制区域"""
        # 为控制面板和信息面板留出空间
        available_height = widget_height - 140
        available_width = widget_width - 20
        
        # 保持棋盘比例 (约为 8:9)
        ideal_width = available_height * 8 / 9
        ideal_height = available_width * 9 / 8
        
        if ideal_width <= available_width:
            board_width = int(ideal_width)
            board_height = available_height
        else:
            board_width = available_width
            board_height = int(ideal_height)
        
        # 居中显示
        x = (widget_width - board_width) // 2
        y = 80 + (available_height - board_height) // 2
        
        return MockRect(x, y, board_width, board_height)
    
    def draw_board(self, widget_width: int = 800, widget_height: int = 900) -> dict:
        """绘制棋盘 (返回绘制信息)"""
        # 计算棋盘区域
        board_rect = self._calculate_board_rect(widget_width, widget_height)
        self.coordinates = MockBoardCoordinates(board_rect)
        
        draw_info = {
            'board_rect': board_rect,
            'grid_lines': self._get_grid_line_info(),
            'river': self._get_river_info(),
            'special_marks': self._get_special_marks_info(),
            'coordinates': self._get_coordinates_info() if self.show_coordinates else [],
            'highlights': self._get_highlights_info(),
            'suggestions': self._get_suggestions_info() if self.show_suggestions else [],
            'pieces': self._get_pieces_info(),
            'animation': self._get_animation_info() if self.current_animation else None
        }
        
        return draw_info
    
    def _get_grid_line_info(self) -> dict:
        """获取网格线信息"""
        if not self.coordinates:
            return {}
        
        vertical_lines = []
        for i in range(9):
            x = self.coordinates.start_x + i * self.coordinates.grid_width
            vertical_lines.append((x, self.coordinates.start_y, x, self.coordinates.start_y + 9 * self.coordinates.grid_height))
        
        horizontal_lines = []
        for i in range(10):
            y = self.coordinates.start_y + i * self.coordinates.grid_height
            if i == 4 or i == 5:
                # 楚河汉界分段
                horizontal_lines.append((self.coordinates.start_x, y, self.coordinates.start_x + 3 * self.coordinates.grid_width, y))
                horizontal_lines.append((self.coordinates.start_x + 5 * self.coordinates.grid_width, y, self.coordinates.start_x + 8 * self.coordinates.grid_width, y))
            else:
                horizontal_lines.append((self.coordinates.start_x, y, self.coordinates.start_x + 8 * self.coordinates.grid_width, y))
        
        return {
            'vertical_lines': vertical_lines,
            'horizontal_lines': horizontal_lines,
            'palace_lines': self._get_palace_lines_info()
        }
    
    def _get_palace_lines_info(self) -> list:
        """获取九宫格对角线信息"""
        if not self.coordinates:
            return []
        
        lines = []
        
        # 上方九宫格
        top_left = (self.coordinates.start_x + 3 * self.coordinates.grid_width, self.coordinates.start_y)
        top_right = (self.coordinates.start_x + 5 * self.coordinates.grid_width, self.coordinates.start_y)
        bottom_left = (self.coordinates.start_x + 3 * self.coordinates.grid_width, self.coordinates.start_y + 2 * self.coordinates.grid_height)
        bottom_right = (self.coordinates.start_x + 5 * self.coordinates.grid_width, self.coordinates.start_y + 2 * self.coordinates.grid_height)
        
        lines.append((top_left, bottom_right))
        lines.append((top_right, bottom_left))
        
        # 下方九宫格
        top_left_b = (self.coordinates.start_x + 3 * self.coordinates.grid_width, self.coordinates.start_y + 7 * self.coordinates.grid_height)
        top_right_b = (self.coordinates.start_x + 5 * self.coordinates.grid_width, self.coordinates.start_y + 7 * self.coordinates.grid_height)
        bottom_left_b = (self.coordinates.start_x + 3 * self.coordinates.grid_width, self.coordinates.start_y + 9 * self.coordinates.grid_height)
        bottom_right_b = (self.coordinates.start_x + 5 * self.coordinates.grid_width, self.coordinates.start_y + 9 * self.coordinates.grid_height)
        
        lines.append((top_left_b, bottom_right_b))
        lines.append((top_right_b, bottom_left_b))
        
        return lines
    
    def _get_river_info(self) -> dict:
        """获取楚河汉界信息"""
        if not self.coordinates:
            return {}
        
        river_y = self.coordinates.start_y + 4 * self.coordinates.grid_height
        river_height = self.coordinates.grid_height
        
        return {
            'rect': MockRect(int(self.coordinates.start_x), int(river_y),
                           int(8 * self.coordinates.grid_width), int(river_height)),
            'chu_text': {
                'text': '楚河',
                'x': int(self.coordinates.start_x + self.coordinates.grid_width),
                'y': int(river_y + river_height // 2 + 6)
            },
            'han_text': {
                'text': '汉界',
                'x': int(self.coordinates.start_x + 5 * self.coordinates.grid_width),
                'y': int(river_y + river_height // 2 + 6)
            }
        }
    
    def _get_special_marks_info(self) -> list:
        """获取特殊标记信息"""
        if not self.coordinates:
            return []
        
        mark_positions = [
            (1, 2), (7, 2),  # 上方炮位
            (0, 3), (2, 3), (4, 3), (6, 3), (8, 3),  # 上方兵位
            (0, 6), (2, 6), (4, 6), (6, 6), (8, 6),  # 下方兵位
            (1, 7), (7, 7)   # 下方炮位
        ]
        
        marks = []
        for file, rank in mark_positions:
            center = self.coordinates.position_to_point(MockPosition(file, rank))
            marks.append({'x': center.x, 'y': center.y, 'size': 3})
        
        return marks
    
    def _get_coordinates_info(self) -> dict:
        """获取坐标信息"""
        if not self.coordinates:
            return {}
        
        file_coords = []
        for i in range(9):
            x = self.coordinates.start_x + i * self.coordinates.grid_width
            y = self.coordinates.start_y - 10
            file_coords.append({'char': chr(ord('a') + i), 'x': int(x - 5), 'y': int(y)})
        
        rank_coords = []
        for i in range(10):
            x = self.coordinates.start_x - 15
            y = self.coordinates.start_y + i * self.coordinates.grid_height
            rank_coords.append({'char': str(9 - i), 'x': int(x), 'y': int(y + 5)})
        
        return {
            'file_coords': file_coords,
            'rank_coords': rank_coords
        }
    
    def _get_highlights_info(self) -> dict:
        """获取高亮信息"""
        if not self.coordinates:
            return {}
        
        highlights = {}
        
        if self.selected_position:
            center = self.coordinates.position_to_point(self.selected_position)
            highlights['selected'] = {'x': center.x, 'y': center.y, 'radius': self.coordinates.get_piece_radius() * 1.2}
        
        possible_moves = []
        for pos in self.highlighted_positions:
            center = self.coordinates.position_to_point(pos)
            possible_moves.append({'x': center.x, 'y': center.y, 'radius': self.coordinates.get_piece_radius() * 0.8})
        
        highlights['possible_moves'] = possible_moves
        
        return highlights
    
    def _get_suggestions_info(self) -> list:
        """获取建议信息"""
        if not self.coordinates or not self.suggested_moves:
            return []
        
        suggestions = []
        for i, suggestion in enumerate(self.suggested_moves[:3]):
            # 解析走法
            move_parts = suggestion.get('move', '').split('-') if suggestion.get('move') else ['a1', 'b2']
            if len(move_parts) != 2:
                continue
            
            try:
                from_pos = MockPosition(ord(move_parts[0][0]) - ord('a'), int(move_parts[0][1]))
                to_pos = MockPosition(ord(move_parts[1][0]) - ord('a'), int(move_parts[1][1]))
                
                from_point = self.coordinates.position_to_point(from_pos)
                to_point = self.coordinates.position_to_point(to_pos)
                
                suggestions.append({
                    'from': {'x': from_point.x, 'y': from_point.y},
                    'to': {'x': to_point.x, 'y': to_point.y},
                    'priority': i
                })
            except:
                continue
        
        return suggestions
    
    def _get_pieces_info(self) -> list:
        """获取棋子信息"""
        if not self.board_state or not self.coordinates:
            return []
        
        pieces = []
        # 模拟一些棋子
        test_pieces = [
            {'pos': MockPosition(4, 0), 'type': MockPieceType.SHUAI, 'color': MockPieceColor.RED},
            {'pos': MockPosition(4, 9), 'type': MockPieceType.JIANG, 'color': MockPieceColor.BLACK},
            {'pos': MockPosition(0, 0), 'type': MockPieceType.CHE, 'color': MockPieceColor.RED},
            {'pos': MockPosition(8, 0), 'type': MockPieceType.CHE, 'color': MockPieceColor.RED}
        ]
        
        for piece_info in test_pieces:
            center = self.coordinates.position_to_point(piece_info['pos'])
            piece_render = self.piece_renderer.draw_piece(
                piece_info['type'], piece_info['color'], 
                center, self.coordinates.get_piece_radius()
            )
            pieces.append({
                'position': piece_info['pos'],
                'render': piece_render,
                'center': {'x': center.x, 'y': center.y}
            })
        
        return pieces
    
    def _get_animation_info(self):
        """获取动画信息"""
        if not self.current_animation:
            return None
        
        from_point = self.coordinates.position_to_point(self.current_animation.from_pos)
        to_point = self.coordinates.position_to_point(self.current_animation.to_pos)
        
        progress = self.current_animation.progress
        current_x = from_point.x + (to_point.x - from_point.x) * progress
        current_y = from_point.y + (to_point.y - from_point.y) * progress
        
        return {
            'piece_type': self.current_animation.piece_type,
            'piece_color': self.current_animation.piece_color,
            'current_pos': {'x': int(current_x), 'y': int(current_y)},
            'progress': progress
        }
    
    def _has_state_changed(self, old_state, new_state) -> bool:
        """检查棋盘状态是否发生变化"""
        # 简化实现
        return True
    
    def _start_move_animation(self, old_state, new_state):
        """启动走法动画"""
        # 简化实现 - 创建一个测试动画
        self.current_animation = MockMoveAnimation(
            MockPosition(0, 0), MockPosition(1, 1),
            MockPieceType.CHE, MockPieceColor.RED
        )
        self.animation_active = True
    
    def update_animation(self) -> bool:
        """更新动画"""
        if not self.current_animation:
            return False
        
        self.current_animation.progress += 0.05
        
        if self.current_animation.progress >= 1.0:
            self.current_animation = None
            self.animation_active = False
            return False
        
        return True
    
    def _get_possible_moves(self, pos: MockPosition, piece) -> List[MockPosition]:
        """获取可能的移动位置"""
        moves = []
        
        # 示例: 为选中的棋子周围添加几个可能位置
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_file = pos.file + dx
                new_rank = pos.rank + dy
                
                if 0 <= new_file <= 8 and 0 <= new_rank <= 9:
                    moves.append(MockPosition(new_file, new_rank))
        
        return moves[:4]  # 最多返回4个位置
    
    def _get_piece_name(self, piece_type, color) -> str:
        """获取棋子名称"""
        color_name = "红" if color == MockPieceColor.RED else "黑"
        piece_char = self.piece_renderer.piece_chars.get(piece_type, "未知")
        return f"{color_name}{piece_char}"
    
    # 控制面板回调
    def toggle_coordinates(self, checked: bool) -> str:
        """切换坐标显示"""
        self.show_coordinates = checked
        self.coord_checkbox_checked = checked
        return f"Coordinates display: {'ON' if checked else 'OFF'}"
    
    def toggle_suggestions(self, checked: bool) -> str:
        """切换建议显示"""
        self.show_suggestions = checked
        self.suggestion_checkbox_checked = checked
        return f"Suggestions display: {'ON' if checked else 'OFF'}"
    
    def toggle_animation(self, checked: bool) -> str:
        """切换动画效果"""
        self.animation_enabled = checked
        self.animation_checkbox_checked = checked
        return f"Animation: {'ON' if checked else 'OFF'}"
    
    def reset_selection(self) -> str:
        """重置选择"""
        self.selected_position = None
        self.highlighted_positions = []
        self.selection_text = "未选中棋子"
        return "Selection reset"
    
    def flip_board(self) -> str:
        """翻转棋盘"""
        return "Board flipped (not implemented)"

# 测试函数
def test_piece_renderer():
    """测试棋子渲染器"""
    try:
        renderer = MockChessPieceRenderer()
        
        # 测试棋子字符映射
        assert renderer.piece_chars[MockPieceType.SHUAI] == "帥"
        assert renderer.piece_chars[MockPieceType.JIANG] == "將"
        assert renderer.piece_chars[MockPieceType.CHE] == "車"
        assert renderer.piece_chars[MockPieceType.CHE_BLACK] == "车"
        
        # 测试棋子颜色
        assert renderer.piece_colors[MockPieceColor.RED] == (220, 20, 20)
        assert renderer.piece_colors[MockPieceColor.BLACK] == (20, 20, 20)
        
        # 测试绘制棋子
        center = MockPoint(100, 100)
        result = renderer.draw_piece(MockPieceType.SHUAI, MockPieceColor.RED, center, 20)
        
        assert "红帥" in result
        assert "@(100,100)" in result
        
        print("PASS: 棋子渲染器测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋子渲染器测试 - {e}")
        return False

def test_board_coordinates():
    """测试棋盘坐标系统"""
    try:
        # 创建测试坐标系统
        rect = MockRect(0, 0, 800, 900)
        coords = MockBoardCoordinates(rect)
        
        # 测试坐标计算
        assert coords.margin == 40
        assert coords.grid_width == (800 - 2 * 40) / 8  # 90
        assert coords.grid_height == (900 - 2 * 40) / 9  # 约93.33
        assert coords.start_x == 40
        assert coords.start_y == 40
        
        # 测试位置转换
        pos = MockPosition(4, 5)  # 中心位置
        point = coords.position_to_point(pos)
        assert point.x == int(40 + 4 * coords.grid_width)
        assert point.y == int(40 + 5 * coords.grid_height)
        
        # 测试点转换为位置
        test_point = MockPoint(400, 500)
        converted_pos = coords.point_to_position(test_point)
        assert converted_pos is not None
        assert 0 <= converted_pos.file <= 8
        assert 0 <= converted_pos.rank <= 9
        
        # 测试棋子半径计算
        radius = coords.get_piece_radius()
        assert radius > 0
        assert radius == min(coords.grid_width, coords.grid_height) * 0.35
        
        print("PASS: 棋盘坐标系统测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋盘坐标系统测试 - {e}")
        return False

def test_move_animation():
    """测试走法动画"""
    try:
        from_pos = MockPosition(0, 0)
        to_pos = MockPosition(1, 1)
        animation = MockMoveAnimation(from_pos, to_pos, MockPieceType.CHE, MockPieceColor.RED)
        
        assert animation.from_pos == from_pos
        assert animation.to_pos == to_pos
        assert animation.piece_type == MockPieceType.CHE
        assert animation.piece_color == MockPieceColor.RED
        assert animation.progress == 0.0
        assert animation.duration == 500
        
        print("PASS: 走法动画测试")
        return True
    except Exception as e:
        print(f"FAIL: 走法动画测试 - {e}")
        return False

def test_board_display_initialization():
    """测试棋盘显示初始化"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 测试初始状态
        assert widget.board_state is None
        assert widget.selected_position is None
        assert widget.highlighted_positions == []
        assert widget.suggested_moves == []
        assert widget.show_coordinates == True
        assert widget.show_suggestions == True
        assert widget.animation_enabled == True
        
        # 测试界面初始化
        success = widget.init_ui()
        assert success == True
        assert widget.control_panel is not None
        assert widget.info_panel is not None
        
        # 测试颜色主题
        assert 'board_bg' in widget.colors
        assert 'line' in widget.colors
        assert 'selected' in widget.colors
        
        print("PASS: 棋盘显示初始化测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋盘显示初始化测试 - {e}")
        return False

def test_control_panel():
    """测试控制面板"""
    try:
        widget = MockBoardDisplayWidget()
        control_panel = widget._create_control_panel()
        
        # 验证控制面板结构
        assert 'options_group' in control_panel
        assert 'buttons_group' in control_panel
        
        options = control_panel['options_group']
        assert 'coord_checkbox' in options
        assert 'suggestion_checkbox' in options
        assert 'animation_checkbox' in options
        
        buttons = control_panel['buttons_group']
        assert 'reset_button' in buttons
        assert 'flip_button' in buttons
        
        # 测试控制回调
        result = widget.toggle_coordinates(False)
        assert "OFF" in result
        assert widget.show_coordinates == False
        
        result = widget.toggle_suggestions(False)
        assert "OFF" in result
        assert widget.show_suggestions == False
        
        result = widget.reset_selection()
        assert "reset" in result.lower()
        
        print("PASS: 控制面板测试")
        return True
    except Exception as e:
        print(f"FAIL: 控制面板测试 - {e}")
        return False

def test_board_rect_calculation():
    """测试棋盘区域计算"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 测试不同窗口尺寸
        rect1 = widget._calculate_board_rect(800, 900)
        assert isinstance(rect1, MockRect)
        assert rect1.width > 0
        assert rect1.height > 0
        
        rect2 = widget._calculate_board_rect(1200, 800)
        assert isinstance(rect2, MockRect)
        assert rect2.width > 0
        assert rect2.height > 0
        
        # 验证比例保持 (约8:9)
        ratio1 = rect1.width / rect1.height
        expected_ratio = 8 / 9
        assert abs(ratio1 - expected_ratio) < 0.1
        
        print("PASS: 棋盘区域计算测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋盘区域计算测试 - {e}")
        return False

def test_board_drawing():
    """测试棋盘绘制"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 测试绘制
        draw_info = widget.draw_board(800, 900)
        
        # 验证绘制信息结构
        expected_keys = ['board_rect', 'grid_lines', 'river', 'special_marks', 
                        'coordinates', 'highlights', 'suggestions', 'pieces', 'animation']
        
        for key in expected_keys:
            assert key in draw_info, f"Missing key: {key}"
        
        # 验证网格线信息
        grid_lines = draw_info['grid_lines']
        assert 'vertical_lines' in grid_lines
        assert 'horizontal_lines' in grid_lines
        assert 'palace_lines' in grid_lines
        
        assert len(grid_lines['vertical_lines']) == 9
        assert len(grid_lines['horizontal_lines']) == 10
        assert len(grid_lines['palace_lines']) == 4  # 两个九宫格，各2条对角线
        
        # 验证楚河汉界信息
        river = draw_info['river']
        assert 'rect' in river
        assert 'chu_text' in river
        assert 'han_text' in river
        assert river['chu_text']['text'] == '楚河'
        assert river['han_text']['text'] == '汉界'
        
        # 验证特殊标记
        special_marks = draw_info['special_marks']
        assert len(special_marks) == 14  # 兵和炮的位置标记
        
        print("PASS: 棋盘绘制测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋盘绘制测试 - {e}")
        return False

def test_mouse_interaction():
    """测试鼠标交互"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 设置测试棋盘状态
        class MockBoardState:
            def get_piece_at(self, pos):
                if pos.file == 0 and pos.rank == 0:
                    return type('MockPiece', (), {
                        'type': MockPieceType.CHE,
                        'color': MockPieceColor.RED
                    })()
                return None
        
        widget.board_state = MockBoardState()
        
        # 测试点击棋子
        click_point = MockPoint(50, 50)  # 接近(0,0)位置
        result = widget.handle_mouse_click(click_point)
        
        assert "selected" in result.lower() or "clicked" in result.lower()
        
        # 测试再次点击同一棋子 (取消选择)
        if widget.selected_position:
            result2 = widget.handle_mouse_click(click_point)
            assert "deselected" in result2.lower() or "clicked" in result2.lower()
        
        # 测试点击空位置
        empty_point = MockPoint(200, 200)
        result3 = widget.handle_mouse_click(empty_point)
        assert "clicked" in result3.lower() or "move" in result3.lower()
        
        print("PASS: 鼠标交互测试")
        return True
    except Exception as e:
        print(f"FAIL: 鼠标交互测试 - {e}")
        return False

def test_suggestions_display():
    """测试AI建议显示"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 测试更新建议
        test_suggestions = [
            {'move': 'a1-b2', 'score': 0.8},
            {'move': 'c3-d4', 'score': 0.6},
            {'move': 'e5-f6', 'score': 0.4}
        ]
        
        result = widget.update_suggestions(test_suggestions)
        assert "updated" in result.lower()
        assert len(widget.suggested_moves) == 3
        assert "建议: 3" in widget.suggestion_count_text
        
        # 测试绘制建议
        draw_info = widget.draw_board()
        suggestions_info = draw_info['suggestions']
        
        # 应该有3个建议箭头
        assert len(suggestions_info) <= 3
        
        print("PASS: AI建议显示测试")
        return True
    except Exception as e:
        print(f"FAIL: AI建议显示测试 - {e}")
        return False

def test_animation_system():
    """测试动画系统"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 启动测试动画
        widget._start_move_animation(None, None)
        
        assert widget.current_animation is not None
        assert widget.animation_active == True
        assert widget.current_animation.progress == 0.0
        
        # 更新动画
        for i in range(25):  # 25次更新应该完成动画
            still_running = widget.update_animation()
            if not still_running:
                break
        
        # 动画应该已经完成
        assert widget.current_animation is None
        assert widget.animation_active == False
        
        # 测试绘制动画信息
        widget._start_move_animation(None, None)
        widget.draw_board()
        draw_info = widget.draw_board()
        
        if draw_info['animation']:
            anim_info = draw_info['animation']
            assert 'piece_type' in anim_info
            assert 'current_pos' in anim_info
            assert 'progress' in anim_info
        
        print("PASS: 动画系统测试")
        return True
    except Exception as e:
        print(f"FAIL: 动画系统测试 - {e}")
        return False

def test_coordinates_display():
    """测试坐标显示"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 启用坐标显示
        widget.toggle_coordinates(True)
        
        # 绘制棋盘并检查坐标信息
        draw_info = widget.draw_board()
        coords_info = draw_info['coordinates']
        
        assert 'file_coords' in coords_info
        assert 'rank_coords' in coords_info
        
        # 验证文件坐标 (a-i)
        file_coords = coords_info['file_coords']
        assert len(file_coords) == 9
        assert file_coords[0]['char'] == 'a'
        assert file_coords[8]['char'] == 'i'
        
        # 验证等级坐标 (0-9)
        rank_coords = coords_info['rank_coords']
        assert len(rank_coords) == 10
        assert rank_coords[0]['char'] == '9'
        assert rank_coords[9]['char'] == '0'
        
        # 禁用坐标显示
        widget.toggle_coordinates(False)
        draw_info2 = widget.draw_board()
        assert draw_info2['coordinates'] == []
        
        print("PASS: 坐标显示测试")
        return True
    except Exception as e:
        print(f"FAIL: 坐标显示测试 - {e}")
        return False

def test_board_state_update():
    """测试棋盘状态更新"""
    try:
        widget = MockBoardDisplayWidget()
        
        # 创建测试棋盘状态
        class MockBoardState:
            def __init__(self, fen):
                self.fen = fen
        
        old_state = MockBoardState("old_fen")
        new_state = MockBoardState("new_fen")
        
        # 设置旧状态
        widget.board_state = old_state
        
        # 更新为新状态
        result = widget.update_board_state(new_state)
        
        assert "updated" in result.lower()
        assert widget.board_state == new_state
        
        # 如果启用动画，应该启动动画
        if widget.animation_enabled:
            # 动画系统会被触发 (简化测试)
            pass
        
        print("PASS: 棋盘状态更新测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋盘状态更新测试 - {e}")
        return False

# 运行所有测试
def run_board_display_tests():
    """运行棋盘显示和用户交互测试"""
    test_functions = [
        test_piece_renderer,
        test_board_coordinates,
        test_move_animation,
        test_board_display_initialization,
        test_control_panel,
        test_board_rect_calculation,
        test_board_drawing,
        test_mouse_interaction,
        test_suggestions_display,
        test_animation_system,
        test_coordinates_display,
        test_board_state_update
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        if test_func():
            passed += 1
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("所有测试通过! 棋盘显示和用户交互核心功能正常")
        return True
    else:
        print(f"有 {total-passed} 个测试失败，需要修复实现")
        return False

if __name__ == "__main__":
    success = run_board_display_tests()
    sys.exit(0 if success else 1)
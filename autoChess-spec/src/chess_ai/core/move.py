#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
走法数据模型
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import re

from chess_ai.core.piece import Piece, Position, PieceType


class MoveType(Enum):
    """走法类型"""
    NORMAL = "普通移动"          # 普通移动
    CAPTURE = "吃子"            # 吃子
    CASTLING = "易位"           # 易位（象棋中不存在，为兼容性保留）
    EN_PASSANT = "吃过路兵"     # 吃过路兵（象棋中不存在，为兼容性保留）
    PROMOTION = "升变"          # 升变（象棋中不存在，为兼容性保留）


@dataclass
class Move:
    """走法类"""
    from_position: Position
    to_position: Position
    piece: Piece
    captured_piece: Optional[Piece] = None
    move_type: MoveType = MoveType.NORMAL
    evaluation_score: float = 0.0  # AI评估分数
    confidence: float = 1.0        # 识别置信度
    
    def __post_init__(self):
        """验证走法数据"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"置信度必须在0.0-1.0之间，当前值: {self.confidence}")
        
        # 自动判断走法类型
        if self.captured_piece is not None:
            self.move_type = MoveType.CAPTURE
    
    @property
    def is_capture(self) -> bool:
        """是否为吃子走法"""
        return self.captured_piece is not None or self.move_type == MoveType.CAPTURE
    
    @property
    def is_check(self) -> bool:
        """是否为将军走法（需要在棋局上下文中判断）"""
        # 此方法需要在BoardState中实现具体逻辑
        return False
    
    @property
    def distance(self) -> int:
        """计算移动距离（曼哈顿距离）"""
        return (abs(self.to_position.row - self.from_position.row) + 
                abs(self.to_position.col - self.from_position.col))
    
    def to_chinese_notation(self) -> str:
        """转换为中文记法"""
        piece_name = self.piece.base_type
        from_pos = self.from_position.to_chinese_notation()
        to_pos = self.to_position.to_chinese_notation()
        
        if self.is_capture:
            return f"{piece_name}{from_pos}吃{to_pos}"
        else:
            return f"{piece_name}{from_pos}到{to_pos}"
    
    def to_algebraic_notation(self) -> str:
        """转换为代数记法"""
        from_pos = self.from_position.to_algebraic_notation()
        to_pos = self.to_position.to_algebraic_notation()
        
        if self.is_capture:
            return f"{from_pos}x{to_pos}"
        else:
            return f"{from_pos}-{to_pos}"
    
    def to_ucci_notation(self) -> str:
        """转换为UCCI记法（象棋引擎标准）"""
        from_pos = f"{self.from_position.col}{self.from_position.row}"
        to_pos = f"{self.to_position.col}{self.to_position.row}"
        return f"{from_pos}{to_pos}"
    
    @classmethod
    def from_ucci_notation(cls, ucci_move: str, piece: Piece, captured_piece: Optional[Piece] = None) -> "Move":
        """从UCCI记法创建走法"""
        if len(ucci_move) != 4:
            raise ValueError(f"UCCI走法格式错误: {ucci_move}")
        
        try:
            from_col = int(ucci_move[0])
            from_row = int(ucci_move[1])
            to_col = int(ucci_move[2])
            to_row = int(ucci_move[3])
            
            from_position = Position(from_row, from_col)
            to_position = Position(to_row, to_col)
            
            return cls(
                from_position=from_position,
                to_position=to_position,
                piece=piece,
                captured_piece=captured_piece
            )
        except ValueError as e:
            raise ValueError(f"UCCI走法解析失败: {ucci_move}, 错误: {e}")
    
    @classmethod
    def from_algebraic_notation(cls, algebraic_move: str, piece: Piece, captured_piece: Optional[Piece] = None) -> "Move":
        """从代数记法创建走法"""
        # 匹配格式: a1-b2 或 a1xb2
        pattern = r'^([a-i][1-9]|[a-i]10)([-x])([a-i][1-9]|[a-i]10)$'
        match = re.match(pattern, algebraic_move)
        
        if not match:
            raise ValueError(f"代数记法格式错误: {algebraic_move}")
        
        from_str, separator, to_str = match.groups()
        
        def parse_position(pos_str: str) -> Position:
            col = ord(pos_str[0]) - ord('a')
            row = int(pos_str[1:]) - 1
            return Position(row, col)
        
        from_position = parse_position(from_str)
        to_position = parse_position(to_str)
        
        move_type = MoveType.CAPTURE if separator == 'x' else MoveType.NORMAL
        
        return cls(
            from_position=from_position,
            to_position=to_position,
            piece=piece,
            captured_piece=captured_piece,
            move_type=move_type
        )
    
    def is_valid_for_piece_type(self) -> bool:
        """验证走法是否符合棋子类型的移动规则"""
        piece_type = self.piece.piece_type
        from_row, from_col = self.from_position.row, self.from_position.col
        to_row, to_col = self.to_position.row, self.to_position.col
        
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # 帅/将的移动规则
        if piece_type in [PieceType.RED_KING, PieceType.BLACK_KING]:
            # 只能在九宫格内移动，每次只能移动一格
            if piece_type == PieceType.RED_KING:
                if not (0 <= to_row <= 2 and 3 <= to_col <= 5):
                    return False
            else:  # BLACK_KING
                if not (7 <= to_row <= 9 and 3 <= to_col <= 5):
                    return False
            return abs(row_diff) + abs(col_diff) == 1
        
        # 仕/士的移动规则
        elif piece_type in [PieceType.RED_ADVISOR, PieceType.BLACK_ADVISOR]:
            # 只能在九宫格内斜向移动
            if piece_type == PieceType.RED_ADVISOR:
                if not (0 <= to_row <= 2 and 3 <= to_col <= 5):
                    return False
            else:  # BLACK_ADVISOR
                if not (7 <= to_row <= 9 and 3 <= to_col <= 5):
                    return False
            return abs(row_diff) == 1 and abs(col_diff) == 1
        
        # 相/象的移动规则
        elif piece_type in [PieceType.RED_ELEPHANT, PieceType.BLACK_ELEPHANT]:
            # 不能过河，斜向移动两格
            if piece_type == PieceType.RED_ELEPHANT:
                if to_row > 4:  # 不能过河
                    return False
            else:  # BLACK_ELEPHANT
                if to_row < 5:  # 不能过河
                    return False
            return abs(row_diff) == 2 and abs(col_diff) == 2
        
        # 马的移动规则
        elif piece_type in [PieceType.RED_HORSE, PieceType.BLACK_HORSE]:
            # 日字形移动
            return ((abs(row_diff) == 2 and abs(col_diff) == 1) or 
                    (abs(row_diff) == 1 and abs(col_diff) == 2))
        
        # 车的移动规则
        elif piece_type in [PieceType.RED_CHARIOT, PieceType.BLACK_CHARIOT]:
            # 直线移动
            return row_diff == 0 or col_diff == 0
        
        # 炮的移动规则
        elif piece_type in [PieceType.RED_CANNON, PieceType.BLACK_CANNON]:
            # 直线移动（吃子时需要翻山，移动时不能翻山）
            return row_diff == 0 or col_diff == 0
        
        # 兵/卒的移动规则
        elif piece_type in [PieceType.RED_PAWN, PieceType.BLACK_PAWN]:
            if piece_type == PieceType.RED_PAWN:
                # 红兵向前或过河后可横移
                if from_row < 5:  # 未过河
                    return row_diff == 1 and col_diff == 0
                else:  # 已过河
                    return ((row_diff == 1 and col_diff == 0) or 
                            (row_diff == 0 and abs(col_diff) == 1))
            else:  # BLACK_PAWN
                # 黑卒向前或过河后可横移
                if from_row > 4:  # 未过河
                    return row_diff == -1 and col_diff == 0
                else:  # 已过河
                    return ((row_diff == -1 and col_diff == 0) or 
                            (row_diff == 0 and abs(col_diff) == 1))
        
        return False
    
    def reverse(self) -> "Move":
        """返回反向走法"""
        return Move(
            from_position=self.to_position,
            to_position=self.from_position,
            piece=self.piece.move_to(self.to_position),
            captured_piece=None,  # 反向走法不涉及吃子
            move_type=MoveType.NORMAL,
            evaluation_score=-self.evaluation_score,
            confidence=self.confidence
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.to_chinese_notation()
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"Move({self.from_position}->{self.to_position}, "
                f"{self.piece.piece_type.value}, "
                f"score={self.evaluation_score:.2f})")


@dataclass
class MoveHistory:
    """走法历史记录"""
    moves: List[Move]
    current_index: int = -1
    
    def add_move(self, move: Move) -> None:
        """添加走法"""
        # 如果当前不在最后位置，删除后续走法
        if self.current_index < len(self.moves) - 1:
            self.moves = self.moves[:self.current_index + 1]
        
        self.moves.append(move)
        self.current_index = len(self.moves) - 1
    
    def undo_move(self) -> Optional[Move]:
        """悔棋"""
        if self.current_index >= 0:
            move = self.moves[self.current_index]
            self.current_index -= 1
            return move
        return None
    
    def redo_move(self) -> Optional[Move]:
        """重做走法"""
        if self.current_index < len(self.moves) - 1:
            self.current_index += 1
            return self.moves[self.current_index]
        return None
    
    def get_current_move(self) -> Optional[Move]:
        """获取当前走法"""
        if 0 <= self.current_index < len(self.moves):
            return self.moves[self.current_index]
        return None
    
    def get_last_move(self) -> Optional[Move]:
        """获取最后一个走法"""
        if self.moves:
            return self.moves[-1]
        return None
    
    def get_moves_from_position(self, start_index: int = 0) -> List[Move]:
        """从指定位置获取走法列表"""
        if start_index < 0:
            start_index = 0
        return self.moves[start_index:self.current_index + 1]
    
    def clear(self) -> None:
        """清空历史"""
        self.moves.clear()
        self.current_index = -1
    
    def to_ucci_list(self) -> List[str]:
        """转换为UCCI记法列表"""
        return [move.to_ucci_notation() for move in self.get_moves_from_position()]
    
    def to_algebraic_list(self) -> List[str]:
        """转换为代数记法列表"""
        return [move.to_algebraic_notation() for move in self.get_moves_from_position()]
    
    def __len__(self) -> int:
        """历史长度"""
        return len(self.moves)
    
    def __iter__(self):
        """迭代器"""
        return iter(self.moves[:self.current_index + 1])


class MoveValidator:
    """走法验证器"""
    
    @staticmethod
    def is_position_valid(position: Position) -> bool:
        """验证位置是否有效"""
        return 0 <= position.row <= 9 and 0 <= position.col <= 8
    
    @staticmethod
    def is_move_format_valid(move_str: str) -> bool:
        """验证走法字符串格式"""
        # UCCI格式: 0123 (4位数字)
        if re.match(r'^\d{4}$', move_str):
            return True
        
        # 代数记法: a1-b2 或 a1xb2
        if re.match(r'^[a-i][1-9]|[a-i]10[-x][a-i][1-9]|[a-i]10$', move_str):
            return True
        
        return False
    
    @classmethod
    def validate_move_basic(cls, move: Move) -> List[str]:
        """基础走法验证"""
        errors = []
        
        # 验证位置有效性
        if not cls.is_position_valid(move.from_position):
            errors.append(f"起始位置无效: {move.from_position}")
        
        if not cls.is_position_valid(move.to_position):
            errors.append(f"目标位置无效: {move.to_position}")
        
        # 验证不能原地不动
        if move.from_position == move.to_position:
            errors.append("起始位置和目标位置相同")
        
        # 验证走法是否符合棋子类型
        if not move.is_valid_for_piece_type():
            errors.append(f"走法不符合{move.piece.piece_type.value}的移动规则")
        
        return errors
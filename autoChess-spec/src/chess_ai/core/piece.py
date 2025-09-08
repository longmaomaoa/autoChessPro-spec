#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋子数据模型
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


class PieceType(Enum):
    """棋子类型枚举"""
    # 红方棋子
    RED_KING = "红帅"         # 帅
    RED_ADVISOR = "红仕"      # 仕
    RED_ELEPHANT = "红相"     # 相
    RED_HORSE = "红马"        # 马
    RED_CHARIOT = "红车"      # 车
    RED_CANNON = "红炮"       # 炮
    RED_PAWN = "红兵"         # 兵
    
    # 黑方棋子
    BLACK_KING = "黑将"       # 将
    BLACK_ADVISOR = "黑士"    # 士
    BLACK_ELEPHANT = "黑象"   # 象
    BLACK_HORSE = "黑马"      # 马
    BLACK_CHARIOT = "黑车"    # 车
    BLACK_CANNON = "黑炮"     # 炮
    BLACK_PAWN = "黑卒"       # 卒


class PieceColor(Enum):
    """棋子颜色枚举"""
    RED = "红"
    BLACK = "黑"


@dataclass(frozen=True)
class Position:
    """棋盘位置"""
    row: int  # 行 (0-9)
    col: int  # 列 (0-8)
    
    def __post_init__(self):
        """验证位置有效性"""
        if not (0 <= self.row <= 9):
            raise ValueError(f"行坐标必须在0-9之间，当前值: {self.row}")
        if not (0 <= self.col <= 8):
            raise ValueError(f"列坐标必须在0-8之间，当前值: {self.col}")
    
    def to_chinese_notation(self) -> str:
        """转换为中文记法"""
        chinese_cols = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
        return f"{chinese_cols[self.col]}{self.row + 1}"
    
    def to_algebraic_notation(self) -> str:
        """转换为代数记法 (a1-i10)"""
        return f"{'abcdefghi'[self.col]}{self.row + 1}"


@dataclass
class Piece:
    """棋子类"""
    piece_type: PieceType
    position: Position
    confidence: float = 1.0  # 识别置信度 (0.0-1.0)
    
    def __post_init__(self):
        """验证棋子数据"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"置信度必须在0.0-1.0之间，当前值: {self.confidence}")
    
    @property
    def color(self) -> PieceColor:
        """获取棋子颜色"""
        if self.piece_type.value.startswith("红"):
            return PieceColor.RED
        else:
            return PieceColor.BLACK
    
    @property
    def base_type(self) -> str:
        """获取棋子基础类型（去除颜色）"""
        type_mapping = {
            PieceType.RED_KING: "帅",
            PieceType.BLACK_KING: "将",
            PieceType.RED_ADVISOR: "仕",
            PieceType.BLACK_ADVISOR: "士",
            PieceType.RED_ELEPHANT: "相",
            PieceType.BLACK_ELEPHANT: "象",
            PieceType.RED_HORSE: "马",
            PieceType.BLACK_HORSE: "马",
            PieceType.RED_CHARIOT: "车",
            PieceType.BLACK_CHARIOT: "车",
            PieceType.RED_CANNON: "炮",
            PieceType.BLACK_CANNON: "炮",
            PieceType.RED_PAWN: "兵",
            PieceType.BLACK_PAWN: "卒",
        }
        return type_mapping[self.piece_type]
    
    @property
    def unicode_symbol(self) -> str:
        """获取Unicode象棋符号"""
        symbols = {
            PieceType.RED_KING: "帥",
            PieceType.BLACK_KING: "將",
            PieceType.RED_ADVISOR: "仕",
            PieceType.BLACK_ADVISOR: "士",
            PieceType.RED_ELEPHANT: "相",
            PieceType.BLACK_ELEPHANT: "象",
            PieceType.RED_HORSE: "馬",
            PieceType.BLACK_HORSE: "馬",
            PieceType.RED_CHARIOT: "車",
            PieceType.BLACK_CHARIOT: "車",
            PieceType.RED_CANNON: "砲",
            PieceType.BLACK_CANNON: "砲",
            PieceType.RED_PAWN: "兵",
            PieceType.BLACK_PAWN: "卒",
        }
        return symbols[self.piece_type]
    
    @property
    def fen_symbol(self) -> str:
        """获取FEN记录符号"""
        fen_mapping = {
            PieceType.RED_KING: "K",
            PieceType.BLACK_KING: "k",
            PieceType.RED_ADVISOR: "A",
            PieceType.BLACK_ADVISOR: "a",
            PieceType.RED_ELEPHANT: "E",
            PieceType.BLACK_ELEPHANT: "e",
            PieceType.RED_HORSE: "H",
            PieceType.BLACK_HORSE: "h",
            PieceType.RED_CHARIOT: "R",
            PieceType.BLACK_CHARIOT: "r",
            PieceType.RED_CANNON: "C",
            PieceType.BLACK_CANNON: "c",
            PieceType.RED_PAWN: "P",
            PieceType.BLACK_PAWN: "p",
        }
        return fen_mapping[self.piece_type]
    
    def move_to(self, new_position: Position) -> "Piece":
        """移动棋子到新位置，返回新的棋子实例"""
        return Piece(
            piece_type=self.piece_type,
            position=new_position,
            confidence=self.confidence
        )
    
    def is_same_color(self, other: "Piece") -> bool:
        """判断是否同色棋子"""
        return self.color == other.color
    
    def is_opponent_color(self, other: "Piece") -> bool:
        """判断是否对方棋子"""
        return self.color != other.color
    
    @classmethod
    def from_fen_symbol(cls, fen_symbol: str, position: Position, confidence: float = 1.0) -> "Piece":
        """从FEN符号创建棋子"""
        fen_to_type = {
            "K": PieceType.RED_KING,
            "k": PieceType.BLACK_KING,
            "A": PieceType.RED_ADVISOR,
            "a": PieceType.BLACK_ADVISOR,
            "E": PieceType.RED_ELEPHANT,
            "e": PieceType.BLACK_ELEPHANT,
            "H": PieceType.RED_HORSE,
            "h": PieceType.BLACK_HORSE,
            "R": PieceType.RED_CHARIOT,
            "r": PieceType.BLACK_CHARIOT,
            "C": PieceType.RED_CANNON,
            "c": PieceType.BLACK_CANNON,
            "P": PieceType.RED_PAWN,
            "p": PieceType.BLACK_PAWN,
        }
        
        if fen_symbol not in fen_to_type:
            raise ValueError(f"无效的FEN符号: {fen_symbol}")
        
        return cls(
            piece_type=fen_to_type[fen_symbol],
            position=position,
            confidence=confidence
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.piece_type.value}@{self.position.to_chinese_notation()}"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"Piece({self.piece_type.value}, {self.position}, confidence={self.confidence:.2f})"


class PieceUtils:
    """棋子工具类"""
    
    @staticmethod
    def get_initial_pieces() -> Dict[Position, Piece]:
        """获取初始棋局的所有棋子"""
        pieces = {}
        
        # 红方棋子 (下方)
        # 第一行
        pieces[Position(0, 0)] = Piece(PieceType.RED_CHARIOT, Position(0, 0))
        pieces[Position(0, 1)] = Piece(PieceType.RED_HORSE, Position(0, 1))
        pieces[Position(0, 2)] = Piece(PieceType.RED_ELEPHANT, Position(0, 2))
        pieces[Position(0, 3)] = Piece(PieceType.RED_ADVISOR, Position(0, 3))
        pieces[Position(0, 4)] = Piece(PieceType.RED_KING, Position(0, 4))
        pieces[Position(0, 5)] = Piece(PieceType.RED_ADVISOR, Position(0, 5))
        pieces[Position(0, 6)] = Piece(PieceType.RED_ELEPHANT, Position(0, 6))
        pieces[Position(0, 7)] = Piece(PieceType.RED_HORSE, Position(0, 7))
        pieces[Position(0, 8)] = Piece(PieceType.RED_CHARIOT, Position(0, 8))
        
        # 红方炮
        pieces[Position(2, 1)] = Piece(PieceType.RED_CANNON, Position(2, 1))
        pieces[Position(2, 7)] = Piece(PieceType.RED_CANNON, Position(2, 7))
        
        # 红方兵
        for col in [0, 2, 4, 6, 8]:
            pieces[Position(3, col)] = Piece(PieceType.RED_PAWN, Position(3, col))
        
        # 黑方棋子 (上方)
        # 第十行
        pieces[Position(9, 0)] = Piece(PieceType.BLACK_CHARIOT, Position(9, 0))
        pieces[Position(9, 1)] = Piece(PieceType.BLACK_HORSE, Position(9, 1))
        pieces[Position(9, 2)] = Piece(PieceType.BLACK_ELEPHANT, Position(9, 2))
        pieces[Position(9, 3)] = Piece(PieceType.BLACK_ADVISOR, Position(9, 3))
        pieces[Position(9, 4)] = Piece(PieceType.BLACK_KING, Position(9, 4))
        pieces[Position(9, 5)] = Piece(PieceType.BLACK_ADVISOR, Position(9, 5))
        pieces[Position(9, 6)] = Piece(PieceType.BLACK_ELEPHANT, Position(9, 6))
        pieces[Position(9, 7)] = Piece(PieceType.BLACK_HORSE, Position(9, 7))
        pieces[Position(9, 8)] = Piece(PieceType.BLACK_CHARIOT, Position(9, 8))
        
        # 黑方炮
        pieces[Position(7, 1)] = Piece(PieceType.BLACK_CANNON, Position(7, 1))
        pieces[Position(7, 7)] = Piece(PieceType.BLACK_CANNON, Position(7, 7))
        
        # 黑方卒
        for col in [0, 2, 4, 6, 8]:
            pieces[Position(6, col)] = Piece(PieceType.BLACK_PAWN, Position(6, col))
        
        return pieces
    
    @staticmethod
    def get_piece_value(piece_type: PieceType) -> int:
        """获取棋子价值分数"""
        values = {
            PieceType.RED_KING: 10000,
            PieceType.BLACK_KING: 10000,
            PieceType.RED_CHARIOT: 900,
            PieceType.BLACK_CHARIOT: 900,
            PieceType.RED_CANNON: 450,
            PieceType.BLACK_CANNON: 450,
            PieceType.RED_HORSE: 400,
            PieceType.BLACK_HORSE: 400,
            PieceType.RED_ELEPHANT: 200,
            PieceType.BLACK_ELEPHANT: 200,
            PieceType.RED_ADVISOR: 200,
            PieceType.BLACK_ADVISOR: 200,
            PieceType.RED_PAWN: 100,
            PieceType.BLACK_PAWN: 100,
        }
        return values[piece_type]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋子模型单元测试
"""

import pytest
from chess_ai.core.piece import (
    Piece, Position, PieceType, PieceColor, PieceUtils
)


class TestPosition:
    """位置类测试"""
    
    def test_valid_position_creation(self):
        """测试有效位置创建"""
        pos = Position(0, 0)
        assert pos.row == 0
        assert pos.col == 0
        
        pos = Position(9, 8)
        assert pos.row == 9
        assert pos.col == 8
    
    def test_invalid_position_creation(self):
        """测试无效位置创建"""
        with pytest.raises(ValueError):
            Position(-1, 0)
        
        with pytest.raises(ValueError):
            Position(10, 0)
        
        with pytest.raises(ValueError):
            Position(0, -1)
        
        with pytest.raises(ValueError):
            Position(0, 9)
    
    def test_position_to_chinese_notation(self):
        """测试位置转中文记法"""
        pos = Position(0, 0)
        assert pos.to_chinese_notation() == "一1"
        
        pos = Position(9, 8)
        assert pos.to_chinese_notation() == "九10"
    
    def test_position_to_algebraic_notation(self):
        """测试位置转代数记法"""
        pos = Position(0, 0)
        assert pos.to_algebraic_notation() == "a1"
        
        pos = Position(9, 8)
        assert pos.to_algebraic_notation() == "i10"


class TestPiece:
    """棋子类测试"""
    
    def test_piece_creation(self):
        """测试棋子创建"""
        pos = Position(0, 4)
        piece = Piece(PieceType.RED_KING, pos)
        
        assert piece.piece_type == PieceType.RED_KING
        assert piece.position == pos
        assert piece.confidence == 1.0
    
    def test_piece_color_property(self):
        """测试棋子颜色属性"""
        red_king = Piece(PieceType.RED_KING, Position(0, 4))
        assert red_king.color == PieceColor.RED
        
        black_king = Piece(PieceType.BLACK_KING, Position(9, 4))
        assert black_king.color == PieceColor.BLACK
    
    def test_piece_base_type_property(self):
        """测试棋子基础类型属性"""
        red_king = Piece(PieceType.RED_KING, Position(0, 4))
        assert red_king.base_type == "帅"
        
        black_king = Piece(PieceType.BLACK_KING, Position(9, 4))
        assert black_king.base_type == "将"
        
        red_horse = Piece(PieceType.RED_HORSE, Position(0, 1))
        black_horse = Piece(PieceType.BLACK_HORSE, Position(9, 1))
        assert red_horse.base_type == black_horse.base_type == "马"
    
    def test_piece_fen_symbol(self):
        """测试棋子FEN符号"""
        red_king = Piece(PieceType.RED_KING, Position(0, 4))
        assert red_king.fen_symbol == "K"
        
        black_king = Piece(PieceType.BLACK_KING, Position(9, 4))
        assert black_king.fen_symbol == "k"
    
    def test_piece_from_fen_symbol(self):
        """测试从FEN符号创建棋子"""
        pos = Position(0, 4)
        piece = Piece.from_fen_symbol("K", pos)
        
        assert piece.piece_type == PieceType.RED_KING
        assert piece.position == pos
        assert piece.confidence == 1.0
        
        with pytest.raises(ValueError):
            Piece.from_fen_symbol("X", pos)
    
    def test_piece_move_to(self):
        """测试棋子移动"""
        pos1 = Position(0, 4)
        pos2 = Position(1, 4)
        piece = Piece(PieceType.RED_KING, pos1)
        
        moved_piece = piece.move_to(pos2)
        
        assert moved_piece.position == pos2
        assert moved_piece.piece_type == piece.piece_type
        assert moved_piece.confidence == piece.confidence
        # 原棋子不变
        assert piece.position == pos1
    
    def test_piece_same_color(self):
        """测试同色棋子判断"""
        red_king = Piece(PieceType.RED_KING, Position(0, 4))
        red_horse = Piece(PieceType.RED_HORSE, Position(0, 1))
        black_king = Piece(PieceType.BLACK_KING, Position(9, 4))
        
        assert red_king.is_same_color(red_horse)
        assert not red_king.is_same_color(black_king)
    
    def test_piece_opponent_color(self):
        """测试对方棋子判断"""
        red_king = Piece(PieceType.RED_KING, Position(0, 4))
        black_king = Piece(PieceType.BLACK_KING, Position(9, 4))
        
        assert red_king.is_opponent_color(black_king)
        assert black_king.is_opponent_color(red_king)
    
    def test_invalid_confidence(self):
        """测试无效置信度"""
        pos = Position(0, 4)
        
        with pytest.raises(ValueError):
            Piece(PieceType.RED_KING, pos, confidence=-0.1)
        
        with pytest.raises(ValueError):
            Piece(PieceType.RED_KING, pos, confidence=1.1)


class TestPieceUtils:
    """棋子工具类测试"""
    
    def test_get_initial_pieces(self):
        """测试获取初始棋局"""
        pieces = PieceUtils.get_initial_pieces()
        
        # 检查总棋子数量 (32个)
        assert len(pieces) == 32
        
        # 检查红帅位置
        red_king_pos = Position(0, 4)
        assert red_king_pos in pieces
        assert pieces[red_king_pos].piece_type == PieceType.RED_KING
        
        # 检查黑将位置
        black_king_pos = Position(9, 4)
        assert black_king_pos in pieces
        assert pieces[black_king_pos].piece_type == PieceType.BLACK_KING
        
        # 检查红兵数量
        red_pawns = [p for p in pieces.values() if p.piece_type == PieceType.RED_PAWN]
        assert len(red_pawns) == 5
        
        # 检查黑卒数量
        black_pawns = [p for p in pieces.values() if p.piece_type == PieceType.BLACK_PAWN]
        assert len(black_pawns) == 5
    
    def test_get_piece_value(self):
        """测试获取棋子价值"""
        assert PieceUtils.get_piece_value(PieceType.RED_KING) == 10000
        assert PieceUtils.get_piece_value(PieceType.BLACK_KING) == 10000
        
        assert PieceUtils.get_piece_value(PieceType.RED_CHARIOT) == 900
        assert PieceUtils.get_piece_value(PieceType.BLACK_CHARIOT) == 900
        
        assert PieceUtils.get_piece_value(PieceType.RED_PAWN) == 100
        assert PieceUtils.get_piece_value(PieceType.BLACK_PAWN) == 100


if __name__ == "__main__":
    pytest.main([__file__])
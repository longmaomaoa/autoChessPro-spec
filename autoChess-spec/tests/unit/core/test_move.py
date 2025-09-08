#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
走法模型单元测试
"""

import pytest
from chess_ai.core.piece import Piece, Position, PieceType
from chess_ai.core.move import Move, MoveType, MoveHistory, MoveValidator


class TestMove:
    """走法类测试"""
    
    def test_move_creation(self):
        """测试走法创建"""
        from_pos = Position(0, 4)
        to_pos = Position(1, 4)
        piece = Piece(PieceType.RED_KING, from_pos)
        
        move = Move(from_pos, to_pos, piece)
        
        assert move.from_position == from_pos
        assert move.to_position == to_pos
        assert move.piece == piece
        assert move.move_type == MoveType.NORMAL
        assert move.captured_piece is None
    
    def test_capture_move(self):
        """测试吃子走法"""
        from_pos = Position(0, 4)
        to_pos = Position(1, 4)
        piece = Piece(PieceType.RED_KING, from_pos)
        captured_piece = Piece(PieceType.BLACK_PAWN, to_pos)
        
        move = Move(from_pos, to_pos, piece, captured_piece)
        
        assert move.is_capture
        assert move.move_type == MoveType.CAPTURE
        assert move.captured_piece == captured_piece
    
    def test_move_distance(self):
        """测试移动距离"""
        from_pos = Position(0, 0)
        to_pos = Position(2, 2)
        piece = Piece(PieceType.RED_HORSE, from_pos)
        
        move = Move(from_pos, to_pos, piece)
        
        assert move.distance == 4  # |2-0| + |2-0| = 4
    
    def test_move_to_chinese_notation(self):
        """测试转换为中文记法"""
        from_pos = Position(0, 4)
        to_pos = Position(1, 4)
        piece = Piece(PieceType.RED_KING, from_pos)
        
        move = Move(from_pos, to_pos, piece)
        notation = move.to_chinese_notation()
        
        assert "帅" in notation
        assert "五1" in notation
        assert "五2" in notation
    
    def test_move_to_ucci_notation(self):
        """测试转换为UCCI记法"""
        from_pos = Position(0, 4)
        to_pos = Position(1, 4)
        piece = Piece(PieceType.RED_KING, from_pos)
        
        move = Move(from_pos, to_pos, piece)
        ucci = move.to_ucci_notation()
        
        assert ucci == "4041"
    
    def test_move_from_ucci_notation(self):
        """测试从UCCI记法创建走法"""
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        move = Move.from_ucci_notation("4041", piece)
        
        assert move.from_position == Position(0, 4)
        assert move.to_position == Position(1, 4)
        assert move.piece == piece
        
        with pytest.raises(ValueError):
            Move.from_ucci_notation("404", piece)  # 长度错误
        
        with pytest.raises(ValueError):
            Move.from_ucci_notation("abcd", piece)  # 非数字
    
    def test_move_from_algebraic_notation(self):
        """测试从代数记法创建走法"""
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        
        # 普通移动
        move = Move.from_algebraic_notation("e1-e2", piece)
        assert move.from_position == Position(0, 4)
        assert move.to_position == Position(1, 4)
        assert move.move_type == MoveType.NORMAL
        
        # 吃子移动
        move = Move.from_algebraic_notation("e1xe2", piece)
        assert move.move_type == MoveType.CAPTURE
        
        with pytest.raises(ValueError):
            Move.from_algebraic_notation("invalid", piece)
    
    def test_piece_movement_validation(self):
        """测试棋子移动规则验证"""
        # 帅的移动
        king = Piece(PieceType.RED_KING, Position(0, 4))
        valid_move = Move(Position(0, 4), Position(0, 3), king)  # 左移一格
        assert valid_move.is_valid_for_piece_type()
        
        invalid_move = Move(Position(0, 4), Position(2, 4), king)  # 移动两格
        assert not invalid_move.is_valid_for_piece_type()
        
        # 马的移动
        horse = Piece(PieceType.RED_HORSE, Position(0, 1))
        valid_move = Move(Position(0, 1), Position(2, 0), horse)  # 日字形
        assert valid_move.is_valid_for_piece_type()
        
        invalid_move = Move(Position(0, 1), Position(1, 1), horse)  # 非日字形
        assert not invalid_move.is_valid_for_piece_type()
    
    def test_move_reverse(self):
        """测试反向走法"""
        from_pos = Position(0, 4)
        to_pos = Position(1, 4)
        piece = Piece(PieceType.RED_KING, from_pos)
        
        move = Move(from_pos, to_pos, piece, evaluation_score=100.0)
        reverse_move = move.reverse()
        
        assert reverse_move.from_position == to_pos
        assert reverse_move.to_position == from_pos
        assert reverse_move.evaluation_score == -100.0
        assert reverse_move.captured_piece is None


class TestMoveHistory:
    """走法历史测试"""
    
    def test_move_history_creation(self):
        """测试走法历史创建"""
        history = MoveHistory([])
        
        assert len(history) == 0
        assert history.current_index == -1
    
    def test_add_move(self):
        """测试添加走法"""
        history = MoveHistory([])
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        move = Move(Position(0, 4), Position(1, 4), piece)
        
        history.add_move(move)
        
        assert len(history) == 1
        assert history.current_index == 0
        assert history.get_current_move() == move
    
    def test_undo_redo_move(self):
        """测试悔棋和重做"""
        history = MoveHistory([])
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        move1 = Move(Position(0, 4), Position(1, 4), piece)
        move2 = Move(Position(1, 4), Position(2, 4), piece)
        
        history.add_move(move1)
        history.add_move(move2)
        
        # 悔棋
        undone_move = history.undo_move()
        assert undone_move == move2
        assert history.current_index == 0
        
        # 重做
        redone_move = history.redo_move()
        assert redone_move == move2
        assert history.current_index == 1
        
        # 继续悔棋
        history.undo_move()
        history.undo_move()
        assert history.current_index == -1
        assert history.undo_move() is None
    
    def test_history_branching(self):
        """测试历史分支"""
        history = MoveHistory([])
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        move1 = Move(Position(0, 4), Position(1, 4), piece)
        move2 = Move(Position(1, 4), Position(2, 4), piece)
        move3 = Move(Position(1, 4), Position(1, 3), piece)
        
        history.add_move(move1)
        history.add_move(move2)
        
        # 悔棋后添加新走法
        history.undo_move()
        history.add_move(move3)
        
        # 应该只有move1和move3
        assert len(history.moves) == 2
        assert history.moves[0] == move1
        assert history.moves[1] == move3
    
    def test_to_ucci_list(self):
        """测试转换为UCCI列表"""
        history = MoveHistory([])
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        move1 = Move(Position(0, 4), Position(1, 4), piece)
        move2 = Move(Position(1, 4), Position(2, 4), piece)
        
        history.add_move(move1)
        history.add_move(move2)
        
        ucci_list = history.to_ucci_list()
        assert ucci_list == ["4041", "4142"]


class TestMoveValidator:
    """走法验证器测试"""
    
    def test_position_validation(self):
        """测试位置验证"""
        assert MoveValidator.is_position_valid(Position(0, 0))
        assert MoveValidator.is_position_valid(Position(9, 8))
        assert not MoveValidator.is_position_valid(Position(-1, 0))
        assert not MoveValidator.is_position_valid(Position(10, 0))
    
    def test_move_format_validation(self):
        """测试走法格式验证"""
        assert MoveValidator.is_move_format_valid("4041")
        assert MoveValidator.is_move_format_valid("a1-b2")
        assert MoveValidator.is_move_format_valid("a1xb2")
        assert not MoveValidator.is_move_format_valid("abc")
        assert not MoveValidator.is_move_format_valid("4-41")
    
    def test_basic_move_validation(self):
        """测试基础走法验证"""
        piece = Piece(PieceType.RED_KING, Position(0, 4))
        
        # 有效走法
        valid_move = Move(Position(0, 4), Position(0, 3), piece)
        errors = MoveValidator.validate_move_basic(valid_move)
        assert len(errors) == 0
        
        # 无效走法 - 原地不动
        invalid_move = Move(Position(0, 4), Position(0, 4), piece)
        errors = MoveValidator.validate_move_basic(invalid_move)
        assert len(errors) > 0
        assert "起始位置和目标位置相同" in errors
        
        # 无效走法 - 不符合移动规则
        invalid_move2 = Move(Position(0, 4), Position(2, 4), piece)
        errors = MoveValidator.validate_move_basic(invalid_move2)
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])
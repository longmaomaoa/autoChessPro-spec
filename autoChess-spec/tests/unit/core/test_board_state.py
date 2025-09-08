#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋局状态模型单元测试
"""

import pytest
from chess_ai.core.piece import Piece, Position, PieceType, PieceColor
from chess_ai.core.move import Move, MoveType, MoveHistory
from chess_ai.core.board_state import BoardState, GameState


class TestBoardState:
    """棋局状态类测试"""
    
    def test_board_state_creation(self):
        """测试棋局状态创建"""
        board = BoardState()
        
        # 检查初始状态
        assert board.current_player == PieceColor.RED
        assert board.game_state == GameState.PLAYING
        assert board.move_count == 0
        assert not board.is_game_over
        
        # 检查是否有棋子
        assert len(board.pieces) == 32  # 初始32个棋子
    
    def test_get_piece_at(self):
        """测试获取指定位置的棋子"""
        board = BoardState()
        
        # 红帅位置
        king_pos = Position(0, 4)
        piece = board.get_piece_at(king_pos)
        assert piece is not None
        assert piece.piece_type == PieceType.RED_KING
        
        # 空位置
        empty_pos = Position(4, 4)
        piece = board.get_piece_at(empty_pos)
        assert piece is None
    
    def test_set_piece_at(self):
        """测试设置指定位置的棋子"""
        board = BoardState()
        pos = Position(4, 4)
        piece = Piece(PieceType.RED_PAWN, pos)
        
        board.set_piece_at(pos, piece)
        retrieved_piece = board.get_piece_at(pos)
        
        assert retrieved_piece is not None
        assert retrieved_piece.piece_type == PieceType.RED_PAWN
        assert retrieved_piece.position == pos
    
    def test_remove_piece_at(self):
        """测试移除指定位置的棋子"""
        board = BoardState()
        king_pos = Position(0, 4)
        
        # 确认棋子存在
        assert board.get_piece_at(king_pos) is not None
        
        # 移除棋子
        removed_piece = board.remove_piece_at(king_pos)
        assert removed_piece is not None
        assert removed_piece.piece_type == PieceType.RED_KING
        
        # 确认棋子已移除
        assert board.get_piece_at(king_pos) is None
    
    def test_get_pieces_by_color(self):
        """测试获取指定颜色的棋子"""
        board = BoardState()
        
        red_pieces = board.get_pieces_by_color(PieceColor.RED)
        black_pieces = board.get_pieces_by_color(PieceColor.BLACK)
        
        assert len(red_pieces) == 16
        assert len(black_pieces) == 16
        
        # 验证颜色
        for piece in red_pieces:
            assert piece.color == PieceColor.RED
        
        for piece in black_pieces:
            assert piece.color == PieceColor.BLACK
    
    def test_get_king_position(self):
        """测试获取帅/将位置"""
        board = BoardState()
        
        red_king_pos = board.get_king_position(PieceColor.RED)
        assert red_king_pos == Position(0, 4)
        
        black_king_pos = board.get_king_position(PieceColor.BLACK)
        assert black_king_pos == Position(9, 4)
    
    def test_simple_move_execution(self):
        """测试简单走法执行"""
        board = BoardState()
        
        # 红兵前进
        from_pos = Position(3, 0)
        to_pos = Position(4, 0)
        piece = board.get_piece_at(from_pos)
        
        move = Move(from_pos, to_pos, piece)
        success = board.make_move(move)
        
        assert success
        assert board.get_piece_at(from_pos) is None
        assert board.get_piece_at(to_pos) is not None
        assert board.current_player == PieceColor.BLACK
        assert board.move_count == 1
    
    def test_capture_move_execution(self):
        """测试吃子走法执行"""
        board = BoardState()
        
        # 手动设置一个吃子场景
        red_piece = Piece(PieceType.RED_CHARIOT, Position(4, 4))
        black_piece = Piece(PieceType.BLACK_PAWN, Position(4, 5))
        
        board.set_piece_at(Position(4, 4), red_piece)
        board.set_piece_at(Position(4, 5), black_piece)
        
        move = Move(Position(4, 4), Position(4, 5), red_piece)
        success = board.make_move(move)
        
        assert success
        assert board.get_piece_at(Position(4, 4)) is None
        assert board.get_piece_at(Position(4, 5)).piece_type == PieceType.RED_CHARIOT
        assert move.captured_piece is not None
        assert move.is_capture
    
    def test_invalid_move_rejection(self):
        """测试无效走法拒绝"""
        board = BoardState()
        
        # 尝试移动空位置的棋子
        empty_pos = Position(4, 4)
        to_pos = Position(5, 4)
        fake_piece = Piece(PieceType.RED_PAWN, empty_pos)
        
        move = Move(empty_pos, to_pos, fake_piece)
        success = board.make_move(move)
        
        assert not success
        assert board.current_player == PieceColor.RED  # 玩家未改变
        assert board.move_count == 0  # 移动计数未改变
    
    def test_undo_move(self):
        """测试悔棋"""
        board = BoardState()
        
        # 执行一个走法
        from_pos = Position(3, 0)
        to_pos = Position(4, 0)
        piece = board.get_piece_at(from_pos)
        original_piece = piece
        
        move = Move(from_pos, to_pos, piece)
        board.make_move(move)
        
        # 悔棋
        success = board.undo_move()
        
        assert success
        assert board.get_piece_at(from_pos) is not None
        assert board.get_piece_at(to_pos) is None
        assert board.current_player == PieceColor.RED
        assert board.move_count == 0
    
    def test_to_fen_conversion(self):
        """测试转换为FEN记录"""
        board = BoardState()
        fen = board.to_fen()
        
        assert isinstance(fen, str)
        parts = fen.split()
        assert len(parts) >= 2
        assert parts[1] == "r"  # 红方先行
    
    def test_from_fen_conversion(self):
        """测试从FEN记录创建棋局"""
        # 使用初始棋局FEN
        original_board = BoardState()
        original_fen = original_board.to_fen()
        
        # 从FEN重建
        rebuilt_board = BoardState.from_fen(original_fen)
        
        assert rebuilt_board.current_player == original_board.current_player
        assert len(rebuilt_board.pieces) == len(original_board.pieces)
        
        # 检查关键位置的棋子
        assert rebuilt_board.get_piece_at(Position(0, 4)).piece_type == PieceType.RED_KING
        assert rebuilt_board.get_piece_at(Position(9, 4)).piece_type == PieceType.BLACK_KING
    
    def test_board_copy(self):
        """测试棋局状态拷贝"""
        original = BoardState()
        
        # 执行一些操作
        piece = original.get_piece_at(Position(3, 0))
        move = Move(Position(3, 0), Position(4, 0), piece)
        original.make_move(move)
        
        # 创建拷贝
        copy_board = original.copy()
        
        # 修改原棋局
        piece2 = original.get_piece_at(Position(3, 2))
        move2 = Move(Position(3, 2), Position(4, 2), piece2)
        original.make_move(move2)
        
        # 拷贝应该不受影响
        assert original.move_count == 2
        assert copy_board.move_count == 1
        assert copy_board.get_piece_at(Position(4, 2)) is None
    
    def test_get_legal_moves(self):
        """测试获取合法走法"""
        board = BoardState()
        
        legal_moves = board.get_legal_moves(PieceColor.RED)
        assert len(legal_moves) > 0
        
        # 所有走法都应该是红方的
        for move in legal_moves:
            assert move.piece.color == PieceColor.RED
        
        # 初始状态下红方应该有兵和马的走法
        pawn_moves = [m for m in legal_moves if m.piece.piece_type == PieceType.RED_PAWN]
        horse_moves = [m for m in legal_moves if m.piece.piece_type == PieceType.RED_HORSE]
        
        assert len(pawn_moves) > 0  # 兵可以前进
        assert len(horse_moves) > 0  # 马可以移动
    
    def test_invalid_fen_handling(self):
        """测试无效FEN处理"""
        with pytest.raises(ValueError):
            BoardState.from_fen("invalid fen string")
        
        with pytest.raises(ValueError):
            BoardState.from_fen("incomplete")


class TestGameStateUpdates:
    """游戏状态更新测试"""
    
    def test_initial_game_state(self):
        """测试初始游戏状态"""
        board = BoardState()
        
        assert board.game_state == GameState.PLAYING
        assert not board.is_game_over
        assert board.winner is None
    
    def test_game_state_after_moves(self):
        """测试移动后的游戏状态"""
        board = BoardState()
        
        # 执行几个正常走法
        moves_to_make = [
            (Position(3, 0), Position(4, 0)),  # 红兵前进
            (Position(6, 0), Position(5, 0)),  # 黑卒前进
        ]
        
        for from_pos, to_pos in moves_to_make:
            piece = board.get_piece_at(from_pos)
            if piece:
                move = Move(from_pos, to_pos, piece)
                board.make_move(move)
        
        # 游戏应该仍在进行中
        assert board.game_state == GameState.PLAYING
        assert not board.is_game_over


if __name__ == "__main__":
    pytest.main([__file__])
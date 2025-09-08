#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋局状态数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set, Tuple, Iterator
from copy import deepcopy
from enum import Enum
import re

from chess_ai.core.piece import Piece, Position, PieceType, PieceColor, PieceUtils
from chess_ai.core.move import Move, MoveHistory, MoveType


class GameState(Enum):
    """游戏状态"""
    PLAYING = "进行中"
    RED_WIN = "红方胜利"
    BLACK_WIN = "黑方胜利"
    DRAW = "平局"
    STALEMATE = "无子可动"


@dataclass
class BoardState:
    """棋局状态类"""
    pieces: Dict[Position, Piece] = field(default_factory=dict)
    current_player: PieceColor = PieceColor.RED
    move_history: MoveHistory = field(default_factory=MoveHistory)
    game_state: GameState = GameState.PLAYING
    move_count: int = 0
    fifty_move_rule: int = 0  # 五十回合规则计数
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.pieces:
            self.pieces = PieceUtils.get_initial_pieces()
    
    @property
    def is_game_over(self) -> bool:
        """游戏是否结束"""
        return self.game_state != GameState.PLAYING
    
    @property
    def winner(self) -> Optional[PieceColor]:
        """获取胜者"""
        if self.game_state == GameState.RED_WIN:
            return PieceColor.RED
        elif self.game_state == GameState.BLACK_WIN:
            return PieceColor.BLACK
        return None
    
    def get_piece_at(self, position: Position) -> Optional[Piece]:
        """获取指定位置的棋子"""
        return self.pieces.get(position)
    
    def set_piece_at(self, position: Position, piece: Optional[Piece]) -> None:
        """设置指定位置的棋子"""
        if piece is None:
            self.pieces.pop(position, None)
        else:
            # 更新棋子位置
            updated_piece = piece.move_to(position)
            self.pieces[position] = updated_piece
    
    def remove_piece_at(self, position: Position) -> Optional[Piece]:
        """移除指定位置的棋子"""
        return self.pieces.pop(position, None)
    
    def get_pieces_by_color(self, color: PieceColor) -> List[Piece]:
        """获取指定颜色的所有棋子"""
        return [piece for piece in self.pieces.values() if piece.color == color]
    
    def get_king_position(self, color: PieceColor) -> Optional[Position]:
        """获取指定颜色的帅/将位置"""
        king_type = PieceType.RED_KING if color == PieceColor.RED else PieceType.BLACK_KING
        for piece in self.pieces.values():
            if piece.piece_type == king_type:
                return piece.position
        return None
    
    def is_position_under_attack(self, position: Position, by_color: PieceColor) -> bool:
        """判断位置是否被指定颜色攻击"""
        attacking_pieces = self.get_pieces_by_color(by_color)
        
        for piece in attacking_pieces:
            # 创建一个假想的走法来检查是否能攻击到目标位置
            test_move = Move(
                from_position=piece.position,
                to_position=position,
                piece=piece
            )
            
            # 检查基本移动规则
            if not test_move.is_valid_for_piece_type():
                continue
            
            # 检查路径是否被阻挡
            if self._is_path_clear(piece.position, position, piece.piece_type):
                return True
        
        return False
    
    def is_in_check(self, color: PieceColor) -> bool:
        """判断指定颜色是否被将军"""
        king_position = self.get_king_position(color)
        if king_position is None:
            return False
        
        opponent_color = PieceColor.BLACK if color == PieceColor.RED else PieceColor.RED
        return self.is_position_under_attack(king_position, opponent_color)
    
    def _is_path_clear(self, from_pos: Position, to_pos: Position, piece_type: PieceType) -> bool:
        """检查路径是否畅通"""
        from_row, from_col = from_pos.row, from_pos.col
        to_row, to_col = to_pos.row, to_pos.col
        
        # 马的移动需要检查蹩马腿
        if piece_type in [PieceType.RED_HORSE, PieceType.BLACK_HORSE]:
            row_diff = to_row - from_row
            col_diff = to_col - from_col
            
            if abs(row_diff) == 2 and abs(col_diff) == 1:
                # 检查竖向蹩马腿
                block_pos = Position(from_row + row_diff // 2, from_col)
                return self.get_piece_at(block_pos) is None
            elif abs(row_diff) == 1 and abs(col_diff) == 2:
                # 检查横向蹩马腿
                block_pos = Position(from_row, from_col + col_diff // 2)
                return self.get_piece_at(block_pos) is None
        
        # 象的移动需要检查塞象眼
        elif piece_type in [PieceType.RED_ELEPHANT, PieceType.BLACK_ELEPHANT]:
            if abs(to_row - from_row) == 2 and abs(to_col - from_col) == 2:
                # 检查象眼位置
                eye_row = (from_row + to_row) // 2
                eye_col = (from_col + to_col) // 2
                eye_pos = Position(eye_row, eye_col)
                return self.get_piece_at(eye_pos) is None
        
        # 车和炮的直线移动
        elif piece_type in [PieceType.RED_CHARIOT, PieceType.BLACK_CHARIOT,
                           PieceType.RED_CANNON, PieceType.BLACK_CANNON]:
            if from_row == to_row:  # 水平移动
                start_col = min(from_col, to_col) + 1
                end_col = max(from_col, to_col)
                pieces_in_path = sum(1 for col in range(start_col, end_col)
                                   if self.get_piece_at(Position(from_row, col)) is not None)
            elif from_col == to_col:  # 垂直移动
                start_row = min(from_row, to_row) + 1
                end_row = max(from_row, to_row)
                pieces_in_path = sum(1 for row in range(start_row, end_row)
                                   if self.get_piece_at(Position(row, from_col)) is not None)
            else:
                return False
            
            # 车需要路径完全畅通，炮吃子时需要恰好一个翻山
            if piece_type in [PieceType.RED_CHARIOT, PieceType.BLACK_CHARIOT]:
                return pieces_in_path == 0
            else:  # 炮
                target_piece = self.get_piece_at(to_pos)
                if target_piece is None:  # 移动
                    return pieces_in_path == 0
                else:  # 吃子
                    return pieces_in_path == 1
        
        return True
    
    def make_move(self, move: Move) -> bool:
        """执行走法"""
        # 验证走法有效性
        if not self.is_move_valid(move):
            return False
        
        # 执行走法
        from_piece = self.get_piece_at(move.from_position)
        captured_piece = self.get_piece_at(move.to_position)
        
        # 移除原位置的棋子
        self.remove_piece_at(move.from_position)
        
        # 在新位置放置棋子
        new_piece = from_piece.move_to(move.to_position)
        self.set_piece_at(move.to_position, new_piece)
        
        # 更新走法信息
        move.captured_piece = captured_piece
        if captured_piece:
            move.move_type = MoveType.CAPTURE
        
        # 添加到历史记录
        self.move_history.add_move(move)
        
        # 切换当前玩家
        self.current_player = (PieceColor.BLACK if self.current_player == PieceColor.RED 
                              else PieceColor.RED)
        
        # 更新移动计数
        self.move_count += 1
        
        # 更新五十回合规则
        if captured_piece or move.piece.piece_type in [PieceType.RED_PAWN, PieceType.BLACK_PAWN]:
            self.fifty_move_rule = 0
        else:
            self.fifty_move_rule += 1
        
        # 检查游戏状态
        self._update_game_state()
        
        return True
    
    def is_move_valid(self, move: Move) -> bool:
        """验证走法是否有效"""
        # 检查起始位置是否有己方棋子
        from_piece = self.get_piece_at(move.from_position)
        if from_piece is None or from_piece.color != self.current_player:
            return False
        
        # 检查目标位置是否有己方棋子
        to_piece = self.get_piece_at(move.to_position)
        if to_piece is not None and to_piece.color == self.current_player:
            return False
        
        # 检查走法是否符合棋子类型
        if not move.is_valid_for_piece_type():
            return False
        
        # 检查路径是否畅通
        if not self._is_path_clear(move.from_position, move.to_position, from_piece.piece_type):
            return False
        
        # 检查走法后是否会导致自己被将军
        temp_board = self.copy()
        temp_move = Move(move.from_position, move.to_position, from_piece, to_piece)
        temp_board._execute_move_without_validation(temp_move)
        if temp_board.is_in_check(self.current_player):
            return False
        
        return True
    
    def _execute_move_without_validation(self, move: Move) -> None:
        """执行走法但不进行验证（用于临时模拟）"""
        self.remove_piece_at(move.from_position)
        self.set_piece_at(move.to_position, move.piece.move_to(move.to_position))
    
    def undo_move(self) -> bool:
        """悔棋"""
        move = self.move_history.undo_move()
        if move is None:
            return False
        
        # 恢复棋子位置
        piece = self.remove_piece_at(move.to_position)
        if piece:
            self.set_piece_at(move.from_position, piece.move_to(move.from_position))
        
        # 恢复被吃的棋子
        if move.captured_piece:
            self.set_piece_at(move.to_position, move.captured_piece)
        
        # 切换当前玩家
        self.current_player = (PieceColor.BLACK if self.current_player == PieceColor.RED 
                              else PieceColor.RED)
        
        # 更新计数（简化处理）
        self.move_count -= 1
        
        # 重新检查游戏状态
        self.game_state = GameState.PLAYING
        self._update_game_state()
        
        return True
    
    def get_legal_moves(self, color: Optional[PieceColor] = None) -> List[Move]:
        """获取指定颜色的所有合法走法"""
        if color is None:
            color = self.current_player
        
        legal_moves = []
        pieces = self.get_pieces_by_color(color)
        
        for piece in pieces:
            # 生成该棋子的所有可能走法
            possible_moves = self._generate_possible_moves(piece)
            
            # 验证每个走法
            for move in possible_moves:
                if self.is_move_valid(move):
                    legal_moves.append(move)
        
        return legal_moves
    
    def _generate_possible_moves(self, piece: Piece) -> List[Move]:
        """生成单个棋子的所有可能走法"""
        moves = []
        piece_type = piece.piece_type
        from_pos = piece.position
        
        # 根据棋子类型生成走法
        if piece_type in [PieceType.RED_KING, PieceType.BLACK_KING]:
            # 帅/将：九宫格内单步移动
            palace_positions = self._get_palace_positions(piece.color)
            for pos in palace_positions:
                if abs(pos.row - from_pos.row) + abs(pos.col - from_pos.col) == 1:
                    moves.append(Move(from_pos, pos, piece))
        
        elif piece_type in [PieceType.RED_ADVISOR, PieceType.BLACK_ADVISOR]:
            # 仕/士：九宫格内斜向移动
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            palace_positions = self._get_palace_positions(piece.color)
            for dr, dc in directions:
                new_pos = Position(from_pos.row + dr, from_pos.col + dc)
                if new_pos in palace_positions:
                    moves.append(Move(from_pos, new_pos, piece))
        
        elif piece_type in [PieceType.RED_ELEPHANT, PieceType.BLACK_ELEPHANT]:
            # 相/象：斜向两格移动，不能过河
            directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
            for dr, dc in directions:
                new_row, new_col = from_pos.row + dr, from_pos.col + dc
                if (0 <= new_row <= 9 and 0 <= new_col <= 8):
                    # 检查是否过河
                    if piece.color == PieceColor.RED and new_row <= 4:
                        moves.append(Move(from_pos, Position(new_row, new_col), piece))
                    elif piece.color == PieceColor.BLACK and new_row >= 5:
                        moves.append(Move(from_pos, Position(new_row, new_col), piece))
        
        elif piece_type in [PieceType.RED_HORSE, PieceType.BLACK_HORSE]:
            # 马：日字形移动
            horse_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                          (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in horse_moves:
                new_row, new_col = from_pos.row + dr, from_pos.col + dc
                if (0 <= new_row <= 9 and 0 <= new_col <= 8):
                    moves.append(Move(from_pos, Position(new_row, new_col), piece))
        
        elif piece_type in [PieceType.RED_CHARIOT, PieceType.BLACK_CHARIOT]:
            # 车：直线移动
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                for i in range(1, 10):
                    new_row, new_col = from_pos.row + dr * i, from_pos.col + dc * i
                    if not (0 <= new_row <= 9 and 0 <= new_col <= 8):
                        break
                    moves.append(Move(from_pos, Position(new_row, new_col), piece))
        
        elif piece_type in [PieceType.RED_CANNON, PieceType.BLACK_CANNON]:
            # 炮：直线移动和吃子
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                for i in range(1, 10):
                    new_row, new_col = from_pos.row + dr * i, from_pos.col + dc * i
                    if not (0 <= new_row <= 9 and 0 <= new_col <= 8):
                        break
                    moves.append(Move(from_pos, Position(new_row, new_col), piece))
        
        elif piece_type in [PieceType.RED_PAWN, PieceType.BLACK_PAWN]:
            # 兵/卒：根据是否过河决定移动方向
            if piece.color == PieceColor.RED:
                # 红兵向上移动
                moves.append(Move(from_pos, Position(from_pos.row + 1, from_pos.col), piece))
                if from_pos.row >= 5:  # 已过河，可以横移
                    if from_pos.col > 0:
                        moves.append(Move(from_pos, Position(from_pos.row, from_pos.col - 1), piece))
                    if from_pos.col < 8:
                        moves.append(Move(from_pos, Position(from_pos.row, from_pos.col + 1), piece))
            else:
                # 黑卒向下移动
                moves.append(Move(from_pos, Position(from_pos.row - 1, from_pos.col), piece))
                if from_pos.row <= 4:  # 已过河，可以横移
                    if from_pos.col > 0:
                        moves.append(Move(from_pos, Position(from_pos.row, from_pos.col - 1), piece))
                    if from_pos.col < 8:
                        moves.append(Move(from_pos, Position(from_pos.row, from_pos.col + 1), piece))
        
        # 过滤无效位置
        valid_moves = []
        for move in moves:
            if (0 <= move.to_position.row <= 9 and 0 <= move.to_position.col <= 8):
                valid_moves.append(move)
        
        return valid_moves
    
    def _get_palace_positions(self, color: PieceColor) -> List[Position]:
        """获取九宫格位置"""
        if color == PieceColor.RED:
            rows = [0, 1, 2]
        else:
            rows = [7, 8, 9]
        
        cols = [3, 4, 5]
        positions = []
        for row in rows:
            for col in cols:
                positions.append(Position(row, col))
        
        return positions
    
    def _update_game_state(self) -> None:
        """更新游戏状态"""
        # 检查是否将死或困毙
        current_legal_moves = self.get_legal_moves(self.current_player)
        
        if not current_legal_moves:
            if self.is_in_check(self.current_player):
                # 将死
                if self.current_player == PieceColor.RED:
                    self.game_state = GameState.BLACK_WIN
                else:
                    self.game_state = GameState.RED_WIN
            else:
                # 困毙
                self.game_state = GameState.STALEMATE
            return
        
        # 检查五十回合规则
        if self.fifty_move_rule >= 100:  # 双方各50回合
            self.game_state = GameState.DRAW
            return
        
        # 检查长将规则等其他平局条件
        # (这里可以添加更复杂的平局判定逻辑)
        
        self.game_state = GameState.PLAYING
    
    def copy(self) -> "BoardState":
        """创建棋局状态的深拷贝"""
        return deepcopy(self)
    
    def to_fen(self) -> str:
        """转换为FEN记录"""
        # FEN格式: 棋盘状态 当前玩家 易位 吃过路兵 半回合计数 全回合计数
        
        # 棋盘状态
        board_str = ""
        for row in range(9, -1, -1):  # 从第10行到第1行
            empty_count = 0
            row_str = ""
            
            for col in range(9):  # 从a列到i列
                piece = self.get_piece_at(Position(row, col))
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece.fen_symbol
            
            if empty_count > 0:
                row_str += str(empty_count)
            
            board_str += row_str
            if row > 0:
                board_str += "/"
        
        # 当前玩家
        current_player_str = "r" if self.current_player == PieceColor.RED else "b"
        
        # 其他信息（象棋中简化处理）
        castling = "-"  # 象棋无易位
        en_passant = "-"  # 象棋无吃过路兵
        halfmove = str(self.fifty_move_rule)
        fullmove = str((self.move_count // 2) + 1)
        
        return f"{board_str} {current_player_str} {castling} {en_passant} {halfmove} {fullmove}"
    
    @classmethod
    def from_fen(cls, fen_str: str) -> "BoardState":
        """从FEN记录创建棋局状态"""
        parts = fen_str.strip().split()
        if len(parts) < 2:
            raise ValueError(f"FEN格式错误: {fen_str}")
        
        board_str = parts[0]
        current_player_str = parts[1]
        
        # 解析棋盘状态
        pieces = {}
        rows = board_str.split("/")
        
        for row_idx, row_str in enumerate(rows):
            row = 9 - row_idx  # FEN从第10行开始
            col = 0
            
            for char in row_str:
                if char.isdigit():
                    col += int(char)  # 跳过空格
                else:
                    position = Position(row, col)
                    piece = Piece.from_fen_symbol(char, position)
                    pieces[position] = piece
                    col += 1
        
        # 解析当前玩家
        current_player = PieceColor.RED if current_player_str == "r" else PieceColor.BLACK
        
        # 创建棋局状态
        board_state = cls(
            pieces=pieces,
            current_player=current_player,
            move_history=MoveHistory([])
        )
        
        # 解析其他信息
        if len(parts) >= 5:
            board_state.fifty_move_rule = int(parts[4])
        if len(parts) >= 6:
            fullmove = int(parts[5])
            board_state.move_count = (fullmove - 1) * 2
            if current_player == PieceColor.BLACK:
                board_state.move_count += 1
        
        return board_state
    
    def __str__(self) -> str:
        """字符串表示"""
        board_str = ""
        for row in range(9, -1, -1):
            board_str += f"{row + 1:2} "
            for col in range(9):
                piece = self.get_piece_at(Position(row, col))
                if piece:
                    board_str += piece.unicode_symbol + " "
                else:
                    board_str += "· "
            board_str += "\n"
        
        board_str += "   " + " ".join("abcdefghi") + "\n"
        board_str += f"当前玩家: {self.current_player.value}\n"
        board_str += f"游戏状态: {self.game_state.value}\n"
        board_str += f"移动次数: {self.move_count}\n"
        
        return board_str
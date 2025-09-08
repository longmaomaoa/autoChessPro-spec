#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
象棋胜率分析和局面评估系统

提供基于多维度算法的智能局面评估和胜率预测功能
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from chess_ai.core.board_state import BoardState, PieceType, PieceColor
from chess_ai.core.config_manager import ConfigManager
from chess_ai.ai_engine.pikafish_engine import PikafishEngine, EngineAnalysis


class EvaluationMethod(Enum):
    """评估方法"""
    MATERIAL_ONLY = "material_only"       # 纯子力评估
    POSITIONAL = "positional"             # 位置评估
    TACTICAL = "tactical"                 # 战术评估
    STRATEGIC = "strategic"               # 战略评估
    COMPREHENSIVE = "comprehensive"        # 综合评估
    ENGINE_ASSISTED = "engine_assisted"   # 引擎辅助


class GamePhase(Enum):
    """对局阶段"""
    OPENING = "opening"       # 开局
    MIDDLE_GAME = "middle"    # 中局
    END_GAME = "endgame"     # 残局
    COMPLEX = "complex"      # 复杂局面


@dataclass
class PieceValues:
    """棋子价值表"""
    # 基础子力价值 (分)
    SHUAI = 1000    # 帅/将
    CHE = 100       # 车
    MA = 45         # 马  
    PAO = 45        # 炮
    XIANG = 20      # 相/象
    SHI = 20        # 士
    BING = 10       # 兵/卒
    
    # 位置奖励系数
    center_bonus = 1.2      # 中心位置奖励
    edge_penalty = 0.8      # 边缘位置惩罚
    development_bonus = 1.1  # 发展奖励
    safety_bonus = 1.15     # 安全奖励


@dataclass
class PositionalFactors:
    """位置因素"""
    king_safety: float = 0.0          # 王的安全 (-1到1)
    center_control: float = 0.0       # 中心控制 (-1到1)
    piece_activity: float = 0.0       # 子力活跃度 (-1到1)
    pawn_structure: float = 0.0       # 兵型结构 (-1到1)
    piece_coordination: float = 0.0    # 子力协调 (-1到1)
    space_advantage: float = 0.0      # 空间优势 (-1到1)


@dataclass
class TacticalElements:
    """战术要素"""
    pins: int = 0              # 牵制
    forks: int = 0             # 双击
    skewers: int = 0           # 串击
    discovered_attacks: int = 0 # 闪击
    sacrificial_threats: int = 0 # 弃子威胁
    back_rank_threats: int = 0  # 底线威胁
    checkmate_threats: int = 0  # 将杀威胁


@dataclass
class WinProbability:
    """胜率分析"""
    white_win_rate: float      # 红方胜率 (0-1)
    black_win_rate: float      # 黑方胜率 (0-1) 
    draw_rate: float           # 和棋率 (0-1)
    confidence: float          # 置信度 (0-1)
    evaluation_method: EvaluationMethod
    
    # 详细分析
    material_advantage: float = 0.0    # 子力优势
    positional_advantage: float = 0.0  # 位置优势
    tactical_advantage: float = 0.0    # 战术优势
    time_advantage: float = 0.0        # 时间优势


@dataclass
class PositionEvaluation:
    """局面评估结果"""
    overall_score: float               # 总体评分 (-100到100)
    game_phase: GamePhase             # 对局阶段
    win_probability: WinProbability   # 胜率分析
    
    # 详细因素
    material_balance: float = 0.0      # 子力平衡
    positional_factors: PositionalFactors = field(default_factory=PositionalFactors)
    tactical_elements: TacticalElements = field(default_factory=TacticalElements)
    
    # 建议信息
    key_features: List[str] = field(default_factory=list)     # 关键特征
    strengths: List[str] = field(default_factory=list)        # 优势
    weaknesses: List[str] = field(default_factory=list)       # 弱点
    critical_moves: List[str] = field(default_factory=list)   # 关键走法
    
    # 元信息
    evaluation_time: float = 0.0       # 评估耗时
    engine_depth: Optional[int] = None # 引擎分析深度


class ComprehensiveEvaluator:
    """综合局面评估器
    
    提供多维度的象棋局面评估和胜率分析功能
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.piece_values = PieceValues()
        self.engine: Optional[PikafishEngine] = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Evaluator")
        
        # 评估配置
        self.evaluation_config = {
            'material_weight': 0.4,       # 子力权重
            'positional_weight': 0.3,     # 位置权重
            'tactical_weight': 0.2,       # 战术权重
            'safety_weight': 0.1,         # 安全权重
            'engine_assist': True,        # 是否启用引擎辅助
            'max_analysis_time': 5.0,     # 最大分析时间
            'evaluation_depth': 12        # 评估深度
        }
        
        # 位置评分表 (9x10棋盘)
        self.position_tables = self._initialize_position_tables()
        
        # 战术模式识别
        self.tactical_patterns = self._initialize_tactical_patterns()
        
        # 历史评估缓存
        self._evaluation_cache: Dict[str, Tuple[float, PositionEvaluation]] = {}
        self._cache_timeout = 60.0  # 缓存1分钟
        
        self.is_initialized = False
        
    def initialize(self, with_engine: bool = True) -> bool:
        """初始化评估器"""
        try:
            if with_engine and self.evaluation_config.get('engine_assist'):
                self.engine = PikafishEngine(self.config)
                if not self.engine.initialize():
                    print("警告: 引擎初始化失败，将使用纯算法评估")
                    self.engine = None
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"评估器初始化失败: {e}")
            return False
    
    def evaluate_position(self, board_state: BoardState, 
                         method: EvaluationMethod = EvaluationMethod.COMPREHENSIVE,
                         include_win_rate: bool = True) -> PositionEvaluation:
        """评估局面"""
        if not self.is_initialized:
            raise RuntimeError("评估器未初始化")
        
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{board_state.fen}_{method.value}"
        if cache_key in self._evaluation_cache:
            cached_time, cached_result = self._evaluation_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_result
        
        try:
            # 基础分析
            game_phase = self._determine_game_phase(board_state)
            material_balance = self._evaluate_material(board_state)
            
            # 根据方法选择评估策略
            if method == EvaluationMethod.MATERIAL_ONLY:
                overall_score = material_balance
                positional = PositionalFactors()
                tactical = TacticalElements()
                
            elif method == EvaluationMethod.COMPREHENSIVE:
                overall_score, positional, tactical = self._comprehensive_evaluation(
                    board_state, game_phase
                )
                
            else:
                # 其他方法的简化实现
                overall_score = material_balance
                positional = self._evaluate_positional_factors(board_state)
                tactical = self._evaluate_tactical_elements(board_state)
            
            # 胜率分析
            win_probability = None
            if include_win_rate:
                win_probability = self._calculate_win_probability(
                    board_state, overall_score, method
                )
            
            # 生成建议和分析
            key_features = self._identify_key_features(board_state, positional, tactical)
            strengths, weaknesses = self._analyze_strengths_weaknesses(
                board_state, positional, tactical
            )
            critical_moves = self._identify_critical_moves(board_state)
            
            # 创建评估结果
            evaluation = PositionEvaluation(
                overall_score=overall_score,
                game_phase=game_phase,
                win_probability=win_probability or WinProbability(
                    white_win_rate=0.5, black_win_rate=0.5, draw_rate=0.0, 
                    confidence=0.5, evaluation_method=method
                ),
                material_balance=material_balance,
                positional_factors=positional,
                tactical_elements=tactical,
                key_features=key_features,
                strengths=strengths,
                weaknesses=weaknesses,
                critical_moves=critical_moves,
                evaluation_time=time.time() - start_time
            )
            
            # 缓存结果
            self._evaluation_cache[cache_key] = (time.time(), evaluation)
            
            return evaluation
            
        except Exception as e:
            print(f"局面评估失败: {e}")
            # 返回默认评估
            return PositionEvaluation(
                overall_score=0.0,
                game_phase=GamePhase.MIDDLE_GAME,
                win_probability=WinProbability(
                    white_win_rate=0.5, black_win_rate=0.5, draw_rate=0.0,
                    confidence=0.3, evaluation_method=method
                ),
                evaluation_time=time.time() - start_time
            )
    
    def calculate_win_rate(self, board_state: BoardState) -> WinProbability:
        """计算胜率 (快速方法)"""
        evaluation = self.evaluate_position(
            board_state, EvaluationMethod.COMPREHENSIVE, include_win_rate=True
        )
        return evaluation.win_probability
    
    def compare_positions(self, board1: BoardState, 
                         board2: BoardState) -> Dict[str, float]:
        """比较两个局面"""
        eval1 = self.evaluate_position(board1)
        eval2 = self.evaluate_position(board2)
        
        return {
            'score_difference': eval1.overall_score - eval2.overall_score,
            'material_difference': eval1.material_balance - eval2.material_balance,
            'positional_difference': (
                eval1.positional_factors.center_control - 
                eval2.positional_factors.center_control
            ),
            'win_rate_difference': (
                eval1.win_probability.white_win_rate - 
                eval2.win_probability.white_win_rate
            )
        }
    
    def _comprehensive_evaluation(self, board_state: BoardState, 
                                phase: GamePhase) -> Tuple[float, PositionalFactors, TacticalElements]:
        """综合评估"""
        # 基础因素评估
        material = self._evaluate_material(board_state)
        positional = self._evaluate_positional_factors(board_state)
        tactical = self._evaluate_tactical_elements(board_state)
        
        # 引擎辅助评估 (如果可用)
        engine_bonus = 0.0
        if self.engine:
            try:
                # 获取引擎评估作为参考
                engine_analysis = self._get_engine_evaluation(board_state)
                if engine_analysis:
                    engine_bonus = engine_analysis.score * 0.1  # 小权重引擎加成
            except:
                pass  # 忽略引擎错误
        
        # 加权综合评分
        weights = self.evaluation_config
        overall_score = (
            material * weights['material_weight'] +
            self._positional_to_score(positional) * weights['positional_weight'] +
            self._tactical_to_score(tactical) * weights['tactical_weight'] +
            engine_bonus
        )
        
        # 根据对局阶段调整
        if phase == GamePhase.OPENING:
            overall_score *= 0.8  # 开局评分相对保守
        elif phase == GamePhase.END_GAME:
            overall_score *= 1.2  # 残局评分更重要
        
        return overall_score, positional, tactical
    
    def _calculate_win_probability(self, board_state: BoardState, 
                                 overall_score: float,
                                 method: EvaluationMethod) -> WinProbability:
        """计算胜率"""
        # 基于评分的胜率转换 (使用sigmoid函数)
        def score_to_probability(score: float) -> float:
            """将评分转换为胜率 (使用logistic函数)"""
            # 调整参数使得评分与胜率有合理对应关系
            k = 0.1  # 调节陡峭程度
            return 1.0 / (1.0 + math.exp(-k * score))
        
        # 计算基础胜率
        white_prob = score_to_probability(overall_score)
        black_prob = score_to_probability(-overall_score)
        
        # 和棋率估算
        score_abs = abs(overall_score)
        if score_abs < 10:  # 评分接近时和棋概率更高
            draw_rate = max(0.1, 0.3 - score_abs * 0.02)
        else:
            draw_rate = max(0.05, 0.1 - score_abs * 0.005)
        
        # 归一化概率
        total = white_prob + black_prob + draw_rate
        white_win_rate = white_prob / total
        black_win_rate = black_prob / total
        draw_rate = draw_rate / total
        
        # 置信度评估
        confidence = self._calculate_confidence(board_state, overall_score, method)
        
        return WinProbability(
            white_win_rate=white_win_rate,
            black_win_rate=black_win_rate,
            draw_rate=draw_rate,
            confidence=confidence,
            evaluation_method=method,
            material_advantage=self._evaluate_material(board_state),
            positional_advantage=overall_score - self._evaluate_material(board_state)
        )
    
    def _determine_game_phase(self, board_state: BoardState) -> GamePhase:
        """判断对局阶段"""
        # 简化实现 - 基于剩余子力判断
        # 实际需要分析FEN字符串中的棋子分布
        total_pieces = len([c for c in board_state.fen if c.isalpha()])
        
        if total_pieces > 28:  # 大部分棋子在棋盘上
            return GamePhase.OPENING
        elif total_pieces > 16:  # 中等数量棋子
            return GamePhase.MIDDLE_GAME
        else:  # 少量棋子
            return GamePhase.END_GAME
    
    def _evaluate_material(self, board_state: BoardState) -> float:
        """评估子力平衡"""
        # 简化实现 - 基于FEN解析
        material_count = {'white': 0, 'black': 0}
        
        # 棋子价值映射
        piece_values = {
            'k': 1000, 'q': 90, 'r': 50, 'b': 30, 'n': 30, 'p': 10,  # 黑方
            'K': 1000, 'Q': 90, 'R': 50, 'B': 30, 'N': 30, 'P': 10   # 红方
        }
        
        # 解析FEN中的棋子 (简化)
        fen_position = board_state.fen.split()[0]
        for char in fen_position:
            if char in piece_values:
                if char.isupper():
                    material_count['white'] += piece_values[char]
                else:
                    material_count['black'] += piece_values[char]
        
        return material_count['white'] - material_count['black']
    
    def _evaluate_positional_factors(self, board_state: BoardState) -> PositionalFactors:
        """评估位置因素"""
        # 简化实现
        return PositionalFactors(
            king_safety=0.0,
            center_control=0.0,
            piece_activity=0.0,
            pawn_structure=0.0,
            piece_coordination=0.0,
            space_advantage=0.0
        )
    
    def _evaluate_tactical_elements(self, board_state: BoardState) -> TacticalElements:
        """评估战术要素"""
        # 简化实现
        return TacticalElements()
    
    def _get_engine_evaluation(self, board_state: BoardState) -> Optional[EngineAnalysis]:
        """获取引擎评估"""
        if not self.engine:
            return None
        
        try:
            best_move = self.engine.get_best_move(
                board_state, 
                depth=8,
                time_limit=2.0
            )
            
            if best_move:
                # 构造简化的分析结果
                return EngineAnalysis(
                    depth=8,
                    score=0.0,  # 需要实际解析
                    time=2.0,
                    nodes=0,
                    pv=[best_move],
                    best_move=best_move
                )
        except:
            pass
        
        return None
    
    def _positional_to_score(self, positional: PositionalFactors) -> float:
        """将位置因素转换为分数"""
        return (
            positional.king_safety * 20 +
            positional.center_control * 15 +
            positional.piece_activity * 10 +
            positional.pawn_structure * 8 +
            positional.piece_coordination * 12 +
            positional.space_advantage * 10
        )
    
    def _tactical_to_score(self, tactical: TacticalElements) -> float:
        """将战术要素转换为分数"""
        return (
            tactical.pins * 5 +
            tactical.forks * 8 +
            tactical.skewers * 6 +
            tactical.discovered_attacks * 10 +
            tactical.sacrificial_threats * 15 +
            tactical.back_rank_threats * 12 +
            tactical.checkmate_threats * 50
        )
    
    def _calculate_confidence(self, board_state: BoardState, 
                            score: float, method: EvaluationMethod) -> float:
        """计算评估置信度"""
        base_confidence = 0.7
        
        # 根据评估方法调整置信度
        method_confidence = {
            EvaluationMethod.MATERIAL_ONLY: 0.6,
            EvaluationMethod.POSITIONAL: 0.7,
            EvaluationMethod.TACTICAL: 0.7,
            EvaluationMethod.STRATEGIC: 0.8,
            EvaluationMethod.COMPREHENSIVE: 0.85,
            EvaluationMethod.ENGINE_ASSISTED: 0.95 if self.engine else 0.8
        }
        
        confidence = method_confidence.get(method, base_confidence)
        
        # 根据局面复杂度调整
        if abs(score) > 50:  # 明显优劣势
            confidence += 0.1
        elif abs(score) < 5:  # 均势
            confidence -= 0.1
        
        return max(0.3, min(0.99, confidence))
    
    def _identify_key_features(self, board_state: BoardState, 
                             positional: PositionalFactors,
                             tactical: TacticalElements) -> List[str]:
        """识别关键特征"""
        features = []
        
        if positional.center_control > 0.3:
            features.append("良好的中心控制")
        if tactical.pins > 0:
            features.append(f"存在{tactical.pins}个牵制")
        if tactical.forks > 0:
            features.append(f"存在{tactical.forks}个双击机会")
        
        return features or ["常规局面"]
    
    def _analyze_strengths_weaknesses(self, board_state: BoardState,
                                    positional: PositionalFactors,
                                    tactical: TacticalElements) -> Tuple[List[str], List[str]]:
        """分析优势和弱点"""
        strengths = []
        weaknesses = []
        
        # 根据位置因素分析
        if positional.king_safety > 0.3:
            strengths.append("王的安全较好")
        elif positional.king_safety < -0.3:
            weaknesses.append("王的安全存在隐患")
        
        if positional.piece_activity > 0.3:
            strengths.append("子力活跃度高")
        elif positional.piece_activity < -0.3:
            weaknesses.append("子力相对被动")
        
        return strengths or ["无明显优势"], weaknesses or ["无明显弱点"]
    
    def _identify_critical_moves(self, board_state: BoardState) -> List[str]:
        """识别关键走法"""
        # 简化实现
        return ["需要详细分析"]
    
    def _initialize_position_tables(self) -> Dict[str, List[List[int]]]:
        """初始化位置价值表"""
        # 简化的位置价值表
        return {
            'pawn': [[0 for _ in range(9)] for _ in range(10)],
            'rook': [[0 for _ in range(9)] for _ in range(10)],
            # ... 其他棋子的位置表
        }
    
    def _initialize_tactical_patterns(self) -> Dict[str, Any]:
        """初始化战术模式"""
        return {
            'pin_patterns': [],
            'fork_patterns': [],
            'skewer_patterns': []
        }
    
    def cleanup(self):
        """清理资源"""
        if self.engine:
            self.engine.shutdown()
        if self.executor:
            self.executor.shutdown(wait=True)
        self._evaluation_cache.clear()
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
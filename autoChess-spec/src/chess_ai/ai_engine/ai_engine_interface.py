#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI引擎智能走法建议系统

提供基于Pikafish引擎的高级AI走法建议和分析功能
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future

from chess_ai.core.board_state import BoardState, PieceType, PieceColor
from chess_ai.config.config_manager import ConfigManager
from chess_ai.ai_engine.pikafish_engine import PikafishEngine, EngineAnalysis


class SuggestionLevel(Enum):
    """建议等级"""
    BEGINNER = "beginner"       # 新手级 - 基础建议
    INTERMEDIATE = "intermediate"  # 中级 - 策略建议
    ADVANCED = "advanced"       # 高级 - 深度分析
    MASTER = "master"          # 大师级 - 专业建议


class MoveType(Enum):
    """走法类型"""
    ATTACK = "attack"          # 攻击
    DEFENSE = "defense"        # 防守
    DEVELOPMENT = "development" # 布局
    TACTICS = "tactics"        # 战术
    ENDGAME = "endgame"       # 残局
    SACRIFICE = "sacrifice"    # 牺牲
    POSITIONAL = "positional"  # 位置


@dataclass
class MoveSuggestion:
    """走法建议"""
    move: str                   # 走法记录 (如: h2e2)
    score: float               # 评分 (-100到100)
    confidence: float          # 置信度 (0-1)
    move_type: MoveType        # 走法类型
    depth: int                 # 分析深度
    
    # 详细分析
    description: str = ""      # 中文描述
    pros: List[str] = field(default_factory=list)     # 优点
    cons: List[str] = field(default_factory=list)     # 缺点
    follow_up: List[str] = field(default_factory=list) # 后续变化
    
    # 战术信息
    threats: List[str] = field(default_factory=list)   # 威胁
    defends: List[str] = field(default_factory=list)   # 防守
    pieces_involved: List[str] = field(default_factory=list) # 涉及棋子
    
    # 引擎信息
    engine_analysis: Optional[EngineAnalysis] = None
    analysis_time: float = 0.0


@dataclass
class SuggestionContext:
    """建议上下文"""
    player_level: SuggestionLevel
    time_limit: float = 5.0    # 分析时间限制
    max_suggestions: int = 3   # 最大建议数
    include_risky: bool = False # 是否包含冒险走法
    analyze_opponent: bool = True # 是否分析对手意图
    
    # 过滤条件
    min_confidence: float = 0.3
    exclude_types: List[MoveType] = field(default_factory=list)


@dataclass
class GameSituation:
    """局面情况分析"""
    phase: str                 # 开局/中局/残局
    material_balance: float    # 子力平衡
    king_safety: Tuple[float, float]  # 双方王的安全度
    center_control: float      # 中心控制
    development: Tuple[float, float]  # 双方出子情况
    threats: List[str]         # 当前威胁
    weaknesses: List[str]      # 弱点
    strategic_themes: List[str] # 战略主题


class AIEngineInterface(ABC):
    """AI引擎抽象接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化引擎"""
        pass
    
    @abstractmethod
    def get_move_suggestions(self, board_state: BoardState, 
                           context: SuggestionContext) -> List[MoveSuggestion]:
        """获取走法建议"""
        pass
    
    @abstractmethod
    def analyze_position(self, board_state: BoardState) -> GameSituation:
        """分析局面"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass


class IntelligentAdvisor(AIEngineInterface):
    """智能AI走法顾问系统
    
    基于Pikafish引擎提供智能走法建议和局面分析
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.engine = PikafishEngine(config)
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="AI-Advisor")
        
        # 建议系统配置
        self.suggestion_config = {
            SuggestionLevel.BEGINNER: {
                'max_depth': 6,
                'analysis_time': 2.0,
                'focus_safety': True,
                'explain_basics': True
            },
            SuggestionLevel.INTERMEDIATE: {
                'max_depth': 10,
                'analysis_time': 5.0,
                'focus_tactics': True,
                'show_variations': True
            },
            SuggestionLevel.ADVANCED: {
                'max_depth': 15,
                'analysis_time': 8.0,
                'deep_analysis': True,
                'strategic_advice': True
            },
            SuggestionLevel.MASTER: {
                'max_depth': 20,
                'analysis_time': 12.0,
                'full_analysis': True,
                'computer_suggestions': True
            }
        }
        
        # 走法类型识别模式
        self.move_patterns = {
            MoveType.ATTACK: ['capture', 'check', 'threat', 'attack'],
            MoveType.DEFENSE: ['defend', 'block', 'protect', 'retreat'],
            MoveType.DEVELOPMENT: ['develop', 'castle', 'connect', 'centralize'],
            MoveType.TACTICS: ['fork', 'pin', 'skewer', 'discovered'],
            MoveType.ENDGAME: ['promote', 'king', 'opposition', 'zugzwang'],
            MoveType.SACRIFICE: ['sacrifice', 'gambit', 'exchange'],
            MoveType.POSITIONAL: ['improve', 'structure', 'weakness', 'control']
        }
        
        # 分析缓存
        self._analysis_cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_timeout = 30.0  # 缓存30秒
        
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化AI顾问系统"""
        try:
            if self.engine.initialize():
                self.is_initialized = True
                return True
            return False
        except Exception as e:
            print(f"AI顾问初始化失败: {e}")
            return False
    
    def get_move_suggestions(self, board_state: BoardState, 
                           context: SuggestionContext) -> List[MoveSuggestion]:
        """获取智能走法建议"""
        if not self.is_initialized:
            raise RuntimeError("AI顾问未初始化")
        
        level_config = self.suggestion_config[context.player_level]
        suggestions = []
        
        try:
            # 获取引擎分析
            primary_analysis = self._get_engine_analysis(
                board_state, 
                level_config['max_depth'],
                context.time_limit * 0.6  # 60%时间用于主要分析
            )
            
            if primary_analysis:
                # 生成主要建议
                main_suggestion = self._create_suggestion_from_analysis(
                    primary_analysis, board_state, context
                )
                if main_suggestion and main_suggestion.confidence >= context.min_confidence:
                    suggestions.append(main_suggestion)
            
            # 获取备选走法 
            if len(suggestions) < context.max_suggestions:
                alternatives = self._get_alternative_moves(
                    board_state, context, len(suggestions)
                )
                suggestions.extend(alternatives)
            
            # 根据等级添加详细分析
            suggestions = self._enhance_suggestions_by_level(
                suggestions, board_state, context
            )
            
            # 排序和过滤
            suggestions = self._filter_and_rank_suggestions(
                suggestions, context
            )
            
            return suggestions[:context.max_suggestions]
            
        except Exception as e:
            print(f"生成走法建议失败: {e}")
            return []
    
    def analyze_position(self, board_state: BoardState) -> GameSituation:
        """分析当前局面"""
        if not self.is_initialized:
            raise RuntimeError("AI顾问未初始化")
        
        try:
            # 基础局面分析
            phase = self._determine_game_phase(board_state)
            material_balance = self._calculate_material_balance(board_state)
            king_safety = self._analyze_king_safety(board_state)
            center_control = self._analyze_center_control(board_state)
            development = self._analyze_development(board_state)
            
            # 战术分析
            threats = self._identify_threats(board_state)
            weaknesses = self._identify_weaknesses(board_state)
            themes = self._identify_strategic_themes(board_state, phase)
            
            return GameSituation(
                phase=phase,
                material_balance=material_balance,
                king_safety=king_safety,
                center_control=center_control,
                development=development,
                threats=threats,
                weaknesses=weaknesses,
                strategic_themes=themes
            )
            
        except Exception as e:
            print(f"局面分析失败: {e}")
            # 返回默认分析
            return GameSituation(
                phase="中局",
                material_balance=0.0,
                king_safety=(0.5, 0.5),
                center_control=0.0,
                development=(0.5, 0.5),
                threats=[],
                weaknesses=[],
                strategic_themes=[]
            )
    
    def _get_engine_analysis(self, board_state: BoardState, 
                           depth: int, time_limit: float) -> Optional[EngineAnalysis]:
        """获取引擎分析"""
        cache_key = f"{board_state.fen}_{depth}_{time_limit}"
        
        # 检查缓存
        if cache_key in self._analysis_cache:
            cached_time, cached_result = self._analysis_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_result
        
        try:
            # 异步获取分析
            future = self.executor.submit(
                self.engine.get_best_move, 
                board_state, depth, time_limit
            )
            
            best_move = future.result(timeout=time_limit + 2.0)
            
            if best_move:
                # 构造分析结果 (简化版)
                analysis = EngineAnalysis(
                    depth=depth,
                    score=0.0,  # 需要实际解析引擎输出
                    time=time_limit,
                    nodes=0,
                    pv=[best_move],
                    best_move=best_move
                )
                
                # 缓存结果
                self._analysis_cache[cache_key] = (time.time(), analysis)
                return analysis
                
        except Exception as e:
            print(f"引擎分析失败: {e}")
        
        return None
    
    def _create_suggestion_from_analysis(self, analysis: EngineAnalysis, 
                                       board_state: BoardState,
                                       context: SuggestionContext) -> Optional[MoveSuggestion]:
        """从引擎分析创建走法建议"""
        if not analysis.best_move:
            return None
        
        # 确定走法类型
        move_type = self._classify_move_type(analysis.best_move, board_state)
        
        # 计算置信度
        confidence = min(1.0, abs(analysis.score) / 5.0 + 0.3)
        
        # 生成描述
        description = self._generate_move_description(
            analysis.best_move, move_type, analysis, board_state
        )
        
        return MoveSuggestion(
            move=analysis.best_move,
            score=analysis.score,
            confidence=confidence,
            move_type=move_type,
            depth=analysis.depth,
            description=description,
            engine_analysis=analysis,
            analysis_time=analysis.time
        )
    
    def _get_alternative_moves(self, board_state: BoardState, 
                             context: SuggestionContext, 
                             current_count: int) -> List[MoveSuggestion]:
        """获取备选走法"""
        alternatives = []
        needed = context.max_suggestions - current_count
        
        if needed <= 0:
            return alternatives
        
        try:
            # 使用较低深度快速分析多个候选走法
            level_config = self.suggestion_config[context.player_level]
            quick_depth = max(4, level_config['max_depth'] // 2)
            quick_time = context.time_limit * 0.3 / needed
            
            # 生成几个候选走法 (简化实现)
            candidate_moves = self._generate_candidate_moves(board_state)
            
            for move in candidate_moves[:needed]:
                # 模拟分析结果
                suggestion = MoveSuggestion(
                    move=move,
                    score=0.0,
                    confidence=0.5,
                    move_type=MoveType.POSITIONAL,
                    depth=quick_depth,
                    description=f"备选走法: {move}",
                    analysis_time=quick_time
                )
                alternatives.append(suggestion)
                
        except Exception as e:
            print(f"获取备选走法失败: {e}")
        
        return alternatives
    
    def _enhance_suggestions_by_level(self, suggestions: List[MoveSuggestion],
                                    board_state: BoardState,
                                    context: SuggestionContext) -> List[MoveSuggestion]:
        """根据用户等级增强建议"""
        level_config = self.suggestion_config[context.player_level]
        
        for suggestion in suggestions:
            if context.player_level == SuggestionLevel.BEGINNER:
                # 新手: 强调基础和安全
                if level_config.get('explain_basics'):
                    suggestion.pros.extend([
                        "这步棋比较安全",
                        "有助于棋子发展"
                    ])
                if level_config.get('focus_safety'):
                    suggestion.description += " (注重安全)"
                    
            elif context.player_level == SuggestionLevel.INTERMEDIATE:
                # 中级: 添加战术分析
                if level_config.get('focus_tactics'):
                    suggestion.pros.append("创造战术机会")
                if level_config.get('show_variations'):
                    suggestion.follow_up = ["后续可考虑...", "对手可能应以..."]
                    
            elif context.player_level == SuggestionLevel.ADVANCED:
                # 高级: 深度战略分析
                if level_config.get('strategic_advice'):
                    suggestion.pros.append("符合长期战略")
                    suggestion.cons.append("可能给对手机会")
                    
            elif context.player_level == SuggestionLevel.MASTER:
                # 大师级: 计算机级分析
                if level_config.get('computer_suggestions'):
                    suggestion.description = f"引擎推荐 (深度{suggestion.depth})"
        
        return suggestions
    
    def _filter_and_rank_suggestions(self, suggestions: List[MoveSuggestion],
                                   context: SuggestionContext) -> List[MoveSuggestion]:
        """过滤和排序建议"""
        # 过滤
        filtered = []
        for suggestion in suggestions:
            # 置信度过滤
            if suggestion.confidence < context.min_confidence:
                continue
            # 类型过滤
            if suggestion.move_type in context.exclude_types:
                continue
            # 风险过滤
            if not context.include_risky and suggestion.move_type == MoveType.SACRIFICE:
                continue
            
            filtered.append(suggestion)
        
        # 排序: 先按置信度，再按分数
        filtered.sort(key=lambda x: (x.confidence, abs(x.score)), reverse=True)
        
        return filtered
    
    def _classify_move_type(self, move: str, board_state: BoardState) -> MoveType:
        """分类走法类型"""
        # 简化实现 - 实际需要分析走法特征
        if 'x' in move:  # 吃子
            return MoveType.ATTACK
        elif move.startswith('K'):  # 王车易位
            return MoveType.DEVELOPMENT
        else:
            return MoveType.POSITIONAL
    
    def _generate_move_description(self, move: str, move_type: MoveType,
                                 analysis: EngineAnalysis, board_state: BoardState) -> str:
        """生成走法描述"""
        type_descriptions = {
            MoveType.ATTACK: "主动攻击",
            MoveType.DEFENSE: "稳固防守", 
            MoveType.DEVELOPMENT: "发展棋子",
            MoveType.TACTICS: "战术打击",
            MoveType.ENDGAME: "残局技巧",
            MoveType.SACRIFICE: "牺牲战术",
            MoveType.POSITIONAL: "位置改善"
        }
        
        base_desc = type_descriptions.get(move_type, "常规走法")
        score_desc = "优势" if analysis.score > 0 else "劣势" if analysis.score < 0 else "均势"
        
        return f"{base_desc} - {move} ({score_desc})"
    
    def _generate_candidate_moves(self, board_state: BoardState) -> List[str]:
        """生成候选走法列表"""
        # 简化实现 - 实际需要根据棋盘状态生成合法走法
        return ["h2e2", "b1c3", "c3e4", "a1d1", "g1f3"]
    
    def _determine_game_phase(self, board_state: BoardState) -> str:
        """判断对局阶段"""
        # 简化实现
        return "中局"
    
    def _calculate_material_balance(self, board_state: BoardState) -> float:
        """计算子力平衡"""
        # 简化实现
        return 0.0
    
    def _analyze_king_safety(self, board_state: BoardState) -> Tuple[float, float]:
        """分析王的安全"""
        # 简化实现
        return (0.5, 0.5)
    
    def _analyze_center_control(self, board_state: BoardState) -> float:
        """分析中心控制"""
        # 简化实现
        return 0.0
    
    def _analyze_development(self, board_state: BoardState) -> Tuple[float, float]:
        """分析出子情况"""
        # 简化实现
        return (0.5, 0.5)
    
    def _identify_threats(self, board_state: BoardState) -> List[str]:
        """识别威胁"""
        # 简化实现
        return ["无明显威胁"]
    
    def _identify_weaknesses(self, board_state: BoardState) -> List[str]:
        """识别弱点"""
        # 简化实现
        return ["无明显弱点"]
    
    def _identify_strategic_themes(self, board_state: BoardState, phase: str) -> List[str]:
        """识别战略主题"""
        # 简化实现
        return ["位置发展", "中心控制"]
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.engine:
            self.engine.shutdown()
        if self.executor:
            self.executor.shutdown(wait=True)
        self._analysis_cache.clear()
        self.is_initialized = False
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
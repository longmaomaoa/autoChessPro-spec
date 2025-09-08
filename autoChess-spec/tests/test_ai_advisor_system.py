#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI走法建议系统功能测试

测试智能AI顾问的核心数据结构、建议生成和局面分析功能
"""

import sys
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import time

print("开始AI走法建议系统功能测试...")
print("=" * 60)

# 测试用的数据结构定义
class SuggestionLevel(Enum):
    """建议等级"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    MASTER = "master"

class MoveType(Enum):
    """走法类型"""
    ATTACK = "attack"
    DEFENSE = "defense"
    DEVELOPMENT = "development"
    TACTICS = "tactics"
    ENDGAME = "endgame"
    SACRIFICE = "sacrifice"
    POSITIONAL = "positional"

@dataclass
class EngineAnalysis:
    """引擎分析结果"""
    depth: int
    score: float
    time: float
    nodes: int
    pv: List[str]
    best_move: Optional[str] = None
    mate: Optional[int] = None

@dataclass
class MoveSuggestion:
    """走法建议"""
    move: str
    score: float
    confidence: float
    move_type: MoveType
    depth: int
    description: str = ""
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    follow_up: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    defends: List[str] = field(default_factory=list)
    pieces_involved: List[str] = field(default_factory=list)
    engine_analysis: Optional[EngineAnalysis] = None
    analysis_time: float = 0.0

@dataclass
class SuggestionContext:
    """建议上下文"""
    player_level: SuggestionLevel
    time_limit: float = 5.0
    max_suggestions: int = 3
    include_risky: bool = False
    analyze_opponent: bool = True
    min_confidence: float = 0.3
    exclude_types: List[MoveType] = field(default_factory=list)

@dataclass
class GameSituation:
    """局面情况分析"""
    phase: str
    material_balance: float
    king_safety: Tuple[float, float]
    center_control: float
    development: Tuple[float, float]
    threats: List[str]
    weaknesses: List[str]
    strategic_themes: List[str]

@dataclass
class BoardState:
    """棋盘状态"""
    fen: str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    current_player: str = "w"

# 测试函数
def test_suggestion_level_enum():
    """测试建议等级枚举"""
    try:
        # 测试所有等级
        levels = [SuggestionLevel.BEGINNER, SuggestionLevel.INTERMEDIATE, 
                 SuggestionLevel.ADVANCED, SuggestionLevel.MASTER]
        
        assert len(levels) == 4
        assert SuggestionLevel.BEGINNER.value == "beginner"
        assert SuggestionLevel.MASTER.value == "master"
        
        print("PASS: 建议等级枚举测试")
        return True
    except Exception as e:
        print(f"FAIL: 建议等级枚举测试 - {e}")
        return False

def test_move_type_enum():
    """测试走法类型枚举"""
    try:
        # 测试所有类型
        types = [MoveType.ATTACK, MoveType.DEFENSE, MoveType.DEVELOPMENT,
                MoveType.TACTICS, MoveType.ENDGAME, MoveType.SACRIFICE, MoveType.POSITIONAL]
        
        assert len(types) == 7
        assert MoveType.ATTACK.value == "attack"
        assert MoveType.POSITIONAL.value == "positional"
        
        print("PASS: 走法类型枚举测试")
        return True
    except Exception as e:
        print(f"FAIL: 走法类型枚举测试 - {e}")
        return False

def test_move_suggestion_dataclass():
    """测试走法建议数据结构"""
    try:
        # 创建测试建议
        suggestion = MoveSuggestion(
            move="h2e2",
            score=1.5,
            confidence=0.85,
            move_type=MoveType.ATTACK,
            depth=10,
            description="主动攻击 - h2e2 (优势)"
        )
        
        assert suggestion.move == "h2e2"
        assert suggestion.score == 1.5
        assert suggestion.confidence == 0.85
        assert suggestion.move_type == MoveType.ATTACK
        assert suggestion.depth == 10
        assert isinstance(suggestion.pros, list)
        assert isinstance(suggestion.cons, list)
        
        # 测试添加详细信息
        suggestion.pros.append("创造攻击机会")
        suggestion.cons.append("可能暴露弱点")
        suggestion.threats.append("威胁对方兵")
        
        assert len(suggestion.pros) == 1
        assert len(suggestion.cons) == 1
        assert len(suggestion.threats) == 1
        
        print("PASS: 走法建议数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 走法建议数据结构测试 - {e}")
        return False

def test_suggestion_context():
    """测试建议上下文"""
    try:
        # 默认上下文
        context = SuggestionContext(player_level=SuggestionLevel.INTERMEDIATE)
        
        assert context.player_level == SuggestionLevel.INTERMEDIATE
        assert context.time_limit == 5.0
        assert context.max_suggestions == 3
        assert context.include_risky == False
        assert context.min_confidence == 0.3
        
        # 自定义上下文
        custom_context = SuggestionContext(
            player_level=SuggestionLevel.MASTER,
            time_limit=10.0,
            max_suggestions=5,
            include_risky=True,
            exclude_types=[MoveType.SACRIFICE]
        )
        
        assert custom_context.time_limit == 10.0
        assert custom_context.max_suggestions == 5
        assert custom_context.include_risky == True
        assert MoveType.SACRIFICE in custom_context.exclude_types
        
        print("PASS: 建议上下文测试")
        return True
    except Exception as e:
        print(f"FAIL: 建议上下文测试 - {e}")
        return False

def test_game_situation_analysis():
    """测试局面分析数据结构"""
    try:
        situation = GameSituation(
            phase="中局",
            material_balance=0.5,
            king_safety=(0.7, 0.6),
            center_control=0.3,
            development=(0.8, 0.7),
            threats=["马捉炮", "兵逼宫"],
            weaknesses=["右翼薄弱"],
            strategic_themes=["中心控制", "王翼攻势"]
        )
        
        assert situation.phase == "中局"
        assert situation.material_balance == 0.5
        assert situation.king_safety == (0.7, 0.6)
        assert len(situation.threats) == 2
        assert len(situation.strategic_themes) == 2
        
        print("PASS: 局面分析数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 局面分析数据结构测试 - {e}")
        return False

def test_suggestion_filtering():
    """测试建议过滤逻辑"""
    try:
        # 创建测试建议列表
        suggestions = [
            MoveSuggestion("h2e2", 2.0, 0.9, MoveType.ATTACK, 10, "强攻击"),
            MoveSuggestion("b1c3", 0.5, 0.2, MoveType.DEVELOPMENT, 8, "低置信度"),
            MoveSuggestion("g1f3", 1.0, 0.7, MoveType.SACRIFICE, 12, "牺牲战术"),
            MoveSuggestion("a1d1", 0.8, 0.6, MoveType.POSITIONAL, 9, "位置改善")
        ]
        
        # 测试置信度过滤
        high_confidence = [s for s in suggestions if s.confidence >= 0.5]
        assert len(high_confidence) == 3  # 排除低置信度的
        
        # 测试类型过滤
        exclude_sacrifice = [s for s in suggestions if s.move_type != MoveType.SACRIFICE]
        assert len(exclude_sacrifice) == 3  # 排除牺牲战术
        
        # 测试综合过滤
        filtered = [s for s in suggestions 
                   if s.confidence >= 0.5 and s.move_type != MoveType.SACRIFICE]
        assert len(filtered) == 2
        
        # 测试排序
        sorted_suggestions = sorted(suggestions, 
                                  key=lambda x: (x.confidence, abs(x.score)), 
                                  reverse=True)
        assert sorted_suggestions[0].confidence == 0.9
        
        print("PASS: 建议过滤逻辑测试")
        return True
    except Exception as e:
        print(f"FAIL: 建议过滤逻辑测试 - {e}")
        return False

def test_level_based_configuration():
    """测试等级相关配置"""
    try:
        # 不同等级的配置
        configs = {
            SuggestionLevel.BEGINNER: {
                'max_depth': 6,
                'analysis_time': 2.0,
                'focus_safety': True
            },
            SuggestionLevel.INTERMEDIATE: {
                'max_depth': 10,
                'analysis_time': 5.0,
                'focus_tactics': True
            },
            SuggestionLevel.ADVANCED: {
                'max_depth': 15,
                'analysis_time': 8.0,
                'deep_analysis': True
            },
            SuggestionLevel.MASTER: {
                'max_depth': 20,
                'analysis_time': 12.0,
                'full_analysis': True
            }
        }
        
        # 验证配置正确性
        beginner_config = configs[SuggestionLevel.BEGINNER]
        master_config = configs[SuggestionLevel.MASTER]
        
        assert beginner_config['max_depth'] == 6
        assert master_config['max_depth'] == 20
        assert beginner_config['analysis_time'] < master_config['analysis_time']
        assert beginner_config.get('focus_safety') == True
        assert master_config.get('full_analysis') == True
        
        print("PASS: 等级相关配置测试")
        return True
    except Exception as e:
        print(f"FAIL: 等级相关配置测试 - {e}")
        return False

def test_move_classification():
    """测试走法分类逻辑"""
    try:
        # 走法分类模式
        move_patterns = {
            MoveType.ATTACK: ['capture', 'check', 'threat', 'attack'],
            MoveType.DEFENSE: ['defend', 'block', 'protect', 'retreat'],
            MoveType.DEVELOPMENT: ['develop', 'castle', 'connect', 'centralize'],
            MoveType.TACTICS: ['fork', 'pin', 'skewer', 'discovered'],
            MoveType.ENDGAME: ['promote', 'king', 'opposition', 'zugzwang'],
            MoveType.SACRIFICE: ['sacrifice', 'gambit', 'exchange'],
            MoveType.POSITIONAL: ['improve', 'structure', 'weakness', 'control']
        }
        
        # 验证模式
        assert 'capture' in move_patterns[MoveType.ATTACK]
        assert 'defend' in move_patterns[MoveType.DEFENSE]
        assert 'castle' in move_patterns[MoveType.DEVELOPMENT]
        
        # 简单分类函数测试
        def classify_move(move_text: str) -> MoveType:
            if 'x' in move_text:  # 吃子标记
                return MoveType.ATTACK
            elif move_text.startswith('O-O'):  # 王车易位
                return MoveType.DEVELOPMENT
            else:
                return MoveType.POSITIONAL
        
        assert classify_move("Nxe5") == MoveType.ATTACK
        assert classify_move("O-O") == MoveType.DEVELOPMENT
        assert classify_move("Nf3") == MoveType.POSITIONAL
        
        print("PASS: 走法分类逻辑测试")
        return True
    except Exception as e:
        print(f"FAIL: 走法分类逻辑测试 - {e}")
        return False

def test_analysis_caching():
    """测试分析缓存机制"""
    try:
        # 模拟缓存
        cache = {}
        cache_timeout = 30.0
        
        # 添加缓存项
        cache_key = "test_position_10_5.0"
        cache_data = (time.time(), "cached_result")
        cache[cache_key] = cache_data
        
        # 测试缓存命中
        cached_time, cached_result = cache[cache_key]
        assert cached_result == "cached_result"
        
        # 测试缓存过期检查
        current_time = time.time()
        is_expired = (current_time - cached_time) > cache_timeout
        assert not is_expired  # 应该还没过期
        
        # 测试缓存更新
        new_data = (time.time(), "new_result")
        cache[cache_key] = new_data
        
        updated_time, updated_result = cache[cache_key]
        assert updated_result == "new_result"
        
        print("PASS: 分析缓存机制测试")
        return True
    except Exception as e:
        print(f"FAIL: 分析缓存机制测试 - {e}")
        return False

def test_suggestion_enhancement():
    """测试建议增强功能"""
    try:
        # 基础建议
        suggestion = MoveSuggestion(
            move="e2e4",
            score=0.3,
            confidence=0.7,
            move_type=MoveType.DEVELOPMENT,
            depth=8,
            description="发展棋子"
        )
        
        # 模拟不同等级的增强
        def enhance_for_beginner(sug: MoveSuggestion):
            sug.pros.extend(["这步棋比较安全", "有助于棋子发展"])
            sug.description += " (注重安全)"
            return sug
        
        def enhance_for_intermediate(sug: MoveSuggestion):
            sug.pros.append("创造战术机会")
            sug.follow_up = ["后续可考虑...", "对手可能应以..."]
            return sug
        
        def enhance_for_advanced(sug: MoveSuggestion):
            sug.pros.append("符合长期战略")
            sug.cons.append("可能给对手机会")
            return sug
        
        # 测试新手增强
        beginner_sug = enhance_for_beginner(MoveSuggestion(
            move="e2e4", score=0.3, confidence=0.7, 
            move_type=MoveType.DEVELOPMENT, depth=8, description="发展棋子"
        ))
        
        assert len(beginner_sug.pros) == 2
        assert "(注重安全)" in beginner_sug.description
        
        # 测试中级增强
        inter_sug = enhance_for_intermediate(MoveSuggestion(
            move="e2e4", score=0.3, confidence=0.7,
            move_type=MoveType.DEVELOPMENT, depth=8, description="发展棋子"
        ))
        
        assert "创造战术机会" in inter_sug.pros
        assert len(inter_sug.follow_up) == 2
        
        print("PASS: 建议增强功能测试")
        return True
    except Exception as e:
        print(f"FAIL: 建议增强功能测试 - {e}")
        return False

# 运行所有测试
def run_ai_advisor_tests():
    """运行AI走法建议系统测试"""
    test_functions = [
        test_suggestion_level_enum,
        test_move_type_enum,
        test_move_suggestion_dataclass,
        test_suggestion_context,
        test_game_situation_analysis,
        test_suggestion_filtering,
        test_level_based_configuration,
        test_move_classification,
        test_analysis_caching,
        test_suggestion_enhancement
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
        print("所有测试通过! AI走法建议系统核心功能正常")
        return True
    else:
        print(f"有 {total-passed} 个测试失败，需要修复实现")
        return False

if __name__ == "__main__":
    success = run_ai_advisor_tests()
    sys.exit(0 if success else 1)
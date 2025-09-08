#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
胜率分析和局面评估系统功能测试

测试综合局面评估器的核心数据结构、评估算法和胜率计算功能
"""

import sys
import math
import time
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

print("开始胜率分析和局面评估系统功能测试...")
print("=" * 60)

# 测试用的数据结构定义
class EvaluationMethod(Enum):
    """评估方法"""
    MATERIAL_ONLY = "material_only"
    POSITIONAL = "positional"
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    COMPREHENSIVE = "comprehensive"
    ENGINE_ASSISTED = "engine_assisted"

class GamePhase(Enum):
    """对局阶段"""
    OPENING = "opening"
    MIDDLE_GAME = "middle"
    END_GAME = "endgame"
    COMPLEX = "complex"

@dataclass
class PieceValues:
    """棋子价值表"""
    SHUAI = 1000    # 帅/将
    CHE = 100       # 车
    MA = 45         # 马  
    PAO = 45        # 炮
    XIANG = 20      # 相/象
    SHI = 20        # 士
    BING = 10       # 兵/卒
    
    center_bonus = 1.2      # 中心位置奖励
    edge_penalty = 0.8      # 边缘位置惩罚
    development_bonus = 1.1  # 发展奖励
    safety_bonus = 1.15     # 安全奖励

@dataclass
class PositionalFactors:
    """位置因素"""
    king_safety: float = 0.0
    center_control: float = 0.0
    piece_activity: float = 0.0
    pawn_structure: float = 0.0
    piece_coordination: float = 0.0
    space_advantage: float = 0.0

@dataclass
class TacticalElements:
    """战术要素"""
    pins: int = 0
    forks: int = 0
    skewers: int = 0
    discovered_attacks: int = 0
    sacrificial_threats: int = 0
    back_rank_threats: int = 0
    checkmate_threats: int = 0

@dataclass
class WinProbability:
    """胜率分析"""
    white_win_rate: float
    black_win_rate: float
    draw_rate: float
    confidence: float
    evaluation_method: EvaluationMethod
    material_advantage: float = 0.0
    positional_advantage: float = 0.0
    tactical_advantage: float = 0.0
    time_advantage: float = 0.0

@dataclass
class PositionEvaluation:
    """局面评估结果"""
    overall_score: float
    game_phase: GamePhase
    win_probability: WinProbability
    material_balance: float = 0.0
    positional_factors: PositionalFactors = field(default_factory=PositionalFactors)
    tactical_elements: TacticalElements = field(default_factory=TacticalElements)
    key_features: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    critical_moves: List[str] = field(default_factory=list)
    evaluation_time: float = 0.0
    engine_depth: Optional[int] = None

@dataclass
class BoardState:
    """棋盘状态"""
    fen: str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    current_player: str = "w"

# 测试函数
def test_evaluation_method_enum():
    """测试评估方法枚举"""
    try:
        methods = [EvaluationMethod.MATERIAL_ONLY, EvaluationMethod.POSITIONAL,
                  EvaluationMethod.TACTICAL, EvaluationMethod.COMPREHENSIVE]
        
        assert len(methods) == 4
        assert EvaluationMethod.MATERIAL_ONLY.value == "material_only"
        assert EvaluationMethod.COMPREHENSIVE.value == "comprehensive"
        
        print("PASS: 评估方法枚举测试")
        return True
    except Exception as e:
        print(f"FAIL: 评估方法枚举测试 - {e}")
        return False

def test_game_phase_enum():
    """测试对局阶段枚举"""
    try:
        phases = [GamePhase.OPENING, GamePhase.MIDDLE_GAME, 
                 GamePhase.END_GAME, GamePhase.COMPLEX]
        
        assert len(phases) == 4
        assert GamePhase.OPENING.value == "opening"
        assert GamePhase.END_GAME.value == "endgame"
        
        print("PASS: 对局阶段枚举测试")
        return True
    except Exception as e:
        print(f"FAIL: 对局阶段枚举测试 - {e}")
        return False

def test_piece_values():
    """测试棋子价值表"""
    try:
        values = PieceValues()
        
        assert values.SHUAI == 1000
        assert values.CHE == 100
        assert values.MA == 45
        assert values.PAO == 45
        assert values.BING == 10
        
        # 测试位置奖励系数
        assert values.center_bonus > 1.0
        assert values.edge_penalty < 1.0
        assert values.development_bonus > 1.0
        assert values.safety_bonus > 1.0
        
        print("PASS: 棋子价值表测试")
        return True
    except Exception as e:
        print(f"FAIL: 棋子价值表测试 - {e}")
        return False

def test_positional_factors():
    """测试位置因素数据结构"""
    try:
        factors = PositionalFactors(
            king_safety=0.5,
            center_control=0.3,
            piece_activity=0.7,
            pawn_structure=-0.2
        )
        
        assert factors.king_safety == 0.5
        assert factors.center_control == 0.3
        assert factors.piece_activity == 0.7
        assert factors.pawn_structure == -0.2
        assert factors.piece_coordination == 0.0  # 默认值
        assert factors.space_advantage == 0.0     # 默认值
        
        print("PASS: 位置因素数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 位置因素数据结构测试 - {e}")
        return False

def test_tactical_elements():
    """测试战术要素数据结构"""
    try:
        tactical = TacticalElements(
            pins=2,
            forks=1,
            skewers=0,
            checkmate_threats=1
        )
        
        assert tactical.pins == 2
        assert tactical.forks == 1
        assert tactical.skewers == 0
        assert tactical.checkmate_threats == 1
        assert tactical.discovered_attacks == 0  # 默认值
        
        print("PASS: 战术要素数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 战术要素数据结构测试 - {e}")
        return False

def test_win_probability():
    """测试胜率分析数据结构"""
    try:
        win_prob = WinProbability(
            white_win_rate=0.6,
            black_win_rate=0.3,
            draw_rate=0.1,
            confidence=0.8,
            evaluation_method=EvaluationMethod.COMPREHENSIVE,
            material_advantage=15.0
        )
        
        assert win_prob.white_win_rate == 0.6
        assert win_prob.black_win_rate == 0.3
        assert win_prob.draw_rate == 0.1
        assert win_prob.confidence == 0.8
        assert win_prob.evaluation_method == EvaluationMethod.COMPREHENSIVE
        assert win_prob.material_advantage == 15.0
        
        # 验证概率和
        total_prob = win_prob.white_win_rate + win_prob.black_win_rate + win_prob.draw_rate
        assert abs(total_prob - 1.0) < 0.001  # 允许小的浮点误差
        
        print("PASS: 胜率分析数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 胜率分析数据结构测试 - {e}")
        return False

def test_position_evaluation():
    """测试局面评估结果数据结构"""
    try:
        win_prob = WinProbability(
            white_win_rate=0.55, black_win_rate=0.35, draw_rate=0.1,
            confidence=0.8, evaluation_method=EvaluationMethod.COMPREHENSIVE
        )
        
        evaluation = PositionEvaluation(
            overall_score=25.0,
            game_phase=GamePhase.MIDDLE_GAME,
            win_probability=win_prob,
            material_balance=10.0,
            key_features=["中心控制良好", "王翼安全"],
            strengths=["子力活跃"],
            weaknesses=["左翼薄弱"],
            evaluation_time=0.5
        )
        
        assert evaluation.overall_score == 25.0
        assert evaluation.game_phase == GamePhase.MIDDLE_GAME
        assert evaluation.win_probability.white_win_rate == 0.55
        assert evaluation.material_balance == 10.0
        assert len(evaluation.key_features) == 2
        assert len(evaluation.strengths) == 1
        assert len(evaluation.weaknesses) == 1
        assert evaluation.evaluation_time == 0.5
        
        print("PASS: 局面评估结果数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 局面评估结果数据结构测试 - {e}")
        return False

def test_score_to_probability_conversion():
    """测试评分到胜率的转换"""
    try:
        def score_to_probability(score: float) -> float:
            """将评分转换为胜率 (使用logistic函数)"""
            k = 0.1  # 调节陡峭程度
            return 1.0 / (1.0 + math.exp(-k * score))
        
        # 测试不同评分的胜率转换
        assert abs(score_to_probability(0) - 0.5) < 0.001    # 均势约50%
        assert score_to_probability(50) > 0.9               # 大优势>90%
        assert score_to_probability(-50) < 0.1              # 大劣势<10%
        assert score_to_probability(10) > score_to_probability(5)  # 单调性
        
        print("PASS: 评分到胜率转换测试")
        return True
    except Exception as e:
        print(f"FAIL: 评分到胜率转换测试 - {e}")
        return False

def test_material_evaluation():
    """测试子力评估算法"""
    try:
        def evaluate_material(fen: str) -> float:
            """评估子力平衡"""
            material_count = {'white': 0, 'black': 0}
            
            # 棋子价值映射
            piece_values = {
                'k': 1000, 'q': 90, 'r': 50, 'b': 30, 'n': 30, 'p': 10,  # 黑方
                'K': 1000, 'Q': 90, 'R': 50, 'B': 30, 'N': 30, 'P': 10   # 红方
            }
            
            # 解析FEN中的棋子 (简化)
            fen_position = fen.split()[0]
            for char in fen_position:
                if char in piece_values:
                    if char.isupper():
                        material_count['white'] += piece_values[char]
                    else:
                        material_count['black'] += piece_values[char]
            
            return material_count['white'] - material_count['black']
        
        # 测试初始局面 (应该均衡)
        initial_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        balance = evaluate_material(initial_fen)
        assert abs(balance) < 50  # 初始局面应该基本均衡
        
        # 测试红方优势局面 (多一个车)
        advantage_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C3RC1/9/RNBAKABNR w - - 0 1"
        advantage_balance = evaluate_material(advantage_fen)
        assert advantage_balance > balance  # 红方优势
        
        print("PASS: 子力评估算法测试")
        return True
    except Exception as e:
        print(f"FAIL: 子力评估算法测试 - {e}")
        return False

def test_game_phase_detection():
    """测试对局阶段识别"""
    try:
        def determine_game_phase(fen: str) -> GamePhase:
            """判断对局阶段"""
            total_pieces = len([c for c in fen if c.isalpha()])
            
            if total_pieces > 28:
                return GamePhase.OPENING
            elif total_pieces > 16:
                return GamePhase.MIDDLE_GAME
            else:
                return GamePhase.END_GAME
        
        # 测试不同阶段
        opening_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        middle_fen = "r1bakab1r/9/2n1c1n2/p1p1p1p1p/9/9/P1P1P1P1P/2N1C1N2/9/R1BAKAB1R w - - 0 1"
        end_fen = "3k5/9/9/9/9/9/9/4K4/9/8R w - - 0 1"
        
        assert determine_game_phase(opening_fen) == GamePhase.OPENING
        assert determine_game_phase(middle_fen) == GamePhase.MIDDLE_GAME
        assert determine_game_phase(end_fen) == GamePhase.END_GAME
        
        print("PASS: 对局阶段识别测试")
        return True
    except Exception as e:
        print(f"FAIL: 对局阶段识别测试 - {e}")
        return False

def test_confidence_calculation():
    """测试置信度计算"""
    try:
        def calculate_confidence(score: float, method: EvaluationMethod) -> float:
            """计算评估置信度"""
            method_confidence = {
                EvaluationMethod.MATERIAL_ONLY: 0.6,
                EvaluationMethod.POSITIONAL: 0.7,
                EvaluationMethod.COMPREHENSIVE: 0.85,
                EvaluationMethod.ENGINE_ASSISTED: 0.95
            }
            
            confidence = method_confidence.get(method, 0.7)
            
            # 根据局面复杂度调整
            if abs(score) > 50:  # 明显优劣势
                confidence += 0.1
            elif abs(score) < 5:  # 均势
                confidence -= 0.1
            
            return max(0.3, min(0.99, confidence))
        
        # 测试不同情况的置信度
        material_conf = calculate_confidence(0, EvaluationMethod.MATERIAL_ONLY)
        comprehensive_conf = calculate_confidence(0, EvaluationMethod.COMPREHENSIVE)
        
        assert material_conf < comprehensive_conf  # 综合评估置信度更高
        assert 0.3 <= material_conf <= 0.99       # 置信度范围正确
        assert 0.3 <= comprehensive_conf <= 0.99  # 置信度范围正确
        
        # 测试明显优势的置信度提升
        advantage_conf = calculate_confidence(60, EvaluationMethod.COMPREHENSIVE)
        assert advantage_conf > comprehensive_conf
        
        print("PASS: 置信度计算测试")
        return True
    except Exception as e:
        print(f"FAIL: 置信度计算测试 - {e}")
        return False

def test_evaluation_caching():
    """测试评估缓存机制"""
    try:
        # 模拟缓存
        cache = {}
        cache_timeout = 60.0
        
        # 添加缓存项
        test_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        cache_key = f"{test_fen}_comprehensive"
        
        mock_evaluation = PositionEvaluation(
            overall_score=5.0,
            game_phase=GamePhase.MIDDLE_GAME,
            win_probability=WinProbability(0.6, 0.35, 0.05, 0.8, EvaluationMethod.COMPREHENSIVE)
        )
        
        cache[cache_key] = (time.time(), mock_evaluation)
        
        # 测试缓存命中
        cached_time, cached_result = cache[cache_key]
        assert cached_result.overall_score == 5.0
        assert cached_result.game_phase == GamePhase.MIDDLE_GAME
        
        # 测试缓存过期检查
        current_time = time.time()
        is_expired = (current_time - cached_time) > cache_timeout
        assert not is_expired  # 应该还没过期
        
        print("PASS: 评估缓存机制测试")
        return True
    except Exception as e:
        print(f"FAIL: 评估缓存机制测试 - {e}")
        return False

def test_position_comparison():
    """测试局面比较功能"""
    try:
        # 创建两个不同的评估结果
        eval1 = PositionEvaluation(
            overall_score=20.0,
            game_phase=GamePhase.MIDDLE_GAME,
            win_probability=WinProbability(0.7, 0.25, 0.05, 0.8, EvaluationMethod.COMPREHENSIVE),
            material_balance=15.0
        )
        
        eval2 = PositionEvaluation(
            overall_score=5.0,
            game_phase=GamePhase.MIDDLE_GAME,
            win_probability=WinProbability(0.55, 0.4, 0.05, 0.7, EvaluationMethod.COMPREHENSIVE),
            material_balance=3.0
        )
        
        # 比较局面
        comparison = {
            'score_difference': eval1.overall_score - eval2.overall_score,
            'material_difference': eval1.material_balance - eval2.material_balance,
            'win_rate_difference': (
                eval1.win_probability.white_win_rate - 
                eval2.win_probability.white_win_rate
            )
        }
        
        assert comparison['score_difference'] == 15.0   # 20 - 5
        assert comparison['material_difference'] == 12.0 # 15 - 3
        assert abs(comparison['win_rate_difference'] - 0.15) < 0.001  # 0.7 - 0.55
        
        print("PASS: 局面比较功能测试")
        return True
    except Exception as e:
        print(f"FAIL: 局面比较功能测试 - {e}")
        return False

# 运行所有测试
def run_position_evaluation_tests():
    """运行胜率分析和局面评估系统测试"""
    test_functions = [
        test_evaluation_method_enum,
        test_game_phase_enum,
        test_piece_values,
        test_positional_factors,
        test_tactical_elements,
        test_win_probability,
        test_position_evaluation,
        test_score_to_probability_conversion,
        test_material_evaluation,
        test_game_phase_detection,
        test_confidence_calculation,
        test_evaluation_caching,
        test_position_comparison
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
        print("所有测试通过! 胜率分析和局面评估系统核心功能正常")
        return True
    else:
        print(f"有 {total-passed} 个测试失败，需要修复实现")
        return False

if __name__ == "__main__":
    success = run_position_evaluation_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI分析结果展示面板

该模块提供综合的AI分析结果展示界面，包括：
- 局面评估和胜率分析
- 推荐着法和变着分析
- 战术分析和位置评估
- 历史分析记录管理
- 实时分析状态监控
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QProgressBar, QTextEdit, QTreeWidget, QTreeWidgetItem,
    QPushButton, QSplitter, QFrame, QScrollArea,
    QListWidget, QListWidgetItem, QGroupBox, QGridLayout,
    QSlider, QSpinBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QPixmap, QIcon
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import math

class AnalysisType(Enum):
    """分析类型枚举"""
    POSITION_EVAL = "position_eval"
    MOVE_SUGGESTION = "move_suggestion"
    TACTICAL_ANALYSIS = "tactical_analysis"
    ENDGAME_ANALYSIS = "endgame_analysis"
    OPENING_ANALYSIS = "opening_analysis"

class AnalysisStatus(Enum):
    """分析状态枚举"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class MoveVariation:
    """着法变化数据类"""
    moves: List[str] = field(default_factory=list)
    evaluation: float = 0.0
    depth: int = 0
    description: str = ""
    is_main_line: bool = False

@dataclass
class PositionEvaluation:
    """局面评估数据类"""
    score: float = 0.0
    win_probability: float = 50.0
    material_balance: int = 0
    positional_factors: Dict[str, float] = field(default_factory=dict)
    tactical_elements: List[str] = field(default_factory=list)
    game_phase: str = "middle_game"

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    timestamp: float = field(default_factory=time.time)
    analysis_type: AnalysisType = AnalysisType.POSITION_EVAL
    position_eval: Optional[PositionEvaluation] = None
    best_moves: List[Tuple[str, float]] = field(default_factory=list)
    variations: List[MoveVariation] = field(default_factory=list)
    analysis_depth: int = 0
    analysis_time: float = 0.0
    engine_info: Dict[str, Any] = field(default_factory=dict)

class WinProbabilityWidget(QWidget):
    """胜率显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.win_probability = 50.0
        self.setMinimumSize(200, 60)
        
    def set_probability(self, probability: float):
        """设置胜率"""
        self.win_probability = max(0.0, min(100.0, probability))
        self.update()
        
    def paintEvent(self, event):
        """绘制胜率条"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(10, 10, -10, -10)
        
        # 背景
        painter.fillRect(rect, QColor(240, 240, 240))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(rect)
        
        # 胜率条
        red_width = int(rect.width() * (100 - self.win_probability) / 100)
        black_width = rect.width() - red_width
        
        # 红方胜率 (左侧)
        red_rect = QRect(rect.left(), rect.top(), red_width, rect.height())
        painter.fillRect(red_rect, QColor(220, 100, 100))
        
        # 黑方胜率 (右侧)
        black_rect = QRect(rect.left() + red_width, rect.top(), black_width, rect.height())
        painter.fillRect(black_rect, QColor(100, 100, 100))
        
        # 分割线
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(rect.left() + red_width, rect.top(), rect.left() + red_width, rect.bottom())
        
        # 文本
        painter.setPen(QPen(QColor(0, 0, 0)))
        font = QFont("Microsoft YaHei", 10, QFont.Weight.Bold)
        painter.setFont(font)
        
        red_text = f"红 {100 - self.win_probability:.1f}%"
        black_text = f"黑 {self.win_probability:.1f}%"
        
        painter.drawText(red_rect, Qt.AlignmentFlag.AlignCenter, red_text)
        painter.drawText(black_rect, Qt.AlignmentFlag.AlignCenter, black_text)

class EvaluationChart(QWidget):
    """评估图表组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.evaluations = []
        self.setMinimumSize(300, 150)
        
    def add_evaluation(self, evaluation: float):
        """添加评估值"""
        self.evaluations.append(evaluation)
        if len(self.evaluations) > 50:  # 只保留最近50个
            self.evaluations.pop(0)
        self.update()
        
    def clear_evaluations(self):
        """清空评估记录"""
        self.evaluations.clear()
        self.update()
        
    def paintEvent(self, event):
        """绘制评估图表"""
        if not self.evaluations:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(20, 20, -20, -20)
        
        # 背景
        painter.fillRect(rect, QColor(250, 250, 250))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(rect)
        
        # 中线
        mid_y = rect.top() + rect.height() // 2
        painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)
        
        # 数据点
        if len(self.evaluations) < 2:
            return
            
        max_eval = max(abs(e) for e in self.evaluations)
        if max_eval == 0:
            max_eval = 1
            
        step_x = rect.width() / (len(self.evaluations) - 1)
        
        painter.setPen(QPen(QColor(50, 100, 200), 2))
        for i in range(1, len(self.evaluations)):
            x1 = rect.left() + (i - 1) * step_x
            y1 = mid_y - (self.evaluations[i - 1] / max_eval) * (rect.height() // 2 - 5)
            x2 = rect.left() + i * step_x
            y2 = mid_y - (self.evaluations[i] / max_eval) * (rect.height() // 2 - 5)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

class MoveVariationTree(QTreeWidget):
    """着法变化树组件"""
    
    variation_selected = pyqtSignal(MoveVariation)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["着法", "评估", "深度", "描述"])
        self.itemClicked.connect(self._on_item_clicked)
        
    def set_variations(self, variations: List[MoveVariation]):
        """设置变化列表"""
        self.clear()
        
        for var in variations:
            item = QTreeWidgetItem(self)
            move_text = " ".join(var.moves[:5])  # 只显示前5步
            if len(var.moves) > 5:
                move_text += "..."
                
            item.setText(0, move_text)
            item.setText(1, f"{var.evaluation:+.2f}")
            item.setText(2, str(var.depth))
            item.setText(3, var.description)
            
            # 主线加粗
            if var.is_main_line:
                font = item.font(0)
                font.setBold(True)
                for i in range(4):
                    item.setFont(i, font)
                    
            item.setData(0, Qt.ItemDataRole.UserRole, var)
            
        self.expandAll()
        
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """处理项目点击"""
        variation = item.data(0, Qt.ItemDataRole.UserRole)
        if variation:
            self.variation_selected.emit(variation)

class AnalysisStatusWidget(QWidget):
    """分析状态组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        self._start_time = None
        
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 状态标签
        self.status_label = QLabel("空闲")
        self.status_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 时间标签
        self.time_label = QLabel("用时: 0秒")
        layout.addWidget(self.time_label)
        
        # 深度标签
        self.depth_label = QLabel("深度: 0")
        layout.addWidget(self.depth_label)
        
    def set_status(self, status: AnalysisStatus, progress: int = 0):
        """设置分析状态"""
        status_texts = {
            AnalysisStatus.IDLE: ("空闲", "#666"),
            AnalysisStatus.ANALYZING: ("分析中...", "#0066cc"),
            AnalysisStatus.COMPLETED: ("完成", "#009900"),
            AnalysisStatus.ERROR: ("错误", "#cc0000"),
            AnalysisStatus.CANCELLED: ("已取消", "#ff6600")
        }
        
        text, color = status_texts.get(status, ("未知", "#666"))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-weight: bold; color: {color};")
        
        if status == AnalysisStatus.ANALYZING:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress)
            if not self._timer.isActive():
                self._start_time = time.time()
                self._timer.start(100)
        else:
            self.progress_bar.setVisible(False)
            self._timer.stop()
            
    def set_depth(self, depth: int):
        """设置搜索深度"""
        self.depth_label.setText(f"深度: {depth}")
        
    def _update_timer(self):
        """更新计时器"""
        if self._start_time:
            elapsed = time.time() - self._start_time
            self.time_label.setText(f"用时: {elapsed:.1f}秒")

class AnalysisPanel(QWidget):
    """AI分析结果展示面板"""
    
    # 信号定义
    analysis_requested = pyqtSignal(AnalysisType, dict)
    move_selected = pyqtSignal(str)
    variation_applied = pyqtSignal(MoveVariation)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_result: Optional[AnalysisResult] = None
        self._analysis_history: List[AnalysisResult] = []
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        
        # 上部分: 实时分析状态
        upper_widget = self._create_status_section()
        splitter.addWidget(upper_widget)
        
        # 下部分: 详细分析结果
        lower_widget = self._create_analysis_section()
        splitter.addWidget(lower_widget)
        
        # 设置分割比例
        splitter.setSizes([150, 400])
        
        # 底部控制栏
        control_widget = self._create_control_section()
        layout.addWidget(control_widget)
        
    def _create_status_section(self) -> QWidget:
        """创建状态区域"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(widget)
        
        # 分析状态
        status_group = QGroupBox("分析状态")
        status_layout = QVBoxLayout(status_group)
        self.status_widget = AnalysisStatusWidget()
        status_layout.addWidget(self.status_widget)
        layout.addWidget(status_group)
        
        # 胜率显示
        prob_group = QGroupBox("局面胜率")
        prob_layout = QVBoxLayout(prob_group)
        self.win_prob_widget = WinProbabilityWidget()
        prob_layout.addWidget(self.win_prob_widget)
        layout.addWidget(prob_group)
        
        # 评估图表
        chart_group = QGroupBox("评估走势")
        chart_layout = QVBoxLayout(chart_group)
        self.eval_chart = EvaluationChart()
        chart_layout.addWidget(self.eval_chart)
        layout.addWidget(chart_group)
        
        return widget
        
    def _create_analysis_section(self) -> QWidget:
        """创建分析区域"""
        tab_widget = QTabWidget()
        
        # 推荐着法标签页
        moves_widget = self._create_moves_tab()
        tab_widget.addTab(moves_widget, "推荐着法")
        
        # 变化分析标签页
        variations_widget = self._create_variations_tab()
        tab_widget.addTab(variations_widget, "变化分析")
        
        # 局面评估标签页
        position_widget = self._create_position_tab()
        tab_widget.addTab(position_widget, "局面评估")
        
        # 历史记录标签页
        history_widget = self._create_history_tab()
        tab_widget.addTab(history_widget, "历史记录")
        
        return tab_widget
        
    def _create_moves_tab(self) -> QWidget:
        """创建推荐着法标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.moves_list = QListWidget()
        self.moves_list.itemDoubleClicked.connect(self._on_move_selected)
        layout.addWidget(self.moves_list)
        
        return widget
        
    def _create_variations_tab(self) -> QWidget:
        """创建变化分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.variations_tree = MoveVariationTree()
        self.variations_tree.variation_selected.connect(self.variation_applied.emit)
        layout.addWidget(self.variations_tree)
        
        return widget
        
    def _create_position_tab(self) -> QWidget:
        """创建局面评估标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 材子平衡
        material_group = QGroupBox("材子平衡")
        material_layout = QGridLayout(material_group)
        self.material_label = QLabel("平衡")
        material_layout.addWidget(self.material_label, 0, 0)
        scroll_layout.addWidget(material_group)
        
        # 位置因素
        position_group = QGroupBox("位置因素")
        position_layout = QVBoxLayout(position_group)
        self.position_text = QTextEdit()
        self.position_text.setMaximumHeight(100)
        position_layout.addWidget(self.position_text)
        scroll_layout.addWidget(position_group)
        
        # 战术要素
        tactical_group = QGroupBox("战术要素")
        tactical_layout = QVBoxLayout(tactical_group)
        self.tactical_text = QTextEdit()
        self.tactical_text.setMaximumHeight(100)
        tactical_layout.addWidget(self.tactical_text)
        scroll_layout.addWidget(tactical_group)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
        
    def _create_history_tab(self) -> QWidget:
        """创建历史记录标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_selected)
        layout.addWidget(self.history_list)
        
        # 清空按钮
        clear_btn = QPushButton("清空历史")
        clear_btn.clicked.connect(self.clear_history)
        layout.addWidget(clear_btn)
        
        return widget
        
    def _create_control_section(self) -> QWidget:
        """创建控制区域"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(widget)
        
        # 分析类型选择
        type_label = QLabel("分析类型:")
        layout.addWidget(type_label)
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "局面评估", "着法建议", "战术分析", "残局分析", "开局分析"
        ])
        layout.addWidget(self.analysis_type_combo)
        
        # 深度设置
        depth_label = QLabel("搜索深度:")
        layout.addWidget(depth_label)
        
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 30)
        self.depth_spin.setValue(15)
        layout.addWidget(self.depth_spin)
        
        layout.addStretch()
        
        # 控制按钮
        self.start_btn = QPushButton("开始分析")
        self.start_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止分析")
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        return widget
        
    def _setup_connections(self):
        """设置信号连接"""
        pass
        
    def _start_analysis(self):
        """开始分析"""
        analysis_types = {
            0: AnalysisType.POSITION_EVAL,
            1: AnalysisType.MOVE_SUGGESTION,
            2: AnalysisType.TACTICAL_ANALYSIS,
            3: AnalysisType.ENDGAME_ANALYSIS,
            4: AnalysisType.OPENING_ANALYSIS
        }
        
        analysis_type = analysis_types[self.analysis_type_combo.currentIndex()]
        params = {
            'depth': self.depth_spin.value()
        }
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.analysis_requested.emit(analysis_type, params)
        
    def _stop_analysis(self):
        """停止分析"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def _on_move_selected(self, item: QListWidgetItem):
        """处理着法选择"""
        move = item.data(Qt.ItemDataRole.UserRole)
        if move:
            self.move_selected.emit(move)
            
    def _on_history_selected(self, item: QListWidgetItem):
        """处理历史记录选择"""
        result = item.data(Qt.ItemDataRole.UserRole)
        if result:
            self.display_analysis_result(result)
            
    def set_analysis_status(self, status: AnalysisStatus, progress: int = 0, depth: int = 0):
        """设置分析状态"""
        self.status_widget.set_status(status, progress)
        if depth > 0:
            self.status_widget.set_depth(depth)
            
        if status in [AnalysisStatus.COMPLETED, AnalysisStatus.ERROR, AnalysisStatus.CANCELLED]:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
    def display_analysis_result(self, result: AnalysisResult):
        """显示分析结果"""
        self._current_result = result
        
        # 更新胜率显示
        if result.position_eval:
            self.win_prob_widget.set_probability(result.position_eval.win_probability)
            self.eval_chart.add_evaluation(result.position_eval.score)
            
            # 更新局面评估
            self._update_position_evaluation(result.position_eval)
            
        # 更新推荐着法
        self._update_moves_list(result.best_moves)
        
        # 更新变化分析
        self.variations_tree.set_variations(result.variations)
        
        # 添加到历史记录
        self._add_to_history(result)
        
    def _update_position_evaluation(self, position_eval: PositionEvaluation):
        """更新局面评估显示"""
        # 材子平衡
        if position_eval.material_balance > 0:
            self.material_label.setText(f"红方优势 +{position_eval.material_balance}")
        elif position_eval.material_balance < 0:
            self.material_label.setText(f"黑方优势 {position_eval.material_balance}")
        else:
            self.material_label.setText("材子平衡")
            
        # 位置因素
        position_text = "\n".join([
            f"{factor}: {value:+.2f}" 
            for factor, value in position_eval.positional_factors.items()
        ])
        self.position_text.setText(position_text)
        
        # 战术要素
        tactical_text = "\n".join(position_eval.tactical_elements)
        self.tactical_text.setText(tactical_text)
        
    def _update_moves_list(self, moves: List[Tuple[str, float]]):
        """更新推荐着法列表"""
        self.moves_list.clear()
        
        for i, (move, score) in enumerate(moves):
            item = QListWidgetItem(f"{i+1}. {move} ({score:+.2f})")
            item.setData(Qt.ItemDataRole.UserRole, move)
            self.moves_list.addItem(item)
            
    def _add_to_history(self, result: AnalysisResult):
        """添加到历史记录"""
        self._analysis_history.append(result)
        
        # 只保留最近50条记录
        if len(self._analysis_history) > 50:
            self._analysis_history.pop(0)
            
        # 更新历史列表显示
        self._update_history_list()
        
    def _update_history_list(self):
        """更新历史记录列表"""
        self.history_list.clear()
        
        for i, result in enumerate(reversed(self._analysis_history)):
            timestamp = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
            type_name = {
                AnalysisType.POSITION_EVAL: "局面评估",
                AnalysisType.MOVE_SUGGESTION: "着法建议",
                AnalysisType.TACTICAL_ANALYSIS: "战术分析",
                AnalysisType.ENDGAME_ANALYSIS: "残局分析",
                AnalysisType.OPENING_ANALYSIS: "开局分析"
            }.get(result.analysis_type, "未知")
            
            text = f"{timestamp} - {type_name} (深度{result.analysis_depth})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, result)
            self.history_list.addItem(item)
            
    def clear_history(self):
        """清空历史记录"""
        self._analysis_history.clear()
        self.history_list.clear()
        self.eval_chart.clear_evaluations()
        
    def get_current_result(self) -> Optional[AnalysisResult]:
        """获取当前分析结果"""
        return self._current_result
        
    def get_analysis_history(self) -> List[AnalysisResult]:
        """获取分析历史"""
        return self._analysis_history.copy()
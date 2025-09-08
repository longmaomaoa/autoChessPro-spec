#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋智能对弈助手

这是一个基于计算机视觉和人工智能的中国象棋分析助手，
提供实时棋局识别、AI走法建议和胜率分析功能。
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from chess_ai.core import *
from chess_ai.config import ConfigManager

# 导出主要类和函数
__all__ = [
    "__version__",
    "ConfigManager",
]
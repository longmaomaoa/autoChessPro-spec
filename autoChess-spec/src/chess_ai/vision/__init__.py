#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算机视觉模块 - 屏幕捕获和棋局识别
"""

from chess_ai.vision.screen_capture import ScreenCaptureModule
from chess_ai.vision.board_detector import ChessBoardDetector
from chess_ai.vision.piece_classifier import ChessPieceClassifier
from chess_ai.vision.board_recognition import BoardRecognitionModule

__all__ = [
    "ScreenCaptureModule",
    "ChessBoardDetector", 
    "ChessPieceClassifier",
    "BoardRecognitionModule",
]
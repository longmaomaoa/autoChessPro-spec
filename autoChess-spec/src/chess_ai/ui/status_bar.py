#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态栏占位符
"""

from PyQt6.QtWidgets import QStatusBar

class StatusBar(QStatusBar):
    """状态栏占位符"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理器占位符
"""

class DataManager:
    """数据管理器占位符"""
    
    def __init__(self, config):
        self.config = config
    
    def initialize(self) -> bool:
        return True
    
    def clear_current_game(self) -> None:
        pass
    
    def save_board_state(self, board_state) -> None:
        pass
    
    def cleanup(self) -> None:
        pass
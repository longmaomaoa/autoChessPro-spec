#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能管理器占位符
"""

class PerformanceManager:
    """性能管理器占位符"""
    
    def __init__(self):
        pass
    
    def get_performance_stats(self) -> dict:
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'fps': 0.0
        }
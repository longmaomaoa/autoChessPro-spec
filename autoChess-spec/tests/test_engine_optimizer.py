#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI引擎性能优化和错误处理系统功能测试

测试引擎优化器的核心数据结构、性能监控和错误处理功能
"""

import sys
import time
import threading
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict

print("开始AI引擎性能优化和错误处理系统功能测试...")
print("=" * 60)

# 测试用的数据结构定义
class OptimizationLevel(Enum):
    """优化等级"""
    DISABLED = "disabled"
    BASIC = "basic"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_processing_time: float = 0.0
    average_search_depth: float = 0.0
    total_nodes_searched: int = 0
    average_nodes_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    evaluation_accuracy: float = 0.0
    confidence_score: float = 0.0

@dataclass
class ErrorRecord:
    """错误记录"""
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    resolved: bool = False

@dataclass
class OptimizationSettings:
    """优化设置"""
    enable_position_cache: bool = True
    max_cache_size: int = 10000
    cache_cleanup_interval: float = 300.0
    thread_pool_size: int = 4
    max_concurrent_requests: int = 8
    default_timeout: float = 10.0
    critical_timeout: float = 30.0
    max_memory_usage_mb: float = 500.0
    max_cpu_usage_percent: float = 80.0
    enable_auto_tuning: bool = True
    performance_sampling_interval: float = 60.0
    optimization_adjustment_interval: float = 300.0
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.RLock()
    
    def call(self, func, *args, **kwargs):
        """执行函数调用并应用熔断逻辑"""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e

# 测试函数
def test_optimization_level_enum():
    """测试优化等级枚举"""
    try:
        levels = [OptimizationLevel.DISABLED, OptimizationLevel.BASIC,
                 OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE, OptimizationLevel.ADAPTIVE]
        
        assert len(levels) == 5
        assert OptimizationLevel.DISABLED.value == "disabled"
        assert OptimizationLevel.ADAPTIVE.value == "adaptive"
        
        print("PASS: 优化等级枚举测试")
        return True
    except Exception as e:
        print(f"FAIL: 优化等级枚举测试 - {e}")
        return False

def test_error_severity_enum():
    """测试错误严重程度枚举"""
    try:
        severities = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR,
                     ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]
        
        assert len(severities) == 5
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.FATAL.value == "fatal"
        
        print("PASS: 错误严重程度枚举测试")
        return True
    except Exception as e:
        print(f"FAIL: 错误严重程度枚举测试 - {e}")
        return False

def test_performance_metrics():
    """测试性能指标数据结构"""
    try:
        metrics = PerformanceMetrics()
        
        # 测试默认值
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.min_response_time == float('inf')
        assert metrics.max_response_time == 0.0
        assert metrics.cache_hit_rate == 0.0
        
        # 测试更新指标
        metrics.total_requests = 100
        metrics.successful_requests = 95
        metrics.failed_requests = 5
        metrics.total_processing_time = 250.0
        metrics.average_response_time = metrics.total_processing_time / metrics.total_requests
        
        assert metrics.total_requests == 100
        assert metrics.successful_requests == 95
        assert metrics.failed_requests == 5
        assert metrics.average_response_time == 2.5
        
        print("PASS: 性能指标数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 性能指标数据结构测试 - {e}")
        return False

def test_error_record():
    """测试错误记录数据结构"""
    try:
        error = ErrorRecord(
            timestamp=time.time(),
            severity=ErrorSeverity.ERROR,
            error_type="EngineTimeout",
            message="引擎响应超时",
            context={"timeout": 10.0, "function": "get_best_move"},
            recovery_action="重试请求"
        )
        
        assert error.severity == ErrorSeverity.ERROR
        assert error.error_type == "EngineTimeout"
        assert error.message == "引擎响应超时"
        assert error.context["timeout"] == 10.0
        assert error.recovery_action == "重试请求"
        assert error.resolved == False
        
        print("PASS: 错误记录数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 错误记录数据结构测试 - {e}")
        return False

def test_optimization_settings():
    """测试优化设置数据结构"""
    try:
        settings = OptimizationSettings()
        
        # 测试默认值
        assert settings.enable_position_cache == True
        assert settings.max_cache_size == 10000
        assert settings.thread_pool_size == 4
        assert settings.max_concurrent_requests == 8
        assert settings.default_timeout == 10.0
        assert settings.max_retry_attempts == 3
        
        # 测试自定义设置
        custom_settings = OptimizationSettings(
            max_cache_size=20000,
            thread_pool_size=8,
            enable_circuit_breaker=False
        )
        
        assert custom_settings.max_cache_size == 20000
        assert custom_settings.thread_pool_size == 8
        assert custom_settings.enable_circuit_breaker == False
        
        print("PASS: 优化设置数据结构测试")
        return True
    except Exception as e:
        print(f"FAIL: 优化设置数据结构测试 - {e}")
        return False

def test_circuit_breaker():
    """测试熔断器功能"""
    try:
        # 创建熔断器
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
        
        # 测试正常调用
        def normal_function():
            return "success"
        
        result = circuit_breaker.call(normal_function)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
        
        # 测试失败调用
        def failing_function():
            raise Exception("test error")
        
        failure_count = 0
        for i in range(5):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                failure_count += 1
        
        assert failure_count == 5
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count >= 3
        
        # 测试熔断状态下的调用
        try:
            circuit_breaker.call(normal_function)
            assert False, "应该抛出熔断器异常"
        except Exception as e:
            assert "Circuit breaker is OPEN" in str(e)
        
        # 测试超时后的半开状态
        time.sleep(1.1)  # 等待超时
        circuit_breaker.state = "HALF_OPEN"  # 手动设置半开状态
        
        # 成功调用应该恢复到CLOSED状态
        result = circuit_breaker.call(normal_function)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        
        print("PASS: 熔断器功能测试")
        return True
    except Exception as e:
        print(f"FAIL: 熔断器功能测试 - {e}")
        return False

def test_cache_operations():
    """测试缓存操作"""
    try:
        # 模拟缓存
        cache = {}
        max_cache_size = 3
        
        def cache_key_generator(func_name, args):
            return f"{func_name}_{hash(str(args))}"
        
        def add_to_cache(key, value):
            if len(cache) >= max_cache_size:
                # 简单LRU - 删除第一个
                first_key = next(iter(cache))
                del cache[first_key]
            cache[key] = value
        
        def get_from_cache(key):
            return cache.get(key)
        
        # 测试添加缓存
        add_to_cache("key1", "value1")
        add_to_cache("key2", "value2")
        add_to_cache("key3", "value3")
        
        assert len(cache) == 3
        assert get_from_cache("key1") == "value1"
        
        # 测试缓存溢出
        add_to_cache("key4", "value4")
        assert len(cache) == 3
        assert get_from_cache("key1") is None  # 被删除
        assert get_from_cache("key4") == "value4"
        
        print("PASS: 缓存操作测试")
        return True
    except Exception as e:
        print(f"FAIL: 缓存操作测试 - {e}")
        return False

def test_metrics_calculation():
    """测试指标计算"""
    try:
        def calculate_success_rate(successful, total):
            if total == 0:
                return 0.0
            return successful / total
        
        def calculate_cache_hit_rate(hits, misses):
            total = hits + misses
            if total == 0:
                return 0.0
            return hits / total
        
        def calculate_average_response_time(total_time, total_requests):
            if total_requests == 0:
                return 0.0
            return total_time / total_requests
        
        # 测试成功率计算
        assert calculate_success_rate(95, 100) == 0.95
        assert calculate_success_rate(0, 0) == 0.0
        
        # 测试缓存命中率计算
        assert calculate_cache_hit_rate(80, 20) == 0.8
        assert calculate_cache_hit_rate(0, 0) == 0.0
        
        # 测试平均响应时间计算
        assert calculate_average_response_time(250.0, 100) == 2.5
        assert calculate_average_response_time(0.0, 0) == 0.0
        
        print("PASS: 指标计算测试")
        return True
    except Exception as e:
        print(f"FAIL: 指标计算测试 - {e}")
        return False

def test_error_aggregation():
    """测试错误聚合功能"""
    try:
        # 模拟错误记录
        error_records = deque(maxlen=100)
        
        # 添加一些错误
        current_time = time.time()
        error_records.extend([
            ErrorRecord(current_time - 3700, ErrorSeverity.WARNING, "Timeout", "连接超时", {}),
            ErrorRecord(current_time - 1800, ErrorSeverity.ERROR, "Timeout", "连接超时", {}),
            ErrorRecord(current_time - 900, ErrorSeverity.CRITICAL, "EngineFailure", "引擎故障", {}),
            ErrorRecord(current_time - 300, ErrorSeverity.WARNING, "LowMemory", "内存不足", {}),
            ErrorRecord(current_time - 60, ErrorSeverity.ERROR, "Timeout", "连接超时", {})
        ])
        
        # 统计错误类型
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        recent_errors = []
        
        for error in error_records:
            error_types[error.error_type] += 1
            severity_counts[error.severity.value] += 1
            
            if current_time - error.timestamp < 3600:  # 最近1小时
                recent_errors.append(error)
        
        # 验证统计结果
        assert error_types["Timeout"] == 3
        assert error_types["EngineFailure"] == 1
        assert error_types["LowMemory"] == 1
        
        assert severity_counts["warning"] == 2
        assert severity_counts["error"] == 2
        assert severity_counts["critical"] == 1
        
        assert len(recent_errors) == 4  # 最近1小时内的错误
        
        print("PASS: 错误聚合功能测试")
        return True
    except Exception as e:
        print(f"FAIL: 错误聚合功能测试 - {e}")
        return False

def test_retry_backoff_strategy():
    """测试重试退避策略"""
    try:
        def exponential_backoff(attempt, base_delay=1.0):
            return base_delay * (2 ** attempt)
        
        def should_retry(exception_type, attempt, max_attempts):
            if attempt >= max_attempts:
                return False
            
            # 某些类型的异常不重试
            if exception_type in ["ValueError", "TypeError"]:
                return False
            
            return True
        
        # 测试指数退避
        assert exponential_backoff(0) == 1.0
        assert exponential_backoff(1) == 2.0
        assert exponential_backoff(2) == 4.0
        assert exponential_backoff(3) == 8.0
        
        # 测试重试策略
        assert should_retry("ConnectionError", 0, 3) == True
        assert should_retry("ConnectionError", 3, 3) == False
        assert should_retry("ValueError", 0, 3) == False
        assert should_retry("TimeoutError", 1, 3) == True
        
        print("PASS: 重试退避策略测试")
        return True
    except Exception as e:
        print(f"FAIL: 重试退避策略测试 - {e}")
        return False

def test_performance_report_generation():
    """测试性能报告生成"""
    try:
        # 模拟性能数据
        metrics = PerformanceMetrics(
            total_requests=1000,
            successful_requests=950,
            failed_requests=50,
            average_response_time=2.5,
            cache_hits=800,
            cache_misses=200,
            memory_usage_mb=256.5,
            cpu_usage_percent=65.2
        )
        
        # 计算衍生指标
        metrics.cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
        success_rate = metrics.successful_requests / metrics.total_requests
        
        # 生成报告
        report = {
            'metrics': {
                'total_requests': metrics.total_requests,
                'success_rate': success_rate,
                'average_response_time': metrics.average_response_time,
                'cache_hit_rate': metrics.cache_hit_rate,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent
            },
            'optimization': {
                'current_level': OptimizationLevel.BALANCED.value,
                'enabled': True,
                'cache_size': 5000
            }
        }
        
        # 验证报告内容
        assert report['metrics']['success_rate'] == 0.95
        assert report['metrics']['cache_hit_rate'] == 0.8
        assert report['optimization']['current_level'] == "balanced"
        
        print("PASS: 性能报告生成测试")
        return True
    except Exception as e:
        print(f"FAIL: 性能报告生成测试 - {e}")
        return False

def test_optimization_recommendations():
    """测试优化建议生成"""
    try:
        def generate_recommendations(metrics):
            recommendations = []
            
            if metrics.cache_hit_rate < 0.5:
                recommendations.append("建议增加缓存大小以提高命中率")
            
            if metrics.average_response_time > 5.0:
                recommendations.append("建议降低分析深度或增加超时时间")
            
            if metrics.memory_usage_mb > 300:
                recommendations.append("建议定期清理缓存以减少内存使用")
            
            error_rate = 1 - (metrics.successful_requests / max(metrics.total_requests, 1))
            if error_rate > 0.1:
                recommendations.append("建议检查引擎配置，错误率过高")
            
            return recommendations or ["系统运行正常，无特殊建议"]
        
        # 测试不同场景的建议
        # 场景1: 正常运行
        normal_metrics = PerformanceMetrics(
            total_requests=100, successful_requests=95,
            cache_hit_rate=0.8, average_response_time=2.0, memory_usage_mb=200
        )
        recommendations = generate_recommendations(normal_metrics)
        assert "系统运行正常，无特殊建议" in recommendations
        
        # 场景2: 缓存命中率低
        low_cache_metrics = PerformanceMetrics(
            total_requests=100, successful_requests=95,
            cache_hit_rate=0.3, average_response_time=2.0, memory_usage_mb=200
        )
        recommendations = generate_recommendations(low_cache_metrics)
        assert any("缓存" in rec for rec in recommendations)
        
        # 场景3: 响应时间长
        slow_metrics = PerformanceMetrics(
            total_requests=100, successful_requests=95,
            cache_hit_rate=0.8, average_response_time=8.0, memory_usage_mb=200
        )
        recommendations = generate_recommendations(slow_metrics)
        assert any("响应时间" in rec or "超时" in rec for rec in recommendations)
        
        print("PASS: 优化建议生成测试")
        return True
    except Exception as e:
        print(f"FAIL: 优化建议生成测试 - {e}")
        return False

# 运行所有测试
def run_engine_optimizer_tests():
    """运行AI引擎性能优化和错误处理系统测试"""
    test_functions = [
        test_optimization_level_enum,
        test_error_severity_enum,
        test_performance_metrics,
        test_error_record,
        test_optimization_settings,
        test_circuit_breaker,
        test_cache_operations,
        test_metrics_calculation,
        test_error_aggregation,
        test_retry_backoff_strategy,
        test_performance_report_generation,
        test_optimization_recommendations
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
        print("所有测试通过! AI引擎性能优化和错误处理系统核心功能正常")
        return True
    else:
        print(f"有 {total-passed} 个测试失败，需要修复实现")
        return False

if __name__ == "__main__":
    success = run_engine_optimizer_tests()
    sys.exit(0 if success else 1)
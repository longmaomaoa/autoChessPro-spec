#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI引擎性能优化和错误处理系统

提供引擎性能监控、自动调优、故障恢复和错误处理功能
"""

import asyncio
import logging
import threading
import time
import traceback
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import psutil
import gc

from chess_ai.core.board_state import BoardState
from chess_ai.core.config_manager import ConfigManager
from chess_ai.ai_engine.pikafish_engine import PikafishEngine, EngineAnalysis, EngineState, PikafishEngineError


class OptimizationLevel(Enum):
    """优化等级"""
    DISABLED = "disabled"     # 禁用优化
    BASIC = "basic"          # 基础优化
    BALANCED = "balanced"    # 平衡优化
    AGGRESSIVE = "aggressive" # 激进优化
    ADAPTIVE = "adaptive"    # 自适应优化


class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    ERROR = "error"         # 错误
    CRITICAL = "critical"   # 致命错误
    FATAL = "fatal"         # 系统故障


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 基础指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # 时间指标
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_processing_time: float = 0.0
    
    # 深度和节点指标
    average_search_depth: float = 0.0
    total_nodes_searched: int = 0
    average_nodes_per_second: float = 0.0
    
    # 资源使用指标
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # 缓存指标
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    # 质量指标
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
    # 缓存设置
    enable_position_cache: bool = True
    max_cache_size: int = 10000
    cache_cleanup_interval: float = 300.0  # 5分钟
    
    # 线程池设置
    thread_pool_size: int = 4
    max_concurrent_requests: int = 8
    
    # 超时设置
    default_timeout: float = 10.0
    critical_timeout: float = 30.0
    
    # 资源限制
    max_memory_usage_mb: float = 500.0
    max_cpu_usage_percent: float = 80.0
    
    # 自动调优设置
    enable_auto_tuning: bool = True
    performance_sampling_interval: float = 60.0  # 1分钟
    optimization_adjustment_interval: float = 300.0  # 5分钟
    
    # 错误处理设置
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


class EngineOptimizer:
    """AI引擎性能优化器
    
    负责监控引擎性能，自动调优参数，处理错误和故障恢复
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.optimization_settings = OptimizationSettings()
        
        # 性能监控
        self.metrics = PerformanceMetrics()
        self.metrics_history = deque(maxlen=1000)  # 保留最近1000次记录
        
        # 错误处理
        self.error_records = deque(maxlen=1000)
        self.circuit_breakers = {}
        
        # 优化状态
        self.current_optimization_level = OptimizationLevel.BALANCED
        self.optimization_enabled = True
        
        # 缓存系统
        self.position_cache = {}
        self.analysis_cache = {}
        self.last_cleanup_time = time.time()
        
        # 并发控制
        self.executor = ThreadPoolExecutor(
            max_workers=self.optimization_settings.thread_pool_size,
            thread_name_prefix="Engine-Optimizer"
        )
        self.active_requests = 0
        self.request_queue = queue.Queue()
        
        # 监控线程
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # 锁
        self.metrics_lock = threading.RLock()
        self.cache_lock = threading.RLock()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化优化器"""
        try:
            # 启动性能监控
            self._start_monitoring()
            
            # 初始化熔断器
            self.circuit_breakers['engine'] = CircuitBreaker(
                self.optimization_settings.circuit_breaker_threshold,
                self.optimization_settings.circuit_breaker_timeout
            )
            
            self.is_initialized = True
            self.logger.info("AI引擎优化器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"优化器初始化失败: {e}")
            return False
    
    def optimize_engine_call(self, engine: PikafishEngine, 
                           func_name: str, *args, **kwargs) -> Any:
        """优化引擎调用"""
        if not self.is_initialized:
            raise RuntimeError("优化器未初始化")
        
        start_time = time.time()
        request_id = f"{func_name}_{id(args)}_{start_time}"
        
        try:
            # 检查并发限制
            if self.active_requests >= self.optimization_settings.max_concurrent_requests:
                raise Exception("达到最大并发请求数限制")
            
            self.active_requests += 1
            
            # 检查缓存
            if func_name == 'get_best_move' and self.optimization_settings.enable_position_cache:
                cache_key = self._generate_cache_key(func_name, args, kwargs)
                cached_result = self._get_cached_result(cache_key)
                if cached_result is not None:
                    self._update_cache_metrics(True)
                    return cached_result
                self._update_cache_metrics(False)
            
            # 应用熔断器
            circuit_breaker = self.circuit_breakers.get('engine')
            if circuit_breaker:
                result = circuit_breaker.call(getattr(engine, func_name), *args, **kwargs)
            else:
                result = getattr(engine, func_name)(*args, **kwargs)
            
            # 缓存结果
            if func_name == 'get_best_move' and self.optimization_settings.enable_position_cache:
                self._cache_result(cache_key, result)
            
            # 更新性能指标
            self._update_metrics(True, time.time() - start_time, func_name, args, kwargs)
            
            return result
            
        except Exception as e:
            self._update_metrics(False, time.time() - start_time, func_name, args, kwargs)
            self._record_error(ErrorSeverity.ERROR, str(type(e).__name__), str(e), {
                'function': func_name,
                'args': str(args),
                'kwargs': str(kwargs)
            })
            
            # 尝试恢复
            if self._should_retry(e):
                return self._retry_with_backoff(engine, func_name, args, kwargs)
            
            raise e
        
        finally:
            self.active_requests -= 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.metrics_lock:
            # 计算当前资源使用情况
            self._update_resource_metrics()
            
            return {
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'success_rate': self._calculate_success_rate(),
                    'average_response_time': self.metrics.average_response_time,
                    'cache_hit_rate': self.metrics.cache_hit_rate,
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'cpu_usage_percent': self.metrics.cpu_usage_percent
                },
                'optimization': {
                    'current_level': self.current_optimization_level.value,
                    'enabled': self.optimization_enabled,
                    'cache_size': len(self.position_cache)
                },
                'errors': {
                    'recent_errors': len([e for e in self.error_records 
                                        if time.time() - e.timestamp < 3600]),  # 最近1小时
                    'critical_errors': len([e for e in self.error_records 
                                          if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]])
                },
                'recommendations': self._generate_recommendations()
            }
    
    def adjust_optimization_level(self, level: OptimizationLevel) -> None:
        """调整优化等级"""
        self.current_optimization_level = level
        
        # 根据等级调整设置
        if level == OptimizationLevel.BASIC:
            self.optimization_settings.max_cache_size = 5000
            self.optimization_settings.thread_pool_size = 2
        elif level == OptimizationLevel.BALANCED:
            self.optimization_settings.max_cache_size = 10000
            self.optimization_settings.thread_pool_size = 4
        elif level == OptimizationLevel.AGGRESSIVE:
            self.optimization_settings.max_cache_size = 20000
            self.optimization_settings.thread_pool_size = 8
        elif level == OptimizationLevel.ADAPTIVE:
            self._enable_adaptive_optimization()
        
        self.logger.info(f"优化等级调整为: {level.value}")
    
    def clear_cache(self) -> None:
        """清理缓存"""
        with self.cache_lock:
            self.position_cache.clear()
            self.analysis_cache.clear()
            gc.collect()  # 强制垃圾回收
            
        self.logger.info("缓存已清理")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        recent_errors = []
        
        current_time = time.time()
        for error in self.error_records:
            error_types[error.error_type] += 1
            severity_counts[error.severity.value] += 1
            
            if current_time - error.timestamp < 3600:  # 最近1小时
                recent_errors.append({
                    'timestamp': error.timestamp,
                    'severity': error.severity.value,
                    'type': error.error_type,
                    'message': error.message[:100]  # 截断长消息
                })
        
        return {
            'error_types': dict(error_types),
            'severity_counts': dict(severity_counts),
            'recent_errors': recent_errors[-10:],  # 最近10个错误
            'total_errors': len(self.error_records)
        }
    
    def _start_monitoring(self) -> None:
        """启动性能监控"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="Engine-Monitor",
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.monitoring_active:
            try:
                # 更新资源指标
                self._update_resource_metrics()
                
                # 清理过期缓存
                if time.time() - self.last_cleanup_time > self.optimization_settings.cache_cleanup_interval:
                    self._cleanup_cache()
                
                # 自适应优化
                if (self.current_optimization_level == OptimizationLevel.ADAPTIVE and 
                    self.optimization_settings.enable_auto_tuning):
                    self._perform_auto_tuning()
                
                # 检查系统健康状况
                self._check_system_health()
                
                time.sleep(self.optimization_settings.performance_sampling_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 简化实现 - 实际需要更复杂的键生成逻辑
        key_parts = [func_name]
        
        if args:
            if hasattr(args[0], 'fen'):  # BoardState对象
                key_parts.append(args[0].fen)
            else:
                key_parts.append(str(args))
        
        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))
        
        return "_".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        with self.cache_lock:
            return self.position_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """缓存结果"""
        with self.cache_lock:
            # 检查缓存大小限制
            if len(self.position_cache) >= self.optimization_settings.max_cache_size:
                # 简单的LRU逻辑 - 删除一些旧条目
                keys_to_remove = list(self.position_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.position_cache[key]
            
            self.position_cache[cache_key] = result
    
    def _update_metrics(self, success: bool, response_time: float, 
                       func_name: str, args: tuple, kwargs: dict) -> None:
        """更新性能指标"""
        with self.metrics_lock:
            self.metrics.total_requests += 1
            
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
            
            # 更新响应时间指标
            self.metrics.total_processing_time += response_time
            self.metrics.average_response_time = (
                self.metrics.total_processing_time / self.metrics.total_requests
            )
            self.metrics.min_response_time = min(self.metrics.min_response_time, response_time)
            self.metrics.max_response_time = max(self.metrics.max_response_time, response_time)
            
            # 记录历史数据
            self.metrics_history.append({
                'timestamp': time.time(),
                'success': success,
                'response_time': response_time,
                'function': func_name
            })
    
    def _update_cache_metrics(self, cache_hit: bool) -> None:
        """更新缓存指标"""
        with self.metrics_lock:
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            total_cache_requests = self.metrics.cache_hits + self.metrics.cache_misses
            if total_cache_requests > 0:
                self.metrics.cache_hit_rate = self.metrics.cache_hits / total_cache_requests
    
    def _update_resource_metrics(self) -> None:
        """更新资源使用指标"""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
        except:
            pass  # 忽略获取系统信息的错误
    
    def _record_error(self, severity: ErrorSeverity, error_type: str, 
                     message: str, context: Dict[str, Any]) -> None:
        """记录错误"""
        error_record = ErrorRecord(
            timestamp=time.time(),
            severity=severity,
            error_type=error_type,
            message=message,
            context=context,
            stack_trace=traceback.format_exc() if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL] else None
        )
        
        self.error_records.append(error_record)
        
        # 日志记录
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        self.logger.log(log_level, f"{error_type}: {message}")
    
    def _should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        # 对于某些类型的错误，不进行重试
        if isinstance(exception, (ValueError, TypeError)):
            return False
        
        # 对于网络或引擎相关错误，可以重试
        return True
    
    def _retry_with_backoff(self, engine: PikafishEngine, 
                           func_name: str, args: tuple, kwargs: dict) -> Any:
        """带退避的重试"""
        max_attempts = self.optimization_settings.max_retry_attempts
        delay = self.optimization_settings.retry_delay_seconds
        
        for attempt in range(max_attempts):
            try:
                time.sleep(delay * (2 ** attempt))  # 指数退避
                return getattr(engine, func_name)(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:  # 最后一次尝试
                    raise e
                
                self.logger.warning(f"重试 {attempt + 1}/{max_attempts} 失败: {e}")
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if self.metrics.total_requests == 0:
            return 0.0
        return self.metrics.successful_requests / self.metrics.total_requests
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        with self.cache_lock:
            # 简单的清理逻辑 - 如果缓存过大就清理一部分
            if len(self.position_cache) > self.optimization_settings.max_cache_size * 0.8:
                keys_to_remove = list(self.position_cache.keys())[:1000]
                for key in keys_to_remove:
                    del self.position_cache[key]
                
                self.logger.info(f"清理了 {len(keys_to_remove)} 个缓存条目")
            
            self.last_cleanup_time = time.time()
    
    def _perform_auto_tuning(self) -> None:
        """执行自动调优"""
        # 基于性能指标调整参数
        if self.metrics.average_response_time > 10.0:  # 响应时间过长
            self.optimization_settings.default_timeout *= 1.2
            self.logger.info("响应时间过长，增加超时时间")
        
        if self.metrics.cache_hit_rate < 0.3:  # 缓存命中率过低
            self.optimization_settings.max_cache_size = min(
                self.optimization_settings.max_cache_size * 1.5,
                50000  # 最大限制
            )
            self.logger.info("缓存命中率低，增加缓存大小")
    
    def _enable_adaptive_optimization(self) -> None:
        """启用自适应优化"""
        self.optimization_settings.enable_auto_tuning = True
        self.logger.info("启用自适应优化")
    
    def _check_system_health(self) -> None:
        """检查系统健康状况"""
        # 检查内存使用
        if self.metrics.memory_usage_mb > self.optimization_settings.max_memory_usage_mb:
            self._record_error(
                ErrorSeverity.WARNING,
                "HighMemoryUsage",
                f"内存使用过高: {self.metrics.memory_usage_mb:.2f}MB",
                {'threshold': self.optimization_settings.max_memory_usage_mb}
            )
            
            # 自动清理缓存
            self.clear_cache()
        
        # 检查错误率
        success_rate = self._calculate_success_rate()
        if success_rate < 0.8:  # 成功率低于80%
            self._record_error(
                ErrorSeverity.WARNING,
                "LowSuccessRate", 
                f"成功率过低: {success_rate:.2f}",
                {'success_rate': success_rate}
            )
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.metrics.cache_hit_rate < 0.5:
            recommendations.append("建议增加缓存大小以提高命中率")
        
        if self.metrics.average_response_time > 5.0:
            recommendations.append("建议降低分析深度或增加超时时间")
        
        if self.metrics.memory_usage_mb > 300:
            recommendations.append("建议定期清理缓存以减少内存使用")
        
        error_rate = 1 - self._calculate_success_rate()
        if error_rate > 0.1:
            recommendations.append("建议检查引擎配置，错误率过高")
        
        return recommendations or ["系统运行正常，无特殊建议"]
    
    def cleanup(self) -> None:
        """清理资源"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.clear_cache()
        self.logger.info("引擎优化器已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
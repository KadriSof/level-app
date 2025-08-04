"""levelapp/utils.monitoring.py"""
import logging
import inspect
import threading
import time
import tracemalloc
from collections import defaultdict, deque

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Deque, Callable, Any, Union, ParamSpec, TypeVar, runtime_checkable, Protocol

from datetime import datetime
from threading import Lock, RLock
from functools import lru_cache, wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class MetricType(Enum):
    """Types of metrics that can be collected."""
    TIMING = "timing"
    MEMORY = "memory"
    API_CALL = "api_call"
    CACHE_HIT = "cache_hit"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class ExecutionMetrics:
    """Comprehensive metrics for a function execution."""
    function_name: str
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    memory_before: int | None = None
    memory_after: int | None = None
    memory_peak: int | None = None
    cache_hit: bool = False
    error: str | None = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class AggregatedStats:
    """Aggregated metrics for monitored functions."""
    total_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    memory_peak: int = 0
    last_called: datetime | None = None

    def update(self, metrics: ExecutionMetrics) -> None:
        """Update aggregated metrics with new execution metrics."""
        self.total_calls += 1
        self.last_called = datetime.now()

        if metrics.duration is not None:
            self.total_duration += metrics.duration
            self.min_duration = min(self.min_duration, metrics.duration)
            self.max_duration = max(self.max_duration, metrics.duration)

        if metrics.error:
            self.error_count += 1

        if metrics.cache_hit:
            self.cache_hits += 1

        if metrics.memory_peak:
            self.memory_peak = max(self.memory_peak, metrics.memory_peak)

    @property
    def average_duration(self) -> float:
        """Average execution duration."""
        return (self.total_duration / self.total_calls) if self.total_calls > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        return (self.cache_hits / self.total_calls * 100) if self.total_calls > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        return (self.error_count / self.total_calls * 100) if self.total_calls > 0 else 0.0


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for custom metrics collectors."""
    def collect_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics before function execution."""
        ...

    def collect_after(self, metadata: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Collect metrics after function execution."""
        ...


class MemoryTracker(MetricsCollector):
    """Memory usage metrics collector."""
    def __init__(self):
        self._tracking = False

    def collect_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self._tracking:
            tracemalloc.start()
            self._tracking = True

        current, peak = tracemalloc.get_traced_memory()
        return {"memory_before": current, "memory_peak": peak}

    def collect_after(self, metadata: Dict[str, Any], result: Any) -> Dict[str, Any]:
        if self._tracking:
            current, peak = tracemalloc.get_traced_memory()
            return {"memory_after": current, "memory_peak": peak}
        return {}


class APICallTracker(MetricsCollector):
    """API call metrics collector for LLM clients."""

    def __init__(self):
        self._api_calls = defaultdict(int)
        self._lock = threading.Lock()

    def collect_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"api_calls_before": dict(self._api_calls)}

    def collect_after(self, metadata: Dict[str, Any], result: Any) -> Dict[str, Any]:
        with self._lock:
            # Detect API calls by inspecting the call stack
            stack = inspect.stack()
            api_calls = 0
            for frame in stack:
                if 'clients' in frame.filename and any(
                        method in frame.function
                        for method in ['_build_payload', '_build_headers', 'chat', 'complete']
                ):
                    api_calls += 1

            if api_calls > 0:
                func_name = metadata.get('function_name', 'unknown')
                self._api_calls[func_name] += api_calls

        return {"api_calls_detected": api_calls, "total_api_calls": dict(self._api_calls)}


class FunctionMonitor:
    """Core function monitoring system."""

    def __init__(self, max_history: int = 1000):
        self._monitored_functions: Dict[str, Callable[..., Any]] = {}
        self._execution_history: Dict[str, Deque[ExecutionMetrics]] = defaultdict(lambda: deque(maxlen=max_history))
        self._aggregated_stats: Dict[str, AggregatedStats] = defaultdict(AggregatedStats)
        self._collectors: List[MetricsCollector] = []
        self._lock = RLock()
        self.add_collector(MemoryTracker())
        self.add_collector(APICallTracker())

    def add_collector(self, collector: MetricsCollector) -> None:
        """
        Add a custom metrics collector to the monitor.

        Args:
            collector: An instance of a class implementing MetricsCollector protocol.
        """
        if not isinstance(collector, MetricsCollector):
            raise TypeError("Collector must implement MetricsCollector protocol.")

        with self._lock:
            self._collectors.append(collector)

    def remove_collector(self, collector: MetricsCollector) -> None:
        """
        Remove a custom metrics collector from the monitor.

        Args:
            collector: An instance of a class implementing MetricsCollector protocol.
        """
        with self._lock:
            if collector in self._collectors:
                self._collectors.remove(collector)

    def _collect_metrics_before(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect metrics before function execution using registered collectors.
        """
        metrics = {}
        for collector in self._collectors:
            try:
                metrics.update(collector.collect_before(metadata=metadata))
            except Exception as e:
                logger.warning(f"Metrics collector failed: {e}")

        return metrics

    def _collect_metrics_after(self, metadata: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """
        Collect metrics after function execution using registered collectors.
        """
        metrics = {}
        for collector in self._collectors:
            try:
                metrics.update(collector.collect_after(metadata=metadata, result=result))
            except Exception as e:
                logger.warning(f"Metrics collector failed: {e}")

        return metrics

    @staticmethod
    def _apply_caching(func: Callable[P, T], maxsize: int | None) -> Callable[P, T]:
        """Apply LRU caching with cache hit tracking."""
        if maxsize is None:
            return func

        cached_func = lru_cache(maxsize=maxsize)(func)

        @wraps(cached_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_info_before = cached_func.cache_info()
            result = cached_func(*args, **kwargs)
            cache_info_after = cached_func.cache_info()

            if not hasattr(wrapper, '_cache_hit_info'):
                wrapper._cache_hit_info = threading.local()

            wrapper._cache_hit_info.is_hit = cache_info_after.hits > cache_info_before.hits

            return result

        # Copy cache methods to wrapper
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    def _wrap_execution(
            self,
            func: Callable[P, T],
            name: str,
            enable_timing: bool,
            track_memory: bool,
            metadata: Dict[str, Any] | None = None
    ) -> Callable[P, T]:
        """
        Wrap function execution with timing and error handling.

        Args:
            func: Function to be wrapped
            name: Unique identifier for the function
            enable_timing: Enable execution time logging
            track_memory: Enable memory tracking
            metadata: Optional metadata for the execution context

        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()

            # Initialize execution metadata
            exec_metadata = {
                'function_name': name,
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                **(metadata or {})
            }

            # Initialize execution metrics
            metrics = ExecutionMetrics(
                function_name=name,
                start_time=start_time,
                metadata=exec_metadata,
            )

            # Collect pre-execution metrics
            if track_memory or self._collectors:
                pre_metrics = self._collect_metrics_before(metadata=exec_metadata)
                metrics.memory_before = pre_metrics.get('memory_before')
                metrics.custom_metrics.update(pre_metrics)

            try:
                result = func(*args, **kwargs)

                # Check for cache hit
                if hasattr(func, 'cache_info') and hasattr(func, '_cache_hit_info'):
                    metrics.cache_hit = getattr(func._cache_hit_info, 'is_hit', False)

                # Collect post-execution metrics
                if track_memory or self._collectors:
                    post_metrics = self._collect_metrics_after(exec_metadata, result)
                    metrics.memory_after = post_metrics.get('memory_after')
                    metrics.memory_peak = post_metrics.get('memory_peak')
                    metrics.custom_metrics.update(post_metrics)

                return result

            except Exception as e:
                metrics.error = str(e)
                logger.error(f"Error in '{name}': {str(e)}", exc_info=True)
                raise

            finally:
                metrics.end_time = time.perf_counter()
                metrics.finalize()

                # store metrics
                with self._lock:
                    self._execution_history[name].append(metrics)
                    self._aggregated_stats[name].update(metrics)

                if enable_timing and metrics.duration:
                    log_message = f"Executed '{name}' in {metrics.duration:.4f}s"
                    if metrics.cache_hit:
                        log_message += " (cache hit)"
                    if metrics.memory_peak:
                        log_message += f" (memory peak: {metrics.memory_peak / 1024 / 1024:.2f} MB)"
                    logger.info(log_message)

        return wrapped

    def monitor(
            self,
            name: str,
            cached: bool = False,
            maxsize: int | None = 128,
            enable_timing: bool = True,
            track_memory: bool = True,
            metadata: Dict[str, Any] | None = None,
            collectors: List[MetricsCollector] | None = None
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Decorator factory for monitoring functions.

        Args:
            name: Unique identifier for the function
            cached: Enable LRU caching
            maxsize: Maximum cache size
            enable_timing: Record execution time
            track_memory: Track memory usage
            metadata: Optional metadata for the execution context
            collectors: Optional list of custom metrics collectors

        Returns:
            Callable[[Callable[P, T]], Callable[P, T]]: Decorator function
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            if collectors:
                for collector in collectors:
                    self.add_collector(collector)

            if cached:
                func = self._apply_caching(func=func, maxsize=maxsize)

            monitored_func = self._wrap_execution(
                func=func,
                name=name,
                enable_timing=enable_timing,
                track_memory=track_memory,
                metadata=metadata or {}
            )

            with self._lock:
                if name in self._monitored_functions:
                    raise ValueError(f"Function '{name}' is already registered.")

                self._monitored_functions[name] = monitored_func

            return monitored_func

        return decorator

    def list_monitored_functions(self) -> Dict[str, Callable[..., Any]]:
        """
        List all registered monitored functions.

        Returns:
            List[str]: Names of all registered functions
        """
        with self._lock:
            return dict(self._monitored_functions)

    def get_stats(self, name: str) -> Dict[str, Any] | None:
        """
        Get comprehensive statistics for a monitored function.

        Args:
            name (str): Name of the monitored function.

        Returns:
            Dict[str, Any] | None: Dictionary containing function statistics or None if not found.
        """
        with self._lock:
            if name not in self._monitored_functions:
                return None

            func = self._monitored_functions[name]
            stats = self._aggregated_stats[name]
            history = list(self._execution_history[name])

            return {
                'name': name,
                'total_calls': stats.total_calls,
                'avg_duration': stats.average_duration,
                'min_duration': stats.min_duration if stats.min_duration != float('inf') else 0,
                'max_duration': stats.max_duration,
                'error_rate': stats.error_rate,
                'cache_hit_rate': stats.cache_hit_rate if hasattr(func, 'cache_info') else None,
                'memory_peak_mb': stats.memory_peak / 1024 / 1024 if stats.memory_peak else 0,
                'last_called': stats.last_called.isoformat() if stats.last_called else None,
                'recent_executions': len(history),
                'is_cached': hasattr(func, 'cache_info'),
                'cache_info': func.cache_info() if hasattr(func, 'cache_info') else None
            }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all monitored functions."""
        with self._lock:
            return {
                name: self.get_stats(name)
                for name in self._monitored_functions.keys()
            }

    def get_execution_history(self, name: str, limit: int | None = None) -> List[ExecutionMetrics]:
        """Get execution history for a specific function."""
        with self._lock:
            if name not in self._execution_history:
                return []

            history = list(self._execution_history[name])
            return history[-limit:] if limit else history

    def clear_history(self, function_name: str | None = None) -> None:
        """Clear execution history."""
        with self._lock:
            if function_name:
                if function_name in self._execution_history:
                    self._execution_history[function_name].clear()
                if function_name in self._aggregated_stats:
                    self._aggregated_stats[function_name] = AggregatedStats()
            else:
                self._execution_history.clear()
                self._aggregated_stats.clear()

    def export_metrics(self, output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Export all metrics in various formats.

        Args:
            output_format (str): Format for exporting metrics ('dict' or 'json').

        Returns:
            Union[Dict[str, Any], str]: Exported metrics in the specified format.
        """
        with self._lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'functions': self.get_all_stats(),
                'total_executions': sum(
                    len(history) for history in self._execution_history.values()
                ),
                'collectors': [type(c).__name__ for c in self._collectors]
            }

        if output_format == 'dict':
            return data
        elif output_format == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {output_format}")


_global_monitor = FunctionMonitor()


# Global monitoring functions for backward compatibility.
def monitor(name: str, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to monitor function execution with global FunctionMonitor.

    Args:
        name: Unique identifier for the function
        **kwargs: Additional parameters for FunctionMonitor

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function
    """
    return _global_monitor.monitor(name=name, **kwargs)


def get_stats(name: str) -> Dict[str, Any] | None:
    """
    Get statistics for a monitored function.

    Args:
        name (str): Name of the monitored function.

    Returns:
        Dict[str, Any] | None: Function statistics or None if not found.
    """
    return _global_monitor.get_stats(name=name)


def list_monitored_functions() -> Dict[str, Callable[..., Any]]:
    """
    List all monitored functions.

    Returns:
        Dict[str, Callable[..., Any]]: Dictionary of monitored function names and their callable objects.
    """
    return _global_monitor.list_monitored_functions()


def clear_history(function_name: str | None = None) -> None:
    """
    Clear execution history.

    Args:
        function_name (str | None): Name of the function to clear history for.
    """
    return _global_monitor.clear_history(function_name)


def export_metrics(output_format: str = 'dict') -> Union[Dict[str, Any], str]:
    """
    Export all metrics.

    Args:
        output_format (str): Format for exporting metrics ('dict' or 'json').

    Returns:
        Union[Dict[str, Any], str]: Exported metrics in the specified format.
    """
    return _global_monitor.export_metrics(output_format)


# Convenience decorators
def monitor_with_cache(name: str, maxsize: int = 128, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Monitor with caching enabled.

    Args:
        name (str): Unique identifier for the function.
        maxsize (int): Maximum size of the cache.
        **kwargs: Additional parameters for FunctionMonitor.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function with caching enabled.
    """
    return monitor(name, cached=True, maxsize=maxsize, **kwargs)


def monitor_memory(name: str, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Monitor with memory tracking.

    Args:
        name (str): Unique identifier for the function.
        **kwargs: Additional parameters for FunctionMonitor.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function with memory
    """
    return monitor(name, track_memory=True, **kwargs)


def monitor_api_calls(name: str, **kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Monitor API calls (includes API call tracker by default).

    Args:
        name (str): Unique identifier for the function.
        **kwargs: Additional parameters for FunctionMonitor.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: Decorator function with API call tracking.
    """
    return monitor(name, track_memory=True, enable_timing=True, **kwargs)

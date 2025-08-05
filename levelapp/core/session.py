"""levelapp/core/session.py"""
import logging

from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any

from levelapp.utils.monitoring import FunctionMonitor, ExecutionMetrics


logger = logging.getLogger(__name__)


class EvaluationSession:
    """Context manager for LLM evaluation sessions with integrated monitoring and stats retrieval."""

    def __init__(self, session_name: str, monitor: FunctionMonitor, enable_monitoring: bool = True):
        self.session_name = session_name
        self.monitor = monitor if enable_monitoring else None
        self.session_metadata = {
            "session_name": session_name,
            "start_time": None,
            "end_time": None,
            "steps": []
        }

    def __enter__(self):
        """Start the evaluation session with monitoring."""
        self.session_metadata["start_time"] = datetime.now()

        if self.monitor:
            logger.info(f"🚀 Starting evaluation session: {self.session_name}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize the session and log metrics."""
        self.session_metadata["end_time"] = datetime.now()

        if self.monitor:
            session_duration = (self.session_metadata["end_time"] - self.session_metadata["start_time"]).total_seconds()
            logger.info(f"✅ Completed session '{self.session_name}' in {session_duration:.2f}s")

            if exc_type:
                logger.error(f"Session ended with error: {exc_val}", exc_info=True)

        return False  # Don't suppress exceptions

    @contextmanager
    def step(self, step_name: str, step_metadata: Dict[str, Any] | None = None):
        """Context manager for monitored evaluation steps."""
        if not self.monitor:
            yield
            return

        full_step_name = f"{self.session_name}.{step_name}"
        step_meta = {
            "step_name": step_name,
            "session_name": self.session_name,
            **(step_metadata or {})
        }
        self.session_metadata["steps"].append(step_name)

        # Create a monitoring decorator for this step
        @self.monitor.monitor(
            name=full_step_name,
            enable_timing=True,
            track_memory=True,
            metadata=step_meta
        )
        def _monitored_step():
            yield  # Execution happens here

        try:
            step_func = _monitored_step()
            next(step_func)  # Advance to yield point
            yield  # Execute the step code

            try:
                next(step_func)  # Finalize monitoring
            except StopIteration:
                pass

        except Exception as e:
            logger.error(f"Error in step '{step_name}': {str(e)}", exc_info=True)
            raise

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire evaluation session."""
        if not self.monitor:
            return {"error": "Monitoring not enabled for this session"}

        stats = {
            "session_name": self.session_name,
            "start_time": self.session_metadata["start_time"].isoformat(),
            "end_time": self.session_metadata["end_time"].isoformat() if self.session_metadata["end_time"] else None,
            "duration_seconds": (
                (self.session_metadata["end_time"] - self.session_metadata["start_time"]).total_seconds()
                if self.session_metadata["end_time"] else None
            ),
            "steps": self.session_metadata["steps"],
            "step_details": {},
            "aggregated": {
                "total_duration": 0.0,
                "total_memory_peak_mb": 0.0,
                "total_api_calls": 0,
                "error_count": 0
            }
        }

        # Get stats for each step
        for step in self.session_metadata["steps"]:
            full_step_name = f"{self.session_name}.{step}"
            step_stats = self.monitor.get_stats(full_step_name)

            if step_stats:
                stats["step_details"][step] = step_stats

                # Aggregate totals
                stats["aggregated"]["total_duration"] += step_stats.get("avg_duration", 0) * step_stats.get(
                    "total_calls", 1)
                stats["aggregated"]["total_memory_peak_mb"] = max(
                    stats["aggregated"]["total_memory_peak_mb"],
                    step_stats.get("memory_peak_mb", 0)
                )
                stats["aggregated"]["error_count"] += step_stats.get("error_count", 0)

                # Count API calls if available
                if "custom_metrics" in step_stats:
                    api_calls = step_stats["custom_metrics"].get("total_api_calls", {}).get(full_step_name, 0)
                    stats["aggregated"]["total_api_calls"] += api_calls

        return stats

    def get_step_metrics(self, step_name: str) -> ExecutionMetrics | None:
        """Get raw execution metrics for a specific step."""
        if not self.monitor:
            return None

        full_step_name = f"{self.session_name}.{step_name}"
        history = self.monitor.get_execution_history(full_step_name)
        return history[-1] if history else None

    def get_step_stats(self, step_name: str) -> Dict[str, Any] | None:
        """Get aggregated statistics for a specific step."""
        if not self.monitor:
            return None

        full_step_name = f"{self.session_name}.{step_name}"
        return self.monitor.get_stats(full_step_name)

    def validate_session(self, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate session metrics against performance thresholds.

        Args:
            thresholds: Dictionary of threshold values to check against
                Example: {
                    "max_duration_seconds": 60,
                    "max_memory_mb": 1024,
                    "max_error_rate": 5.0,
                    "max_api_calls": 100
                }

        Returns:
            Dictionary with validation results
        """
        stats = self.get_session_stats()
        results = {
            "passed": True,
            "checks": {}
        }

        # Check total duration
        if "max_duration_seconds" in thresholds:
            duration = stats.get("duration_seconds", 0)
            max_duration = thresholds["max_duration_seconds"]
            results["checks"]["duration"] = {
                "value": duration,
                "threshold": max_duration,
                "passed": duration <= max_duration if duration else None
            }
            results["passed"] &= results["checks"]["duration"]["passed"] if results["checks"]["duration"]["passed"] else False

        # Check memory usage
        if "max_memory_mb" in thresholds:
            memory = stats["aggregated"].get("total_memory_peak_mb", 0)
            max_memory = thresholds["max_memory_mb"]
            results["checks"]["memory"] = {
                "value": memory,
                "threshold": max_memory,
                "passed": memory <= max_memory
            }
            results["passed"] &= results["checks"]["memory"]["passed"]

        # Check error rate
        if "max_error_rate" in thresholds:
            total_calls = sum(
                s.get("total_calls", 1)
                for s in stats["step_details"].values()
            )
            error_count = stats["aggregated"].get("error_count", 0)
            error_rate = (error_count / total_calls * 100) if total_calls > 0 else 0
            max_error_rate = thresholds["max_error_rate"]
            results["checks"]["error_rate"] = {
                "value": error_rate,
                "threshold": max_error_rate,
                "passed": error_rate <= max_error_rate
            }
            results["passed"] &= results["checks"]["error_rate"]["passed"]

        # Check API calls
        if "max_api_calls" in thresholds:
            api_calls = stats["aggregated"].get("total_api_calls", 0)
            max_api_calls = thresholds["max_api_calls"]
            results["checks"]["api_calls"] = {
                "value": api_calls,
                "threshold": max_api_calls,
                "passed": api_calls <= max_api_calls
            }
            results["passed"] &= results["checks"]["api_calls"]["passed"]

        return results
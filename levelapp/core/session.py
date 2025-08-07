"""levelapp/core/session.py"""
import logging
import threading

from datetime import datetime
from contextlib import contextmanager

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Any, Callable, ContextManager

from levelapp.utils.monitoring import FunctionMonitor


logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata for an evaluation session."""
    session_name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    total_executions: int = 0
    total_duration: float = 0.0
    steps: Dict[str, 'StepMetadata'] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if the session is currently active."""
        return self.ended_at is None

    @property
    def duration(self) -> float | None:
        """Calculate the duration of the session in seconds."""
        if not self.is_active:
            return (self.ended_at - self.started_at).total_seconds()
        return None


@dataclass
class StepMetadata:
    """Metadata for a specific step within an evaluation session."""
    step_name: str
    session_name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    memory_peak_mb: float | None = None
    error_count: int = 0

    @property
    def is_active(self) -> bool:
        """Check if the step is currently active."""
        return self.ended_at is None

    @property
    def duration(self) -> float | None:
        """Calculate the duration of the step in seconds."""
        if not self.is_active:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class EvaluationSession:
    """Context manager for LLM evaluation sessions with integrated monitoring and stats retrieval."""

    def __init__(self, session_name: str, monitor: FunctionMonitor | None = None):
        self.session_name = session_name
        self.monitor = monitor or FunctionMonitor()
        self.session_metadata = SessionMetadata(session_name=session_name)
        self._lock = threading.RLock()

    def __enter__(self):
        """Start the evaluation session with monitoring."""
        self.session_metadata.started_at = datetime.now()
        logger.info(f"Starting evaluation session: {self.session_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize the session and log metrics."""
        self.session_metadata.ended_at = datetime.now()
        session_duration = self.session_metadata.duration
        logger.info(f"Completed session '{self.session_name}' in {session_duration:.2f}s")

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

        with self._lock:
            step_meta = StepMetadata(
                step_name=step_name,
                session_name=self.session_name,
                started_at=datetime.now()
            )
            self.session_metadata.steps[step_name] = step_meta

        # Create the monitored function
        @self.monitor.monitor(
            name=full_step_name,
            enable_timing=True,
            track_memory=True,
            metadata={
                "step_name": step_name,
                "session_name": self.session_name,
                **(step_metadata or {})
            }
        )
        def _monitored_step():
            yield  # This is where the step execution happens

        # Create and manage the generator
        step_gen = _monitored_step()

        try:
            next(step_gen)  # Enter the monitored context
            yield  # Execute the step code here

        except Exception as e:
            with self._lock:
                step_meta.error_count += 1
            logger.error(f"Error in step '{step_name}': {str(e)}", exc_info=True)
            raise

        finally:
            try:
                next(step_gen)  # Exit the monitored context
            except StopIteration:
                pass

            with self._lock:
                step_meta.ended_at = datetime.now()
                self.session_metadata.total_executions += 1
                if step_meta.duration:
                    self.session_metadata.total_duration += step_meta.duration

            logger.info(f"Completed step '{step_name}' in {step_meta.duration:.2f}s")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        stats = {
            "session": {
                "name": self.session_name,
                "duration": self.session_metadata.duration,
                "start_time": self.session_metadata.started_at.isoformat() if self.session_metadata.started_at else None,
                "end_time": self.session_metadata.ended_at.isoformat() if self.session_metadata.ended_at else None,
                "steps": len(self.session_metadata.steps),
                "errors": sum(s.error_count for s in self.session_metadata.steps.values())
            },
            "stats": self.monitor.get_all_stats()
        }
        return stats

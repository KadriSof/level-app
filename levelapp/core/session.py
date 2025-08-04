"""levelapp/core/session.py"""
import threading
import uuid
import weakref

from datetime import datetime, timedelta
from contextlib import contextmanager

from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, Set, List, Any, Deque, Generator

from levelapp.utils.monitoring import FunctionMonitor, ExecutionMetrics


class EvaluationPhase(Enum):
    """Different phases of the evaluation process."""
    SETUP = "setup"
    DATA_LOADING = "data_loading"
    INFERENCE = "inference"
    SCORING = "scoring"
    AGGREGATION = "aggregation"
    CLEANUP = "cleanup"


@dataclass
class ExecutionContext:
    """Context information for the evaluation execution."""
    session_id: str
    evaluation_id: str | None = None
    phase: EvaluationPhase | None = None
    batch_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: 'ExecutionContext' | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now)

    def child_context(self, **kwargs) -> 'ExecutionContext':
        """
        Create a child context inheriting from the current context.

        Args:
            **kwargs: Additional attributes to set for the child context.

        Returns:
            ExecutionContext: A new child context with inherited properties.
        """
        return ExecutionContext(
            session_id=self.session_id,
            evaluation_id=self.evaluation_id or kwargs.get('evaluation_id'),
            phase=self.phase or kwargs.get('phase'),
            batch_id=self.batch_id or kwargs.get('batch_id'),
            metadata={**self.metadata, **kwargs.get('metadata', {})},
            parent_context=self
        )

@dataclass
class SessionMetrics:
    """Metrics for an evaluation session."""
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    total_executions: int = 0
    total_duration: float = 0.0
    total_errors: int = 0
    phases_executed: Set[str] = field(default_factory=set)
    evaluations: Set[str] = field(default_factory=set)
    functions_called: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if the session is still active."""
        return self.ended_at is None

    @property
    def duration(self) -> float | None:
        """Calculate the total duration of the session."""
        if not self.ended_at:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the session."""
        if self.total_executions == 0:
            return 100.0
        return ((self.total_executions - self.total_errors) / self.total_executions) * 100.0


class EvaluationSessionManager:
    """Manager for handling evaluation sessions and their metrics."""

    def __init__(self, monitor: FunctionMonitor):
        self._monitor = monitor or FunctionMonitor()
        self._context_stack: threading.local = threading.local()
        self._sessions: Dict[str, SessionMetrics] = {}
        self._session_contexts: Dict[str, List[ExecutionContext]] = defaultdict(list)
        self._execution_timeline = Dict[str, Deque[ExecutionMetrics]] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._active_sessions: Set[str] = set()
        self._context_refs: weakref.WeakSet = weakref.WeakSet()

    def _get_context_stack(self) -> List[ExecutionContext]:
        """Get the current context stack for the thread."""
        if not hasattr(self._context_stack, 'contexts'):
            self._context_stack.contexts = []
        return self._context_stack.contexts

    def _get_current_context(self) -> ExecutionContext | None:
        """Get the current execution context."""
        stack = self._get_context_stack()
        return stack[-1] if stack else None

    def _update_session_metrics(self, session_id: str, execution_metrics: ExecutionMetrics) -> None:
        """Update or create session metrics."""
        with self._lock:
            if session_id not in self._sessions:
                return

            session = self._sessions[session_id]
            session.total_executions += 1

            if execution_metrics.duration:
                session.total_duration += execution_metrics.duration

            if execution_metrics.error:
                session.total_errors += 1

            session.functions_called.add(execution_metrics.function_name)

            if hasattr(execution_metrics, 'metadata') and execution_metrics.metadata:
                if 'phase' in execution_metrics.metadata:
                    session.phases_executed.add(execution_metrics.metadata['phase'])
                if 'evaluation_id' in execution_metrics.metadata:
                    session.evaluations.add(execution_metrics.metadata['evaluation_id'])
                if 'batch_id' in execution_metrics.metadata:
                    session.metadata['batch_id'] = execution_metrics.metadata['batch_id']

    @contextmanager
    def session(self, session_id: str | None = None, **metadata) -> Generator[ExecutionContext, None, None]:
        """
        Context manager for managing an evaluation session.

        Args:
            session_id (str | None): Optional session ID. If None, a new ID will be generated.
            **metadata: Additional metadata for the session.

        Yields:
            ExecutionContext: The current execution context.
        """
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:8]}"

        context = ExecutionContext(session_id=session_id, metadata=metadata)
        stack = self._get_context_stack()
        stack.append(context)

        with self._lock:
            if session_id in self._active_sessions:
                raise RuntimeError(f"Session {session_id} is already active.")

            self._active_sessions.add(session_id)
            started_at = datetime.now()
            session_metrics = SessionMetrics(session_id=session_id, started_at=started_at, metadata=metadata)
            self._sessions[session_id] = session_metrics
            self._session_contexts[session_id].append(context)

        self._context_refs.add(context)

        original_wrap = self._monitor._wrap_execution

        def enhanced_wrap(func, name, enable_timing, track_memory, metadata=None):
            """Enhanced wrapper to include session context."""
            wrapped_func = original_wrap(func, name, enable_timing, track_memory, metadata)

            def session_aware_wrapper(*args, **kwargs):
                result = wrapped_func(*args, **kwargs)

                history = self._monitor.get_execution_history(name, limit=1)

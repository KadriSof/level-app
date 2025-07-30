"""levelapp/utils.aspects.py"""
from functools import wraps
from threading import Lock

from typing import Callable, Dict, Any

METRICS: Dict[str, Callable[[str, str], Any]] = {}
_metrics_lock = Lock()  # Thread-safe registration

def register_metric(name: str):
    """Decorator to register a metric function under a given name."""
    def decorator(func: Callable[[str, str], Any]):
        if func.__code__.co_argcount != 2:
            raise ValueError(f"Metric '{name}' must take exactly 2 arguments (generated, reference)")

        # TODO-0: Add wrapping logic here.
        @wraps(func)
        def wrapped(generated: str, reference: str) -> Any:
            return func(reference, generated)

        with _metrics_lock:
            if name in METRICS:
                raise KeyError(f"Metric '{name}' already registered!")

            METRICS[name] = wrapped
            # TODO-2: Change the print to log.
            print(f"Registered metric: {name}")

        return wrapped

    return decorator
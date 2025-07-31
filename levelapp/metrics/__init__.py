"""levelapp/metrics/__init__.py"""
import logging

from typing import Dict, Type

from levelapp.core.base import BaseMetric
from levelapp.metrics.fuzzy import FUZZY_METRICS

logger = logging.getLogger(__name__)


class MetricRegistry:
    _metrics: Dict[str, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, name: str, metric_class: Type[BaseMetric]) -> None:
        """
        Register a metric class under a given name.

        Args:
            name (str): Unique identifier for the metric.
            metric_class (Type[BaseMetric]): The metric class to register.
        """
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"Metric '{name}' must be a subclass of BaseMetric")

        if name in cls._metrics:
            raise KeyError(f"Metric '{name}' is already registered")

        cls._metrics[name] = metric_class
        logger.info(f"Metric '{name}' registered successfully.")

    @classmethod
    def get(cls, name: str, **kwargs) -> Type[BaseMetric]:
        """
        Retrieve a registered metric class by its name.

        Args:
            name (str): The name of the metric to retrieve.

        Returns:
            Type[BaseMetric]: The metric class associated with the given name.

        Raises:
            KeyError: If the metric is not found.
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")

        return cls._metrics[name](**kwargs)


for name, metric_class in FUZZY_METRICS.items():
    try:
        MetricRegistry.register(name, metric_class)
        logger.info(f"Successfully registered metric: {name}")

    except Exception as e:
        logger.info(f"Failed to register metric {name}: {e}")

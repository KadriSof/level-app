"""levelapp/core/workflow.py"""
from typing import Any, Dict

from levelapp.core.base import BaseWorkflow, BaseSimulator, BaseComparator
from levelapp.core.comparator import MetadataComparator
from levelapp.core.simulator import ConversationSimulator


class SimulatorWorkflow(BaseWorkflow):

    def __init__(self) -> None:
        super().__init__(name="ConversationSimulator")
        self.simulator: BaseSimulator = ConversationSimulator()

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Set up the simulator component.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.simulator = ConversationSimulator(**self.config)

    def load_data(self, data_loader: Any) -> None:
        self.data = data_loader.load()

    def execute(self) -> None:
        if not (self.simulator and self.data):
            raise RuntimeError("[SimulatorWorkflow] Workflow not properly initialized.")
        self.results = self.simulator.simulate()

    def collect_results(self) -> Any:
        return self.results


class ComparatorWorkflow(BaseWorkflow):
    """Workflow for metadata extraction evaluation."""

    def __init__(self) -> None:
        super().__init__(name="MetadataComparator")
        self.comparator: BaseComparator = MetadataComparator()

    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.comparator = MetadataComparator(**self.config)

    def load_data(self, data_loader: Any) -> None:
        self.data = data_loader.load()

    def execute(self) -> None:
        if not (self.comparator and self.data):
            raise RuntimeError("[ComparatorWorkflow] Workflow not properly initialized.")
        self.results = self.comparator.compare()

    def collect_results(self) -> Any:
        return self.results

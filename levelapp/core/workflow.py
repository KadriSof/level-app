"""levelapp/core/workflow.py"""
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

from levelapp.config.interaction_request import EndpointConfig
from levelapp.core.evaluator import JudgeEvaluator
from levelapp.repository.firestore import FirestoreRepository
from levelapp.simulator.schemas import ScriptsBatch
from levelapp.utils.data_loader import load_json_file

from levelapp.core.base import BaseWorkflow, BaseEngine, BaseEvaluator, BaseRepository
from levelapp.core.comparator import MetadataComparator
from levelapp.core.simulator import ConversationSimulator


class WorkflowType(Enum):
    SIMULATOR = "SIMULATOR"
    COMPARATOR = "COMPARATOR"
    ASSESSOR = "ASSESSOR"


class RepositoryType(Enum):
    FIRESTORE = "FIRESTORE"
    FILESYSTEM = "FILESYSTEM"


class EvaluatorType(Enum):
    JUDGE = "JUDGE"
    METADATA = "METADATA"
    RAG = "RAG"


@dataclass
class WorkflowConfiguration:
    workflow_type: WorkflowType
    repository_type: RepositoryType
    evaluator_type: EvaluatorType

    # Optional overrides
    data_source: Path | None = None
    endpoint: EndpointConfig | None = None

    @property
    def engine(self) -> BaseEngine:
        match self.workflow_type:
            case WorkflowType.SIMULATOR:
                return ConversationSimulator()
            case WorkflowType.COMPARATOR:
                return MetadataComparator()
            case WorkflowType.ASSESSOR:
                return ConversationSimulator()

    @property
    def repository(self) -> BaseRepository:
        match self.repository_type:
            case RepositoryType.FIRESTORE:
                return FirestoreRepository()
            case RepositoryType.FILESYSTEM:
                return FirestoreRepository()

    @property
    def evaluator(self) -> BaseEvaluator:
        match self.evaluator_type:
            case EvaluatorType.JUDGE:
                return JudgeEvaluator()
            case EvaluatorType.METADATA:
                return JudgeEvaluator()
            case EvaluatorType.RAG:
                return JudgeEvaluator()


class WorkflowFactory:
    """Builds configured workflows from a WorkflowConfiguration."""

    @staticmethod
    def build(config: WorkflowConfiguration) -> BaseWorkflow:
        repository = WorkflowFactory._build_repository(config.repository_type)
        evaluator = WorkflowFactory._build_evaluator(config.evaluator_type)

        if config.workflow_type == WorkflowType.SIMULATOR:
            workflow = SimulatorWorkflow()
            workflow.repository = repository
            workflow.evaluator = evaluator
            workflow.engine = ConversationSimulator()
            return workflow

        elif config.workflow_type == WorkflowType.COMPARATOR:
            workflow = ComparatorWorkflow()
            workflow.config.repository = repository
            workflow.config.evaluators = [evaluator]
            workflow.config.engine = MetadataComparator()
            return workflow

        else:
            raise NotImplementedError(f"Workflow type {config.workflow_type} not supported yet.")

    # --- Helpers ---
    @staticmethod
    def _build_repository(rtype: RepositoryType) -> BaseRepository:
        match rtype:
            case RepositoryType.FIRESTORE:
                return FirestoreRepository()
            case RepositoryType.FILESYSTEM:
                return FirestoreRepository()

    @staticmethod
    def _build_evaluator(etype: EvaluatorType) -> BaseEvaluator:
        match etype:
            case EvaluatorType.JUDGE:
                return JudgeEvaluator()
            case EvaluatorType.METADATA:
                return JudgeEvaluator()
            case EvaluatorType.RAG:
                return JudgeEvaluator()


class SimulatorWorkflow(BaseWorkflow):

    def __init__(self) -> None:
        super().__init__(name="ConversationSimulator")
        self.config: WorkflowConfiguration | None = None
        self.engine: BaseEngine | None = None
        self.repository: BaseRepository | None = None
        self.evaluator: BaseEvaluator | None = None
        self.endpoint_config: EndpointConfig | None = None
        self.data: ScriptsBatch | None = None
        self.results: Any | None = None

    def setup(self, config: WorkflowConfiguration | None = None) -> None:
        """
        Set up the simulator component.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        if not config:
            self.engine = ConversationSimulator(
                storage_service=FirestoreRepository(),
                evaluation_service=JudgeEvaluator(),
                endpoint_configuration=EndpointConfig()
            )

        self.engine.setup(
            repository=self.repository,
            evaluator=self.evaluator,
            endpoint_config=config.endpoint
        )

    def load_data(self) -> None:
        file_path = Path(config.get("file_path", "no-file-path"))
        if not file_path.exists():
            raise FileNotFoundError(f"No file path was provide (default value: {file_path})")

        self.data = load_json_file(model=ScriptsBatch, file_path=file_path)

    def execute(self, config: Dict[str, Any]) -> None:
        if not (self.simulator and self.data):
            raise RuntimeError("[SimulatorWorkflow] Workflow not properly initialized.")
        config["test_batch"] = self.data
        self.config.base_engine.run(**self.config.engine_config)
        self.results = self.simulator.simulate(**config)

    def collect_results(self) -> Any:
        return self.results


class ComparatorWorkflow(BaseWorkflow):
    """Workflow for metadata extraction evaluation."""

    def __init__(self) -> None:
        super().__init__(name="MetadataComparator")
        self.config: WorkflowConfiguration | None = None
        self.engine: BaseEngine | None = None
        self.repository: BaseRepository | None = None
        self.evaluator: BaseEvaluator | None = None
        self.data: ScriptsBatch | None = None
        self.results: Any | None = None

    def setup(self, config: Dict[str, Any]) -> None:
        # TODO-1: Add a default config for the comparator workflow.
        self.config = config
        self.comparator = MetadataComparator(**self.config)

    def load_data(self, config: Any) -> None:
        self.data = config.load()

    def execute(self, config: dict) -> None:
        if not (self.comparator and self.data):
            raise RuntimeError("[ComparatorWorkflow] Workflow not properly initialized.")
        self.results = self.comparator.compare()

    def collect_results(self) -> Any:
        return self.results

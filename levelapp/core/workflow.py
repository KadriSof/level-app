"""levelapp/core/workflow.py"""
import json
import asyncio
import logging

from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Dict, Any

from levelapp.config.interaction_request import EndpointConfig
from levelapp.core.evaluator import JudgeEvaluator
from levelapp.repository.firestore import FirestoreRepository
from levelapp.simulator.schemas import ScriptsBatch
from levelapp.utils.loader import DataLoader

from levelapp.core.base import BaseWorkflow, BaseProcess, BaseEvaluator, BaseRepository
from levelapp.core.comparator import MetadataComparator
from levelapp.core.simulator import ConversationSimulator


logger = logging.getLogger(__name__)


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return [e.value for e in cls]


class WorkflowType(ExtendedEnum):
    SIMULATOR = "SIMULATOR"
    COMPARATOR = "COMPARATOR"
    ASSESSOR = "ASSESSOR"


class RepositoryType(ExtendedEnum):
    FIRESTORE = "FIRESTORE"
    FILESYSTEM = "FILESYSTEM"


class EvaluatorType(ExtendedEnum):
    JUDGE = "JUDGE"
    REFERENCE = "REFERENCE"
    RAG = "RAG"


@dataclass
class WorkflowConfiguration:
    def __init__(self):
        self._fields_list: List[str] = [
            'project_name', 'project_params', 'workflow',
            'repository', 'repository', 'evaluators', 'reference_data'
        ]
        self.process: BaseProcess | None = None
        self.workflow: WorkflowType | None = None
        self.repository: RepositoryType | None = None
        self.evaluators: List[EvaluatorType] | None = None
        #
        self.endpoint_config: EndpointConfig | None = None
        self.reference_data_path: str | None = None

    def load_configuration(self, path: str | None = None) -> BaseModel | None:
        loader = DataLoader()
        config_dict = loader.load_configuration(path=path)
        config = loader.load_data(data=config_dict, model_name="WorkflowConfiguration")

        if not (self.fields_check(config=config) and self.values_check(config=config)):
            raise ValueError("[WorkflowConfiguration] Invalid configuration")

        self.workflow = WorkflowType(config.workflow)
        self.repository = RepositoryType(config.repository)
        self.evaluators = [EvaluatorType(_) for _ in config.evaluators]
        self.reference_data_path = config.reference_data.path
        self.endpoint_config = EndpointConfig.model_validate(config.endpoint_configuration.model_dump())

        return config

    # TODO-1: Once the RAG Assessor is complete remove the 'None' from the return type.
    def build_workflow(self, workflow_type: WorkflowType) -> BaseWorkflow | None:
        # repository = self.set_repository(repository_type=self.repository)
        # evaluators = self.set_evaluator(evaluator_types=self.evaluators)
        match workflow_type:
            case WorkflowType.SIMULATOR:
                return SimulatorWorkflow()
            case WorkflowType.COMPARATOR:
                return ComparatorWorkflow()
            case WorkflowType.ASSESSOR:
                return None

    def set_repository(self, repository_type: RepositoryType) -> BaseRepository | None:
        match repository_type:
            case RepositoryType.FIRESTORE:
                return FirestoreRepository()
            case RepositoryType.FILESYSTEM:
                return None

    @staticmethod
    def set_evaluator(evaluator_types: List[EvaluatorType]) -> List[BaseEvaluator]:
        selected_evaluators: List[BaseEvaluator | None] = []
        for _ in evaluator_types:
            match _:
                case EvaluatorType.JUDGE:
                    selected_evaluators.append(JudgeEvaluator())
                case EvaluatorType.REFERENCE:
                    selected_evaluators.append(None)
                case EvaluatorType.RAG:
                    selected_evaluators.append(None)

        return selected_evaluators

    def fields_check(self, config: BaseModel) -> bool:
        config_fields: List = []
        for field, info in type(config).model_fields.items():
            config_fields.append(field)

        for field in self._fields_list:
            if field not in config_fields:
                logger.warning(f"[WorkflowConfiguration] Field '{field}' not found in configuration.")
                return False

        return True

    def values_check(self, config: BaseModel):
        workflow_type = getattr(config, 'workflow')
        repository_type = getattr(config, 'repository')
        evaluator_type = getattr(config, 'evaluators')

        if workflow_type not in WorkflowType.list():
            logger.warning(f"[WorkflowConfiguration] Workflow type '{workflow_type}' is not supported.")
            return False

        if repository_type not in RepositoryType.list():
            logger.warning(f"[WorkflowConfiguration] Repository type '{repository_type}' is not supported.")
            return False

        for _ in evaluator_type:
            if _ not in EvaluatorType.list():
                logger.warning(f"[WorkflowConfiguration] Evaluator type '{evaluator_type}' is not supported.")
                return False

        if len(evaluator_type) == 0:
            logger.warning(f"[WorkflowConfiguration] No evaluator type specified.")
            return False

        return True


class SimulatorWorkflow(BaseWorkflow):

    def __init__(self) -> None:
        super().__init__(name="ConversationSimulator")
        self.config: WorkflowConfiguration = WorkflowConfiguration()
        self.process: ConversationSimulator = ConversationSimulator()
        self.repository: BaseRepository | None = None
        self.evaluators: List[BaseEvaluator] | None = None
        self.endpoint_config: EndpointConfig | None = EndpointConfig()
        self.data: ScriptsBatch | None = None
        self.results: Any | None = None

    def setup(self, config: WorkflowConfiguration | None = None) -> None:
        """
        Set up the simulator component.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        if config:
            self.config = config

        self.config.load_configuration()
        self.evaluators = self.config.set_evaluator(evaluator_types=self.config.evaluators)
        self.repository = self.config.set_repository(repository_type=self.config.repository)
        self.process.setup(
            repository=self.repository,
            evaluator=self.evaluators[0],
            endpoint_config=self.config.endpoint_config,
        )

    def load_data(self) -> None:
        loader = DataLoader()
        file_path = Path(self.config.reference_data_path or 'no-path')
        if not file_path.exists():
            raise FileNotFoundError(f"No file path was provide (default value: {file_path})")

        data = loader.load_configuration(path=self.config.reference_data_path)
        self.data = loader.load_data(data=data, model_name="ScriptsBatch")

    def execute(self) -> None:
        if asyncio.iscoroutinefunction(self.process.run):
            self.results = asyncio.run(
                self.process.run(test_batch=self.data)
            )
        else:
            self.results = self.process.run(self.data)

    def collect_results(self) -> Any:
        return self.results


class ComparatorWorkflow(BaseWorkflow):
    """Workflow for metadata extraction evaluation."""

    def __init__(self) -> None:
        super().__init__(name="MetadataComparator")
        self.config: WorkflowConfiguration | None = None
        self.engine: BaseProcess | None = None
        self.repository: BaseRepository | None = None
        self.evaluator: BaseEvaluator | None = None
        self.data: ScriptsBatch | None = None
        self.results: Any | None = None

    def setup(self, config: WorkflowConfiguration | None = None) -> None:
        # TODO document why this method is empty
        pass

    def load_data(self) -> None:
        # TODO document why this method is empty
        pass

    def execute(self) -> None:
        # TODO document why this method is empty
        pass

    def collect_results(self) -> Any:
        # TODO document why this method is empty
        pass


if __name__ == '__main__':
    loader_ = DataLoader()
    config_dict_ = loader_.load_configuration()
    config_ = loader_.load_data(data=config_dict_, model_name="WorkflowConfiguration")
    print(f"Workflow configuration:\n{config_.model_dump_json(indent=4)}\n---")

    endpoint_config = EndpointConfig.model_validate(config_.endpoint_configuration.model_dump())
    endpoint_config.variables = {"user_message": "Hello, world!"}
    print(f"Endpoint configuration:\n{endpoint_config.model_dump()}\n---")


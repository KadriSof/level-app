"""levelapp/core/workflow.py"""
import json
import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import yaml

from levelapp.config.interaction_request import EndpointConfig
from levelapp.core.evaluator import JudgeEvaluator
from levelapp.repository.firestore import FirestoreRepository
from levelapp.simulator.schemas import ScriptsBatch
from levelapp.utils.loader import load_json_file

from levelapp.core.base import BaseWorkflow, BaseEngine, BaseEvaluator, BaseRepository
from levelapp.core.comparator import MetadataComparator
from levelapp.core.simulator import ConversationSimulator


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
    workflow_type: WorkflowType
    repository_type: RepositoryType
    evaluator_type: EvaluatorType

    # Optional overrides
    endpoint: EndpointConfig | None = None
    data_source: Path | Any = None

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
            case EvaluatorType.REFERENCE:
                return JudgeEvaluator()
            case EvaluatorType.RAG:
                return JudgeEvaluator()

    def load_configuration(self, path: str | None = None):
        try:
            if not path:
                path = os.getenv('WORKFLOW_CONFIG_PATH')

            if not os.path.exists(path):
                raise FileNotFoundError(f"The provided configuration file path '{path}' does not exist.")

            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith((".yaml", ".yml")):
                    content = yaml.safe_load(f)

                elif path.endswith(".json"):
                    content = json.load(f)

                else:
                    raise ValueError("[WorkflowConfiguration] Unsupported file format.")

                return content
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"[EndpointConfig] Payload template file '{e.filename}' not found in path.")

        except yaml.YAMLError as e:
            raise ValueError(f"[EndpointConfig] Error parsing YAML file:\n{e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"[EndpointConfig] Error parsing JSON file:\n{e}")

        except IOError as e:
            raise IOError(f"[EndpointConfig] Error reading file:\n{e}")

        except Exception as e:
            raise ValueError(f"[EndpointConfig] Unexpected error loading configuration:\n{e}")


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
            case EvaluatorType.REFERENCE:
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
        # TODO-0: Add DataLoader base class and concrete implementation.
        self.loader = None
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
            repository=config.repository,
            evaluator=config.evaluator,
            endpoint_config=config.endpoint
        )

    def load_data(self) -> None:
        file_path = Path(self.config.data_source or 'no-path')
        if not file_path.exists():
            raise FileNotFoundError(f"No file path was provide (default value: {file_path})")

        self.data = load_json_file(model=ScriptsBatch, file_path=file_path)

    def execute(self) -> None:
        data = {"test_batch": self.data}
        self.results = self.engine.run(**data)

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
        self.engine = MetadataComparator(**self.config)

    def load_data(self, config: Any) -> None:
        self.data = config.load()

    def execute(self, config: dict) -> None:
        if not (self.comparator and self.data):
            raise RuntimeError("[ComparatorWorkflow] Workflow not properly initialized.")
        self.results = self.comparator.compare()

    def collect_results(self) -> Any:
        return self.results


if __name__ == '__main__':
    from dotenv import load_dotenv


    def load_configuration(path: str | None = None) -> Dict[str, Any]:
        try:
            load_dotenv()
            if not path:
                path = os.getenv('WORKFLOW_CONFIG_PATH')

            if not os.path.exists(path):
                raise FileNotFoundError(f"The provided configuration file path '{path}' does not exist.")

            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith((".yaml", ".yml")):
                    content = yaml.safe_load(f)

                elif path.endswith(".json"):
                    content = json.load(f)

                else:
                    raise ValueError("[WorkflowConfiguration] Unsupported file format.")

                return content

        except FileNotFoundError as e:
            raise FileNotFoundError(f"[EndpointConfig] Payload template file '{e.filename}' not found in path.")

        except yaml.YAMLError as e:
            raise ValueError(f"[EndpointConfig] Error parsing YAML file:\n{e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"[EndpointConfig] Error parsing JSON file:\n{e}")

        except IOError as e:
            raise IOError(f"[EndpointConfig] Error reading file:\n{e}")

        except Exception as e:
            raise ValueError(f"[EndpointConfig] Unexpected error loading configuration:\n{e}")

    sample_config = load_configuration()
    print(type(sample_config))
    print(sample_config)

    repository_engine = sample_config['repository']['engine']
    flag = repository_engine in RepositoryType.list()
    print(flag)
    evaluators = sample_config['evaluators']
    print(len(evaluators))

from dataclasses import dataclass
from typing import List
from pydantic import BaseModel
from levelapp.config.interaction_request import EndpointConfig
from levelapp.core.base import BaseRepository, BaseEvaluator
from levelapp.utils.loader import DataLoader
from enum import Enum


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


class WorkflowConfig:
    """Configuration for a workflow, loaded from JSON/YAML via DataLoader."""

    # Class-level constant
    _fields_list: List[str] = [
        "project_name",
        "project_params",
        "workflow",
        "repository",
        "evaluator",
        "reference_data",
    ]

    def __init__(
        self,
        workflow: WorkflowType,
        repository: RepositoryType,
        evaluator: EvaluatorType,
        endpoint_config: EndpointConfig,
        reference_data_path: str | None = None,
    ):
        self.workflow = workflow
        self.repository = repository
        self.evaluator = evaluator
        self.endpoint_config = endpoint_config
        self.reference_data_path = reference_data_path

    @classmethod
    def load(cls, path: str | None = None) -> "WorkflowConfig":
        """Load and validate workflow configuration from a file."""
        loader = DataLoader()
        config_dict = loader.load_configuration(path=path)
        model_config: BaseModel = loader.load_data(data=config_dict, model_name="WorkflowConfiguration")

        cls._check_fields(model_config)
        cls._check_values(model_config)

        workflow = WorkflowType(model_config.workflow)
        repository = RepositoryType(model_config.repository)
        evaluator = EvaluatorType(model_config.evaluator)
        reference_data_path = getattr(model_config.reference_data, "path", None)
        endpoint_config = EndpointConfig.model_validate(model_config.endpoint_configuration.model_dump())

        return cls(
            workflow=workflow,
            repository=repository,
            evaluator=evaluator,
            endpoint_config=endpoint_config,
            reference_data_path=reference_data_path
        )

    @classmethod
    def _check_fields(cls, config: BaseModel) -> None:
        for field_name in cls._fields_list:
            if field_name not in config.model_fields:
                raise ValueError(f"[WorkflowConfig] Field '{field_name}' missing in configuration")

    @staticmethod
    def _check_values(config: BaseModel) -> None:
        if config.workflow not in WorkflowType.list():
            raise ValueError(f"[WorkflowConfig] Unsupported workflow type '{config.workflow}'")
        if config.repository not in RepositoryType.list():
            raise ValueError(f"[WorkflowConfig] Unsupported repository type '{config.repository}'")
        if config.evaluator not in EvaluatorType.list():
            raise ValueError(f"[WorkflowConfig] Invalid or missing evaluator type '{config.evaluator}'")


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable data holder for workflow execution context."""
    config: WorkflowConfig
    repository: BaseRepository
    evaluator: BaseEvaluator
    endpoint_config: EndpointConfig
    reference_data_path: str

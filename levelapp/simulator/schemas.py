"""
levelapp/simulator/schemas.py

Defines Pydantic models for simulator-related data structures,
including test configurations, batch metadata, and evaluation results.
"""
from enum import Enum
from uuid import UUID, uuid4

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, HttpUrl, SecretStr, Field, computed_field


class InteractionLevel(str, Enum):
    """Enum representing the type of interaction."""
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


# TODO: Remove this?
class InteractionDetails(BaseModel):
    """Model representing details of a simulated interaction."""
    reply: Optional[str] = "No response"
    extracted_metadata: Optional[Dict[str, Any]] = {}
    handoff_details: Optional[Dict[str, Any]] = {}
    interaction_type: Optional[InteractionLevel] = InteractionLevel.INITIAL


class Interaction(BaseModel):
    """Represents a single interaction within a conversation."""
    id: UUID = Field(default_factory=uuid4, description="Interaction identifier")
    user_message: str = Field(..., description="The user's query message")
    generated_reply: str = Field(..., description="The agent's reply message")
    reference_reply: str = Field(..., description="The preset reference message")
    interaction_type: InteractionLevel = Field(..., description="Type of interaction")
    reference_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Expected metadata")
    generated_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extracted metadata")
    guardrail_flag: bool = Field(default=False, description="Flag for guardrail signaling")


class ConversationScript(BaseModel):
    """Represents a basic conversation with multiple interactions."""
    id: UUID = Field(default_factory=uuid4, description="Conversation identifier")
    interactions: List[Interaction] = Field(default_factory=list, description="List of interactions")
    description: str = Field(..., description="A short description of the conversation")
    details: Dict[str, str] = Field(default_factory=dict, description="Conversation details")


class ScriptsBatch(BaseModel):
    id: UUID = Field(default=uuid4, description="Batch identifier")
    scripts: List[ConversationScript] = Field(default_factory=list, description="List of conversation scripts")


# ---- Conversation Agent Configuration & Interaction Details Models ----
class EndpointConfig(BaseModel):
    url: HttpUrl
    api_key: SecretStr
    payload_template: Dict[str, Any]

    @property
    def headers(self) -> Dict[str, Any]:
        return {
            "x-api-key": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

    @property
    def full_url(self) -> str:
        return f"{self.url}/api/conversations/events"


class InteractionResults(BaseModel):
    """Represents metadata extracted from a VLA interaction."""
    generated_reply: Optional[str] = "No response"
    generated_metadata: Optional[Dict[str, Any]] = {}
    guardrail_details: Optional[Dict[str, Any]] = {}
    # TODO: Remove 'interaction_type'?
    interaction_type: Optional[str] = ""


class InteractionEvaluationResult(BaseModel):
    """Model representing the evaluation result of an interaction."""
    evaluations: Dict[str, Any]
    extracted_metadata_evaluation: float
    scenario_id: str


class BatchDetails(BaseModel):
    # Initial data
    project_id: str = Field(default_factory=uuid4, description="Project identifier")
    user_id: str = Field(default_factory=uuid4, description="User identifier")
    batch_id: str = Field(default_factory=uuid4, description="Batch identifier")
    # Collected data
    started_at: Optional[float]
    finished_at: Optional[float]
    evaluation_summary: Optional[Dict[str, Any]]
    average_scores: Optional[Dict[str, Any]]
    simulation_results: Optional[List[Dict[str, Any]]]

    @computed_field
    @property
    def elapsed_time(self) -> float:
        return self.finished_at - self.started_at


class TestResults(BaseModel):
    api_host: str = Field(..., alias="apiHost")
    ionos_model_name: str = Field(..., alias="ionosModelName")
    test_name: str = Field(..., alias="testName")
    test_type: str = Field(..., alias="testType")
    batch_details: Optional[BatchDetails] = Field(..., alias="results")

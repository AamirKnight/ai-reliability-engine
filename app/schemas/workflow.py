from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStep(BaseModel):
    step_id: str
    name: str
    prompt_template: str
    depends_on: List[str] = Field(default_factory=list)
    required: bool = True
    retry_count: int = 3

class WorkflowDefinition(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    
class StepExecution(BaseModel):
    step_id: str
    status: StepStatus
    attempts: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence_score: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class WorkflowExecution(BaseModel):
    execution_id: str
    workflow_id: str
    status: str  # "running", "completed", "failed"
    current_step: Optional[str] = None
    step_executions: Dict[str, StepExecution] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)  # Shared data between steps
    created_at: str
    completed_at: Optional[str] = None
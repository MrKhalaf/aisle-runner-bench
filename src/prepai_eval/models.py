"""Core data models for the eval framework."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """Definition of a browser automation task."""

    id: str
    name: str
    description: str
    start_url: str
    success_criteria: list[dict[str, Any]]
    max_steps: int = 50
    timeout_seconds: int = 300
    requires_auth: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    id: str
    provider: str  # anthropic, openai, google
    model_id: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    supports_vision: bool = True
    max_tokens: int = 4096


@dataclass
class RunConfig:
    """Configuration for a single evaluation run."""

    task: Task
    model: ModelConfig
    library: str  # browser-use, skyvern, stagehand, playwright
    headless: bool = True
    seed: Optional[int] = None
    max_actions_per_step: int = 5


@dataclass
class StepTrace:
    """Trace of a single step in the automation."""

    step_number: int
    timestamp: datetime
    screenshot_path: Optional[str]
    page_url: str
    action: dict[str, Any]
    reasoning: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class RunResult:
    """Result of a single evaluation run."""

    run_id: str
    task_id: str
    model_id: str
    library: str
    status: TaskStatus
    success: bool
    steps: int
    total_time_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None
    traces: list[StepTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_per_step(self) -> float:
        return self.total_time_seconds / self.steps if self.steps > 0 else 0

    @property
    def tokens_per_step(self) -> float:
        total = self.input_tokens + self.output_tokens
        return total / self.steps if self.steps > 0 else 0


@dataclass
class BenchmarkResult:
    """Aggregated results from multiple runs."""

    task_id: str
    model_id: str
    library: str
    runs: list[RunResult]

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.success) / len(self.runs)

    @property
    def avg_steps(self) -> float:
        successful = [r for r in self.runs if r.success]
        if not successful:
            return 0.0
        return sum(r.steps for r in successful) / len(successful)

    @property
    def avg_time(self) -> float:
        successful = [r for r in self.runs if r.success]
        if not successful:
            return 0.0
        return sum(r.total_time_seconds for r in successful) / len(successful)

    @property
    def avg_cost(self) -> float:
        successful = [r for r in self.runs if r.success]
        if not successful:
            return 0.0
        return sum(r.cost_usd for r in successful) / len(successful)

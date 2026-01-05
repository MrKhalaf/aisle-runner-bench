"""Aisle Runner Bench - Browser automation benchmark suite."""

from .benchmark import BenchmarkOrchestrator, run_benchmark
from .config import DEFAULT_MODELS, BUDGET_MODELS, PREMIUM_MODELS, get_model_by_id
from .models import (
    BenchmarkResult,
    ModelConfig,
    RunConfig,
    RunResult,
    StepTrace,
    Task,
    TaskStatus,
)
from .runners import BrowserUseRunner
from .tasks import TASKS, get_task, list_tasks

__version__ = "0.1.0"

__all__ = [
    # Benchmark
    "BenchmarkOrchestrator",
    "run_benchmark",
    # Config
    "DEFAULT_MODELS",
    "BUDGET_MODELS",
    "PREMIUM_MODELS",
    "get_model_by_id",
    # Models
    "BenchmarkResult",
    "ModelConfig",
    "RunConfig",
    "RunResult",
    "StepTrace",
    "Task",
    "TaskStatus",
    # Runners
    "BrowserUseRunner",
    # Tasks
    "TASKS",
    "get_task",
    "list_tasks",
]

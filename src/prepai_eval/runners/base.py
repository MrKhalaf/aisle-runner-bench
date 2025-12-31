"""Base class for browser automation runners."""

from abc import ABC, abstractmethod
from typing import Protocol

from ..models import RunConfig, RunResult


class BrowserRunner(Protocol):
    """Protocol for browser automation runners."""

    @abstractmethod
    async def run(self, config: RunConfig) -> RunResult:
        """Execute a task and return results."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        ...


class BaseRunner(ABC):
    """Base class with common functionality."""

    def __init__(self):
        self._traces = []

    @abstractmethod
    async def run(self, config: RunConfig) -> RunResult:
        """Execute a task and return results."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_config
    ) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output
        return input_cost + output_cost

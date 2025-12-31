"""Browser-use library adapter."""

import time
import uuid
from datetime import datetime
from typing import Optional

from browser_use import Agent, Browser, BrowserConfig
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from ..models import RunConfig, RunResult, StepTrace, TaskStatus
from .base import BaseRunner


class BrowserUseRunner(BaseRunner):
    """Runner using the browser-use library."""

    def __init__(self):
        super().__init__()
        self.browser: Optional[Browser] = None

    def _create_llm(self, config: RunConfig):
        """Create the appropriate LLM based on provider."""
        model_config = config.model

        if model_config.provider == "anthropic":
            return ChatAnthropic(
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
            )
        elif model_config.provider == "openai":
            return ChatOpenAI(
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

    async def run(self, config: RunConfig) -> RunResult:
        """Execute a task using browser-use."""
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        traces = []
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            # Create browser
            browser_config = BrowserConfig(headless=config.headless)
            self.browser = Browser(config=browser_config)

            # Create LLM
            llm = self._create_llm(config)

            # Build task prompt
            task_prompt = self._build_task_prompt(config.task)

            # Create and run agent
            agent = Agent(
                task=task_prompt,
                llm=llm,
                browser=self.browser,
                max_actions_per_step=config.max_actions_per_step,
            )

            # Run with step tracking
            history = await agent.run(max_steps=config.task.max_steps)

            # Extract metrics from history
            if history:
                for i, step in enumerate(history.history):
                    trace = StepTrace(
                        step_number=i + 1,
                        timestamp=datetime.now(),
                        screenshot_path=None,
                        page_url=step.state.url if step.state else "",
                        action=step.model_output.model_dump() if step.model_output else {},
                        reasoning=str(step.model_output) if step.model_output else "",
                        input_tokens=0,  # TODO: Extract from langchain callback
                        output_tokens=0,
                        duration_seconds=0,
                        success=True,
                    )
                    traces.append(trace)

                # Check success criteria
                final_result = history.final_result()
                success = self._check_success(config.task, final_result)
                status = TaskStatus.SUCCESS if success else TaskStatus.FAILED
            else:
                success = False
                status = TaskStatus.FAILED

            total_time = time.time() - start_time

            return RunResult(
                run_id=run_id,
                task_id=config.task.id,
                model_id=config.model.id,
                library="browser-use",
                status=status,
                success=success,
                steps=len(traces),
                total_time_seconds=total_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost_usd=self._calculate_cost(
                    total_input_tokens,
                    total_output_tokens,
                    config.model
                ),
                traces=traces,
            )

        except Exception as e:
            total_time = time.time() - start_time
            return RunResult(
                run_id=run_id,
                task_id=config.task.id,
                model_id=config.model.id,
                library="browser-use",
                status=TaskStatus.FAILED,
                success=False,
                steps=len(traces),
                total_time_seconds=total_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost_usd=0,
                error=str(e),
                traces=traces,
            )

        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Close browser."""
        if self.browser:
            await self.browser.close()
            self.browser = None

    def _build_task_prompt(self, task) -> str:
        """Build the task prompt for the agent."""
        return f"""Task: {task.name}

{task.description}

Start URL: {task.start_url}

Success criteria:
{self._format_criteria(task.success_criteria)}
"""

    def _format_criteria(self, criteria: list) -> str:
        """Format success criteria as string."""
        lines = []
        for c in criteria:
            for key, value in c.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _check_success(self, task, final_result) -> bool:
        """Check if task was completed successfully."""
        # TODO: Implement proper success checking based on criteria
        # For now, just check if we got a result
        return final_result is not None

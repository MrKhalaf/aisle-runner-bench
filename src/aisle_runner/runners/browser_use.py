"""Browser-use library adapter."""

import time
import uuid
from datetime import datetime
from typing import Any, Optional

from browser_use import Agent, Browser
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..models import RunConfig, RunResult, StepTrace, TaskStatus
from .base import BaseRunner


class TokenTracker(BaseCallbackHandler):
    """Callback handler to track token usage."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.step_tokens: list[dict[str, int]] = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Track tokens after LLM call completes."""
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            input_t = usage.get("prompt_tokens", 0)
            output_t = usage.get("completion_tokens", 0)
            self.input_tokens += input_t
            self.output_tokens += output_t
            self.step_tokens.append({"input": input_t, "output": output_t})

    def reset(self):
        """Reset token counts."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.step_tokens = []


class BrowserUseRunner(BaseRunner):
    """Runner using the browser-use library."""

    def __init__(self):
        super().__init__()
        self.browser: Optional[Browser] = None
        self.token_tracker = TokenTracker()

    def _create_llm(self, config: RunConfig):
        """Create the appropriate LLM based on provider."""
        model_config = config.model
        callbacks = [self.token_tracker]

        if model_config.provider == "anthropic":
            return ChatAnthropic(
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
                callbacks=callbacks,
            )
        elif model_config.provider == "openai":
            return ChatOpenAI(
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
                callbacks=callbacks,
            )
        elif model_config.provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_config.model_id,
                max_output_tokens=model_config.max_tokens,
                callbacks=callbacks,
            )
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

    async def run(self, config: RunConfig) -> RunResult:
        """Execute a task using browser-use."""
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        traces = []

        # Reset token tracker for this run
        self.token_tracker.reset()

        try:
            # Create browser with headless config
            self.browser = Browser(headless=config.headless)

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
                    # Get token counts for this step if available
                    step_tokens = (
                        self.token_tracker.step_tokens[i]
                        if i < len(self.token_tracker.step_tokens)
                        else {"input": 0, "output": 0}
                    )

                    trace = StepTrace(
                        step_number=i + 1,
                        timestamp=datetime.now(),
                        screenshot_path=None,
                        page_url=step.state.url if step.state else "",
                        action=step.model_output.model_dump() if step.model_output else {},
                        reasoning=str(step.model_output) if step.model_output else "",
                        input_tokens=step_tokens["input"],
                        output_tokens=step_tokens["output"],
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
            total_input = self.token_tracker.input_tokens
            total_output = self.token_tracker.output_tokens

            return RunResult(
                run_id=run_id,
                task_id=config.task.id,
                model_id=config.model.id,
                library="browser-use",
                status=status,
                success=success,
                steps=len(traces),
                total_time_seconds=total_time,
                input_tokens=total_input,
                output_tokens=total_output,
                cost_usd=self._calculate_cost(total_input, total_output, config.model),
                traces=traces,
            )

        except Exception as e:
            total_time = time.time() - start_time
            total_input = self.token_tracker.input_tokens
            total_output = self.token_tracker.output_tokens

            return RunResult(
                run_id=run_id,
                task_id=config.task.id,
                model_id=config.model.id,
                library="browser-use",
                status=TaskStatus.FAILED,
                success=False,
                steps=len(traces),
                total_time_seconds=total_time,
                input_tokens=total_input,
                output_tokens=total_output,
                cost_usd=self._calculate_cost(total_input, total_output, config.model),
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

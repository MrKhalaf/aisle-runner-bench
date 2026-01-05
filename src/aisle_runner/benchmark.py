"""Benchmark orchestrator for running evaluations across models."""

import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .config import DEFAULT_MODELS, get_model_by_id
from .models import BenchmarkResult, ModelConfig, RunConfig, RunResult, Task
from .runners.browser_use import BrowserUseRunner
from .tasks import get_task, list_tasks

console = Console()


class BenchmarkOrchestrator:
    """Orchestrates benchmark runs across multiple models and tasks."""

    def __init__(
        self,
        output_dir: Path | None = None,
        headless: bool = True,
        on_run_complete: Callable[[RunResult], None] | None = None,
    ):
        self.output_dir = output_dir or Path("results")
        self.headless = headless
        self.on_run_complete = on_run_complete
        self.results: list[RunResult] = []

    def _check_api_keys(self, models: list[ModelConfig]) -> dict[str, bool]:
        """Check which API keys are available."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        providers = set(m.provider for m in models)
        return {p: bool(os.environ.get(key_map.get(p, ""))) for p in providers}

    def _filter_available_models(self, models: list[ModelConfig]) -> list[ModelConfig]:
        """Filter models to only those with available API keys."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        return [m for m in models if os.environ.get(key_map.get(m.provider, ""))]

    async def run_single(
        self,
        task: Task,
        model: ModelConfig,
        runs: int = 1,
    ) -> list[RunResult]:
        """Run a single task with a single model multiple times."""
        results = []
        runner = BrowserUseRunner()

        config = RunConfig(
            task=task,
            model=model,
            library="browser-use",
            headless=self.headless,
        )

        for run_num in range(runs):
            console.print(f"  Run {run_num + 1}/{runs}...", style="dim")
            result = await runner.run(config)
            results.append(result)

            if self.on_run_complete:
                self.on_run_complete(result)

        return results

    async def run_task_across_models(
        self,
        task: Task,
        models: list[ModelConfig] | None = None,
        runs_per_model: int = 1,
    ) -> dict[str, BenchmarkResult]:
        """Run a single task across multiple models."""
        models = models or DEFAULT_MODELS
        available_models = self._filter_available_models(models)

        if not available_models:
            console.print("[red]No models available - check API keys![/red]")
            return {}

        # Report skipped models
        skipped = len(models) - len(available_models)
        if skipped > 0:
            console.print(f"[yellow]Skipping {skipped} models (missing API keys)[/yellow]")

        results: dict[str, BenchmarkResult] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_progress = progress.add_task(
                f"Running {task.name}...",
                total=len(available_models),
            )

            for model in available_models:
                progress.update(task_progress, description=f"Testing {model.id}...")

                run_results = await self.run_single(task, model, runs_per_model)
                self.results.extend(run_results)

                results[model.id] = BenchmarkResult(
                    task_id=task.id,
                    model_id=model.id,
                    library="browser-use",
                    runs=run_results,
                )

                progress.advance(task_progress)

        return results

    async def run_all(
        self,
        task_ids: list[str] | None = None,
        model_ids: list[str] | None = None,
        runs_per_model: int = 1,
    ) -> dict[str, dict[str, BenchmarkResult]]:
        """Run multiple tasks across multiple models."""
        # Get tasks
        if task_ids:
            tasks = [get_task(tid) for tid in task_ids if get_task(tid)]
        else:
            tasks = list_tasks()

        if not tasks:
            console.print("[red]No tasks found![/red]")
            return {}

        # Get models
        if model_ids:
            models = [get_model_by_id(mid) for mid in model_ids if get_model_by_id(mid)]
        else:
            models = DEFAULT_MODELS

        if not models:
            console.print("[red]No models found![/red]")
            return {}

        # Check API keys
        key_status = self._check_api_keys(models)
        console.print("\n[bold]API Key Status:[/bold]")
        for provider, available in key_status.items():
            status = "[green]OK[/green]" if available else "[red]MISSING[/red]"
            console.print(f"  {provider}: {status}")

        available_models = self._filter_available_models(models)
        if not available_models:
            console.print("\n[red]No models available![/red]")
            return {}

        console.print(f"\n[bold]Benchmark Configuration:[/bold]")
        console.print(f"  Tasks: {len(tasks)}")
        console.print(f"  Models: {len(available_models)}")
        console.print(f"  Runs per model: {runs_per_model}")
        console.print(f"  Total runs: {len(tasks) * len(available_models) * runs_per_model}")
        console.print()

        all_results: dict[str, dict[str, BenchmarkResult]] = {}

        for task in tasks:
            console.print(f"\n[bold cyan]Task: {task.name}[/bold cyan]")
            task_results = await self.run_task_across_models(
                task, available_models, runs_per_model
            )
            all_results[task.id] = task_results

        return all_results

    def display_results(self, results: dict[str, dict[str, BenchmarkResult]]):
        """Display benchmark results in a table."""
        for task_id, model_results in results.items():
            table = Table(title=f"Results: {task_id}")
            table.add_column("Model", style="cyan")
            table.add_column("Success Rate", justify="right", style="green")
            table.add_column("Avg Steps", justify="right")
            table.add_column("Avg Time (s)", justify="right")
            table.add_column("Avg Cost", justify="right", style="yellow")
            table.add_column("Runs", justify="right", style="dim")

            for model_id, benchmark in model_results.items():
                success_rate = f"{benchmark.success_rate * 100:.0f}%"
                avg_steps = f"{benchmark.avg_steps:.1f}" if benchmark.avg_steps else "-"
                avg_time = f"{benchmark.avg_time:.1f}" if benchmark.avg_time else "-"
                avg_cost = f"${benchmark.avg_cost:.4f}" if benchmark.avg_cost else "-"

                table.add_row(
                    model_id,
                    success_rate,
                    avg_steps,
                    avg_time,
                    avg_cost,
                    str(len(benchmark.runs)),
                )

            console.print(table)
            console.print()

    def save_results(self, results: dict[str, dict[str, BenchmarkResult]]):
        """Save results to JSON files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = self.output_dir / f"benchmark_{timestamp}.json"
        detailed_data = {}

        for task_id, model_results in results.items():
            detailed_data[task_id] = {}
            for model_id, benchmark in model_results.items():
                detailed_data[task_id][model_id] = {
                    "success_rate": benchmark.success_rate,
                    "avg_steps": benchmark.avg_steps,
                    "avg_time": benchmark.avg_time,
                    "avg_cost": benchmark.avg_cost,
                    "runs": [
                        {
                            "run_id": r.run_id,
                            "success": r.success,
                            "steps": r.steps,
                            "time": r.total_time_seconds,
                            "cost": r.cost_usd,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "error": r.error,
                        }
                        for r in benchmark.runs
                    ],
                }

        with open(detailed_file, "w") as f:
            json.dump(detailed_data, f, indent=2)

        console.print(f"[green]Results saved to:[/green] {detailed_file}")

        # Save summary CSV
        summary_file = self.output_dir / f"summary_{timestamp}.csv"
        with open(summary_file, "w") as f:
            f.write("task_id,model_id,success_rate,avg_steps,avg_time,avg_cost,runs\n")
            for task_id, model_results in results.items():
                for model_id, benchmark in model_results.items():
                    f.write(
                        f"{task_id},{model_id},"
                        f"{benchmark.success_rate:.2f},"
                        f"{benchmark.avg_steps:.1f},"
                        f"{benchmark.avg_time:.1f},"
                        f"{benchmark.avg_cost:.4f},"
                        f"{len(benchmark.runs)}\n"
                    )

        console.print(f"[green]Summary saved to:[/green] {summary_file}")


async def run_benchmark(
    task_ids: list[str] | None = None,
    model_ids: list[str] | None = None,
    runs: int = 1,
    headless: bool = True,
    output_dir: Path | None = None,
    save: bool = True,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Convenience function to run benchmarks."""
    orchestrator = BenchmarkOrchestrator(
        output_dir=output_dir,
        headless=headless,
    )

    results = await orchestrator.run_all(
        task_ids=task_ids,
        model_ids=model_ids,
        runs_per_model=runs,
    )

    orchestrator.display_results(results)

    if save and results:
        orchestrator.save_results(results)

    return results

"""CLI for Aisle Runner Bench."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .benchmark import BenchmarkOrchestrator, run_benchmark
from .config import DEFAULT_MODELS, BUDGET_MODELS, PREMIUM_MODELS, get_model_by_id
from .models import RunConfig
from .runners.browser_use import BrowserUseRunner
from .tasks import TASKS, get_task, list_tasks as get_all_tasks

app = typer.Typer(help="Aisle Runner Bench - Browser automation benchmarks")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task ID to run"),
    model: str = typer.Option("claude-sonnet-4", help="Model to use"),
    runs: int = typer.Option(1, help="Number of runs"),
    headless: bool = typer.Option(True, help="Run headless"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
    save: bool = typer.Option(True, help="Save results to file"),
):
    """Run a benchmark task with a single model."""
    # Get task
    task_obj = get_task(task)
    if not task_obj:
        console.print(f"[red]Task not found:[/red] {task}")
        console.print("\nAvailable tasks:")
        for t in get_all_tasks():
            console.print(f"  [cyan]{t.id}[/cyan]: {t.name}")
        raise typer.Exit(1)

    # Get model
    model_config = get_model_by_id(model)
    if not model_config:
        console.print(f"[red]Model not found:[/red] {model}")
        console.print("\nAvailable models:")
        for m in DEFAULT_MODELS:
            console.print(f"  [cyan]{m.id}[/cyan] ({m.provider})")
        raise typer.Exit(1)

    # Check API key
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(model_config.provider)
    if not env_var or not os.environ.get(env_var):
        console.print(f"[red]Missing API key:[/red] {env_var}")
        console.print(f"\nSet it with: export {env_var}='your-key'")
        raise typer.Exit(1)

    console.print(f"[bold]Task:[/bold] {task_obj.name}")
    console.print(f"[bold]Model:[/bold] {model_config.id} ({model_config.provider})")
    console.print(f"[bold]Runs:[/bold] {runs}")
    console.print(f"[bold]Headless:[/bold] {headless}")
    console.print()

    # Run benchmark
    async def _run():
        return await run_benchmark(
            task_ids=[task],
            model_ids=[model],
            runs=runs,
            headless=headless,
            output_dir=output,
            save=save,
        )

    asyncio.run(_run())


@app.command()
def run_all(
    task: str = typer.Argument(..., help="Task ID to run"),
    subset: str = typer.Option("all", help="Model subset: all, budget, premium"),
    providers: Optional[str] = typer.Option(None, help="Comma-separated providers"),
    runs: int = typer.Option(1, help="Number of runs per model"),
    headless: bool = typer.Option(True, help="Run headless"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
    save: bool = typer.Option(True, help="Save results to file"),
):
    """Run benchmark against all default models."""
    # Get task
    task_obj = get_task(task)
    if not task_obj:
        console.print(f"[red]Task not found:[/red] {task}")
        console.print("\nAvailable tasks:")
        for t in get_all_tasks():
            console.print(f"  [cyan]{t.id}[/cyan]: {t.name}")
        raise typer.Exit(1)

    # Get models based on subset
    if subset == "budget":
        models = BUDGET_MODELS
    elif subset == "premium":
        models = PREMIUM_MODELS
    else:
        models = DEFAULT_MODELS

    # Filter by providers if specified
    if providers:
        provider_list = [p.strip() for p in providers.split(",")]
        models = [m for m in models if m.provider in provider_list]

    if not models:
        console.print("[red]No models selected![/red]")
        raise typer.Exit(1)

    model_ids = [m.id for m in models]

    console.print(f"[bold]Task:[/bold] {task_obj.name}")
    console.print(f"[bold]Models:[/bold] {len(models)}")
    console.print(f"[bold]Runs per model:[/bold] {runs}")
    console.print()

    # Run benchmarks
    async def _run():
        return await run_benchmark(
            task_ids=[task],
            model_ids=model_ids,
            runs=runs,
            headless=headless,
            output_dir=output,
            save=save,
        )

    asyncio.run(_run())


@app.command()
def benchmark(
    tasks: Optional[str] = typer.Option(None, help="Comma-separated task IDs (all if not specified)"),
    models: Optional[str] = typer.Option(None, help="Comma-separated model IDs (all if not specified)"),
    subset: str = typer.Option("all", help="Model subset if --models not specified: all, budget, premium"),
    runs: int = typer.Option(1, help="Number of runs per model"),
    headless: bool = typer.Option(True, help="Run headless"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
    save: bool = typer.Option(True, help="Save results to file"),
):
    """Run full benchmark suite across multiple tasks and models."""
    # Parse task IDs
    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None

    # Parse model IDs
    if models:
        model_ids = [m.strip() for m in models.split(",")]
    elif subset == "budget":
        model_ids = [m.id for m in BUDGET_MODELS]
    elif subset == "premium":
        model_ids = [m.id for m in PREMIUM_MODELS]
    else:
        model_ids = None  # Use all

    console.print("[bold blue]Aisle Runner Bench - Full Benchmark Suite[/bold blue]")
    console.print()

    # Run benchmarks
    async def _run():
        return await run_benchmark(
            task_ids=task_ids,
            model_ids=model_ids,
            runs=runs,
            headless=headless,
            output_dir=output,
            save=save,
        )

    asyncio.run(_run())


@app.command()
def list_tasks_cmd():
    """List available tasks."""
    console.print("[bold]Available Tasks:[/bold]")
    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Tags", style="dim")
    table.add_column("Max Steps", justify="right")
    table.add_column("Timeout", justify="right")

    for task in get_all_tasks():
        table.add_row(
            task.id,
            task.name,
            ", ".join(task.tags),
            str(task.max_steps),
            f"{task.timeout_seconds}s",
        )

    console.print(table)


@app.command()
def list_models(
    provider: Optional[str] = typer.Option(None, help="Filter by provider"),
    subset: str = typer.Option("all", help="Model subset: all, budget, premium"),
):
    """List available models."""
    if subset == "budget":
        models = BUDGET_MODELS
    elif subset == "premium":
        models = PREMIUM_MODELS
    else:
        models = DEFAULT_MODELS

    if provider:
        models = [m for m in models if m.provider == provider]

    console.print("[bold]Available Models:[/bold]")
    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Model ID", style="dim")
    table.add_column("Input/1M", justify="right")
    table.add_column("Output/1M", justify="right")
    table.add_column("Vision")

    for m in models:
        vision = "[green]Yes[/green]" if m.supports_vision else "[red]No[/red]"
        table.add_row(
            m.id,
            m.provider,
            m.model_id,
            f"${m.cost_per_1k_input * 1000:.2f}",
            f"${m.cost_per_1k_output * 1000:.2f}",
            vision,
        )

    console.print(table)


@app.command()
def check_keys():
    """Check API key availability."""
    keys = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT)"),
        ("GOOGLE_API_KEY", "Google (Gemini)"),
    ]

    console.print("[bold]API Key Status:[/bold]")
    all_set = True
    for env_var, provider in keys:
        value = os.environ.get(env_var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            console.print(f"  [green]{provider}[/green]: {masked}")
        else:
            console.print(f"  [red]{provider}[/red]: Not set")
            all_set = False

    if not all_set:
        console.print("\n[dim]Set keys with: export KEY_NAME='your-api-key'[/dim]")
    else:
        console.print("\n[green]All API keys configured![/green]")


@app.command()
def compare(
    results_dir: Path = typer.Argument(..., help="Results directory"),
):
    """Compare benchmark results from a results directory."""
    import json

    if not results_dir.exists():
        console.print(f"[red]Directory not found:[/red] {results_dir}")
        raise typer.Exit(1)

    # Find JSON result files
    json_files = list(results_dir.glob("benchmark_*.json"))
    if not json_files:
        console.print(f"[yellow]No benchmark results found in {results_dir}[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Found {len(json_files)} result file(s)[/bold]")

    for json_file in sorted(json_files):
        console.print(f"\n[bold cyan]Results: {json_file.name}[/bold cyan]")

        with open(json_file) as f:
            data = json.load(f)

        for task_id, model_results in data.items():
            table = Table(title=f"Task: {task_id}")
            table.add_column("Model", style="cyan")
            table.add_column("Success Rate", justify="right", style="green")
            table.add_column("Avg Steps", justify="right")
            table.add_column("Avg Time (s)", justify="right")
            table.add_column("Avg Cost", justify="right", style="yellow")

            for model_id, results in model_results.items():
                success_rate = f"{results['success_rate'] * 100:.0f}%"
                avg_steps = f"{results['avg_steps']:.1f}" if results['avg_steps'] else "-"
                avg_time = f"{results['avg_time']:.1f}" if results['avg_time'] else "-"
                avg_cost = f"${results['avg_cost']:.4f}" if results['avg_cost'] else "-"

                table.add_row(model_id, success_rate, avg_steps, avg_time, avg_cost)

            console.print(table)


# Alias for list-tasks
app.command(name="tasks")(list_tasks_cmd)
app.command(name="models")(list_models)


if __name__ == "__main__":
    app()

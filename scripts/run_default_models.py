#!/usr/bin/env python3
"""Run benchmarks against all default models.

Usage:
    python scripts/run_default_models.py --task sysco-login
    python scripts/run_default_models.py --task sysco-login --subset budget
    python scripts/run_default_models.py --task sysco-login --providers anthropic,openai

Required environment variables:
    ANTHROPIC_API_KEY: For Claude models
    OPENAI_API_KEY: For GPT models
    GOOGLE_API_KEY: For Gemini models
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from aisle_runner.config import (
    DEFAULT_MODELS,
    BUDGET_MODELS,
    PREMIUM_MODELS,
    get_models_by_provider,
)
from aisle_runner.models import ModelConfig


console = Console()


def check_api_keys(models: list[ModelConfig]) -> dict[str, bool]:
    """Check which API keys are available."""
    providers = set(m.provider for m in models)
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    status = {}
    for provider in providers:
        env_var = key_map.get(provider)
        if env_var:
            status[provider] = bool(os.environ.get(env_var))

    return status


def filter_models_by_available_keys(models: list[ModelConfig]) -> list[ModelConfig]:
    """Filter models to only those with available API keys."""
    key_status = check_api_keys(models)
    return [m for m in models if key_status.get(m.provider, False)]


def get_model_subset(subset: str) -> list[ModelConfig]:
    """Get a subset of models based on name."""
    subsets = {
        "all": DEFAULT_MODELS,
        "budget": BUDGET_MODELS,
        "premium": PREMIUM_MODELS,
        "anthropic": get_models_by_provider("anthropic"),
        "openai": get_models_by_provider("openai"),
        "google": get_models_by_provider("google"),
    }
    return subsets.get(subset, DEFAULT_MODELS)


async def run_benchmark(task: str, model: ModelConfig, runs: int, headless: bool) -> dict:
    """Run a benchmark for a single model (placeholder)."""
    # TODO: Implement actual benchmark logic using browser-use
    console.print(f"  [dim]Running {model.id}...[/dim]")

    # Placeholder result
    await asyncio.sleep(0.5)  # Simulate work

    return {
        "model_id": model.id,
        "provider": model.provider,
        "task": task,
        "runs": runs,
        "success_rate": 0.0,
        "avg_steps": 0,
        "avg_time": 0.0,
        "avg_cost": 0.0,
        "status": "not_implemented",
    }


async def run_all_benchmarks(
    task: str,
    models: list[ModelConfig],
    runs: int = 1,
    headless: bool = True,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run benchmarks across all specified models."""
    results = []

    # Check API keys
    key_status = check_api_keys(models)
    available_models = filter_models_by_available_keys(models)

    # Report on API key status
    console.print("\n[bold]API Key Status:[/bold]")
    for provider, available in key_status.items():
        status = "[green]OK[/green]" if available else "[red]MISSING[/red]"
        console.print(f"  {provider}: {status}")

    if not available_models:
        console.print("\n[red]No models available - please set API keys![/red]")
        console.print("\nSet environment variables:")
        console.print("  export ANTHROPIC_API_KEY='your-key'")
        console.print("  export OPENAI_API_KEY='your-key'")
        console.print("  export GOOGLE_API_KEY='your-key'")
        return []

    skipped = len(models) - len(available_models)
    if skipped > 0:
        console.print(f"\n[yellow]Skipping {skipped} models due to missing API keys[/yellow]")

    console.print(f"\n[bold]Running benchmarks for task:[/bold] {task}")
    console.print(f"[bold]Models to test:[/bold] {len(available_models)}")
    console.print(f"[bold]Runs per model:[/bold] {runs}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Running benchmarks...", total=len(available_models))

        for model in available_models:
            progress.update(task_progress, description=f"Testing {model.id}...")
            result = await run_benchmark(task, model, runs, headless)
            results.append(result)
            progress.advance(task_progress)

    return results


def display_results(results: list[dict]):
    """Display benchmark results in a table."""
    if not results:
        return

    table = Table(title="Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="dim")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Steps", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Cost", justify="right", style="yellow")
    table.add_column("Status", style="dim")

    for r in results:
        success = f"{r['success_rate']*100:.0f}%" if r['success_rate'] else "-"
        steps = str(r['avg_steps']) if r['avg_steps'] else "-"
        time = f"{r['avg_time']:.1f}" if r['avg_time'] else "-"
        cost = f"${r['avg_cost']:.4f}" if r['avg_cost'] else "-"

        table.add_row(
            r['model_id'],
            r['provider'],
            success,
            steps,
            time,
            cost,
            r['status'],
        )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks against default models")
    parser.add_argument("--task", required=True, help="Task ID to benchmark")
    parser.add_argument("--subset", default="all",
                       choices=["all", "budget", "premium", "anthropic", "openai", "google"],
                       help="Model subset to use")
    parser.add_argument("--providers", help="Comma-separated list of providers to include")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per model")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show browser")
    parser.add_argument("--output", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Get models based on subset
    models = get_model_subset(args.subset)

    # Filter by providers if specified
    if args.providers:
        providers = [p.strip() for p in args.providers.split(",")]
        models = [m for m in models if m.provider in providers]

    if not models:
        console.print("[red]No models selected![/red]")
        sys.exit(1)

    console.print("[bold blue]Aisle Runner Bench - Default Models[/bold blue]")
    console.print(f"Selected {len(models)} models from subset '{args.subset}'")

    # Run benchmarks
    results = asyncio.run(run_all_benchmarks(
        task=args.task,
        models=models,
        runs=args.runs,
        headless=args.headless,
        output_dir=args.output,
    ))

    # Display results
    display_results(results)

    # Save results if output specified
    if args.output and results:
        args.output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output / f"results_{args.task}_{timestamp}.json"

        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to:[/green] {output_file}")


if __name__ == "__main__":
    main()

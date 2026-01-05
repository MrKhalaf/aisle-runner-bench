#!/usr/bin/env python3
"""Run benchmarks against all default models.

Usage:
    python scripts/run_default_models.py --task google-search
    python scripts/run_default_models.py --task google-search --subset budget
    python scripts/run_default_models.py --task google-search --providers anthropic,openai

Required environment variables:
    ANTHROPIC_API_KEY: For Claude models
    OPENAI_API_KEY: For GPT models
    GOOGLE_API_KEY: For Gemini models
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from aisle_runner import (
    run_benchmark,
    DEFAULT_MODELS,
    BUDGET_MODELS,
    PREMIUM_MODELS,
    get_model_by_id,
    list_tasks,
    get_task,
)


console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks against default models")
    parser.add_argument("--task", help="Task ID to benchmark (omit for all tasks)")
    parser.add_argument("--tasks", help="Comma-separated task IDs")
    parser.add_argument("--subset", default="all",
                       choices=["all", "budget", "premium"],
                       help="Model subset to use")
    parser.add_argument("--providers", help="Comma-separated list of providers to include")
    parser.add_argument("--models", help="Comma-separated list of model IDs")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per model")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show browser")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    # Handle list commands
    if args.list_tasks:
        console.print("[bold]Available Tasks:[/bold]")
        for task in list_tasks():
            console.print(f"  [cyan]{task.id}[/cyan]: {task.name}")
            console.print(f"    {task.description[:80]}...")
        return

    if args.list_models:
        console.print("[bold]Available Models:[/bold]")
        for model in DEFAULT_MODELS:
            console.print(f"  [cyan]{model.id}[/cyan] ({model.provider})")
            console.print(f"    API ID: {model.model_id}")
        return

    # Get task IDs
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
    elif args.task:
        task_ids = [args.task]
    else:
        task_ids = None  # Run all tasks

    # Validate tasks if specified
    if task_ids:
        for tid in task_ids:
            if not get_task(tid):
                console.print(f"[red]Task not found: {tid}[/red]")
                console.print("\nAvailable tasks:")
                for task in list_tasks():
                    console.print(f"  [cyan]{task.id}[/cyan]: {task.name}")
                sys.exit(1)

    # Get model IDs
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        # Get models based on subset
        if args.subset == "budget":
            models = BUDGET_MODELS
        elif args.subset == "premium":
            models = PREMIUM_MODELS
        else:
            models = DEFAULT_MODELS

        # Filter by providers if specified
        if args.providers:
            providers = [p.strip() for p in args.providers.split(",")]
            models = [m for m in models if m.provider in providers]

        model_ids = [m.id for m in models]

    if not model_ids:
        console.print("[red]No models selected![/red]")
        sys.exit(1)

    console.print("[bold blue]Aisle Runner Bench - Default Models Benchmark[/bold blue]")
    console.print()

    # Run benchmarks
    results = asyncio.run(run_benchmark(
        task_ids=task_ids,
        model_ids=model_ids,
        runs=args.runs,
        headless=args.headless,
        output_dir=args.output,
        save=not args.no_save,
    ))

    if results:
        console.print("\n[green]Benchmark complete![/green]")
    else:
        console.print("\n[yellow]No results - check API keys and task/model configuration[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()

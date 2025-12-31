"""CLI for PrepAI Eval."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="PrepAI Eval - Browser automation benchmarks")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task ID to run"),
    model: str = typer.Option("claude-sonnet-4", help="Model to use"),
    library: str = typer.Option("browser-use", help="Library to use"),
    runs: int = typer.Option(1, help="Number of runs"),
    headless: bool = typer.Option(True, help="Run headless"),
    output: Optional[Path] = typer.Option(None, help="Output file"),
):
    """Run a benchmark task."""
    console.print(f"[bold]Running task:[/bold] {task}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Library:[/bold] {library}")
    console.print(f"[bold]Runs:[/bold] {runs}")

    # TODO: Implement actual run logic
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command()
def compare(
    results_dir: Path = typer.Argument(..., help="Results directory"),
    output: Optional[Path] = typer.Option(None, help="Output file"),
):
    """Compare benchmark results."""
    console.print(f"[bold]Comparing results from:[/bold] {results_dir}")

    # TODO: Load and compare results
    table = Table(title="Benchmark Comparison")
    table.add_column("Config", style="cyan")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Steps", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Cost", justify="right", style="yellow")

    # Placeholder data
    table.add_row("opus/browser-use", "95%", "24", "142", "$0.89")
    table.add_row("sonnet/browser-use", "88%", "31", "98", "$0.23")
    table.add_row("haiku/browser-use", "72%", "45", "67", "$0.08")

    console.print(table)


@app.command()
def list_tasks():
    """List available tasks."""
    console.print("[bold]Available Tasks:[/bold]")
    tasks = [
        ("sysco-login", "Log into Sysco website"),
        ("sysco-search", "Search for a product"),
        ("sysco-add-to-cart", "Add a product to cart"),
        ("sysco-full-order", "Complete order flow"),
    ]
    for task_id, desc in tasks:
        console.print(f"  [cyan]{task_id}[/cyan]: {desc}")


@app.command()
def list_models():
    """List available models."""
    console.print("[bold]Available Models:[/bold]")
    models = [
        ("claude-opus-4-5", "anthropic", "$0.015/$0.075"),
        ("claude-sonnet-4", "anthropic", "$0.003/$0.015"),
        ("claude-haiku-35", "anthropic", "$0.001/$0.005"),
        ("gpt-4o", "openai", "$0.005/$0.015"),
    ]
    for model_id, provider, cost in models:
        console.print(f"  [cyan]{model_id}[/cyan] ({provider}): {cost} per 1k tokens")


if __name__ == "__main__":
    app()

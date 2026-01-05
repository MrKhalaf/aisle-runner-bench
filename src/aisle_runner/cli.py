"""CLI for Aisle Runner Bench."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_MODELS, BUDGET_MODELS, PREMIUM_MODELS, get_model_by_id

app = typer.Typer(help="Aisle Runner Bench - Browser automation benchmarks")
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
    table.add_column("Input/1k", justify="right")
    table.add_column("Output/1k", justify="right")
    table.add_column("Vision")

    for m in models:
        vision = "[green]Yes[/green]" if m.supports_vision else "[red]No[/red]"
        table.add_row(
            m.id,
            m.provider,
            m.model_id,
            f"${m.cost_per_1k_input:.4f}",
            f"${m.cost_per_1k_output:.4f}",
            vision,
        )

    console.print(table)


@app.command()
def run_all(
    task: str = typer.Argument(..., help="Task ID to run"),
    subset: str = typer.Option("all", help="Model subset: all, budget, premium"),
    providers: Optional[str] = typer.Option(None, help="Comma-separated providers"),
    runs: int = typer.Option(1, help="Number of runs per model"),
    headless: bool = typer.Option(True, help="Run headless"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Run benchmark against all default models."""
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

    # Check API keys
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    console.print("[bold]API Key Status:[/bold]")
    for provider in set(m.provider for m in models):
        env_var = key_map.get(provider, "UNKNOWN")
        available = bool(os.environ.get(env_var))
        status = "[green]OK[/green]" if available else "[red]MISSING[/red]"
        console.print(f"  {provider} ({env_var}): {status}")

    # Filter to available models
    available_models = [
        m for m in models
        if os.environ.get(key_map.get(m.provider, ""))
    ]

    if not available_models:
        console.print("\n[red]No models available - please set API keys![/red]")
        console.print("\nRequired environment variables:")
        console.print("  export ANTHROPIC_API_KEY='your-key'")
        console.print("  export OPENAI_API_KEY='your-key'")
        console.print("  export GOOGLE_API_KEY='your-key'")
        raise typer.Exit(1)

    console.print(f"\n[bold]Running task:[/bold] {task}")
    console.print(f"[bold]Models:[/bold] {len(available_models)}")
    console.print(f"[bold]Runs per model:[/bold] {runs}")

    for model in available_models:
        console.print(f"\n[cyan]Testing {model.id}...[/cyan]")
        # TODO: Implement actual run logic
        console.print("[yellow]  Not yet implemented[/yellow]")

    console.print("\n[green]Benchmark complete![/green]")


@app.command()
def check_keys():
    """Check API key availability."""
    keys = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT)"),
        ("GOOGLE_API_KEY", "Google (Gemini)"),
    ]

    console.print("[bold]API Key Status:[/bold]")
    for env_var, provider in keys:
        value = os.environ.get(env_var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            console.print(f"  [green]{provider}[/green]: {masked}")
        else:
            console.print(f"  [red]{provider}[/red]: Not set")

    console.print("\n[dim]Set keys with: export KEY_NAME='your-api-key'[/dim]")


if __name__ == "__main__":
    app()

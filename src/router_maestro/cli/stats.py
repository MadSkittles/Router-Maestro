"""Token usage statistics command."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from router_maestro.cli.client import AdminClientError, get_admin_client

console = Console()


def stats(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    model: str = typer.Option(None, "--model", "-m", help="Filter by model"),
) -> None:
    """Show token usage statistics."""
    client = get_admin_client()

    try:
        data = asyncio.run(client.get_stats(days=days, provider=provider, model=model))
    except AdminClientError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to get stats: {e}[/red]")
        raise typer.Exit(1)

    if data.get("total_requests", 0) == 0:
        console.print("[dim]No usage data available.[/dim]")
        return

    # Summary table
    console.print(f"\n[bold]Token Usage Summary (Last {days} Days)[/bold]\n")

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total Requests", f"{data.get('total_requests', 0):,}")
    summary_table.add_row("Total Tokens", f"{data.get('total_tokens', 0):,}")
    summary_table.add_row("  Prompt", f"{data.get('prompt_tokens', 0):,}")
    summary_table.add_row("  Completion", f"{data.get('completion_tokens', 0):,}")

    console.print(summary_table)

    # By model table
    by_model = data.get("by_model", {})
    if by_model:
        console.print("\n[bold]Usage by Model[/bold]\n")

        model_table = Table()
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Provider", style="magenta")
        model_table.add_column("Requests", justify="right")
        model_table.add_column("Total Tokens", justify="right", style="green")
        model_table.add_column("Avg Latency", justify="right")

        for model_key, record in by_model.items():
            parts = model_key.split("/", 1)
            provider_name = parts[0] if len(parts) > 1 else "-"
            model_name = parts[1] if len(parts) > 1 else model_key

            avg_latency = record.get("avg_latency_ms")
            latency = f"{avg_latency:.0f} ms" if avg_latency else "-"
            model_table.add_row(
                model_name,
                provider_name,
                f"{record.get('request_count', 0):,}",
                f"{record.get('total_tokens', 0):,}",
                latency,
            )

        console.print(model_table)

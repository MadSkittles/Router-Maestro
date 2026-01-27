"""Terminal heatmap generation for token usage."""

from datetime import datetime, timedelta

import plotext as plt
from rich.console import Console
from rich.table import Table

from router_maestro.stats.storage import StatsStorage

console = Console()


def generate_heatmap(days: int = 7, provider: str | None = None, model: str | None = None) -> None:
    """Generate and display a terminal heatmap of token usage.

    Args:
        days: Number of days to show
        provider: Filter by provider (optional)
        model: Filter by model (optional)
    """
    storage = StatsStorage()
    hourly_data = storage.get_usage_by_hour(days=days, provider=provider, model=model)

    if not hourly_data:
        console.print("[dim]No usage data available for the specified period.[/dim]")
        return

    # Build a matrix for the heatmap: rows = days, columns = hours
    # Initialize with zeros
    today = datetime.now().date()
    dates = [(today - timedelta(days=i)) for i in range(days - 1, -1, -1)]

    # Create a 2D matrix (days x 24 hours)
    matrix = [[0 for _ in range(24)] for _ in range(len(dates))]
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for record in hourly_data:
        date = datetime.fromisoformat(record["date"]).date()
        hour = record["hour"]
        tokens = record["total_tokens"]

        if date in date_to_idx:
            matrix[date_to_idx[date]][hour] = tokens

    # Display using plotext
    plt.clear_figure()
    plt.title(f"Token Usage Heatmap (Last {days} Days)")

    # Create a simple bar chart by day since plotext doesn't have heatmap
    daily_data = storage.get_usage_by_day(days=days, provider=provider, model=model)

    if daily_data:
        dates_str = [record["date"] for record in daily_data]
        tokens = [record["total_tokens"] for record in daily_data]

        plt.bar(dates_str, tokens)
        plt.xlabel("Date")
        plt.ylabel("Total Tokens")
        plt.show()

    # Also show a text-based heatmap using Rich
    _display_text_heatmap(dates, matrix)


def _display_text_heatmap(dates: list, matrix: list[list[int]]) -> None:
    """Display a text-based heatmap using Rich.

    Args:
        dates: List of dates
        matrix: 2D matrix of token counts (days x hours)
    """
    console.print("\n[bold]Hourly Activity Heatmap:[/bold]")

    # Find max value for scaling
    max_val = max(max(row) for row in matrix) if matrix else 1
    if max_val == 0:
        max_val = 1

    # Create intensity characters
    intensity_chars = " ░▒▓█"

    # Build the heatmap
    hour_labels = "    " + "".join(f"{h:2d}" for h in range(0, 24, 2))
    console.print(f"[dim]{hour_labels}[/dim]")

    for i, date in enumerate(dates):
        row_str = f"{date.strftime('%m/%d')} "
        for h in range(24):
            value = matrix[i][h]
            intensity = int((value / max_val) * (len(intensity_chars) - 1))
            char = intensity_chars[intensity]
            row_str += char
        console.print(row_str)

    # Legend
    console.print(f"\n[dim]Legend: {' '.join(intensity_chars)} (low to high)[/dim]")


def display_stats_summary(days: int = 7) -> None:
    """Display a summary of token usage statistics.

    Args:
        days: Number of days to summarize
    """
    storage = StatsStorage()
    total = storage.get_total_usage(days=days)
    by_model = storage.get_usage_by_model(days=days)

    if not total or total.get("total_tokens") is None:
        console.print("[dim]No usage data available.[/dim]")
        return

    # Summary table
    console.print(f"\n[bold]Token Usage Summary (Last {days} Days)[/bold]\n")

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total Requests", f"{total.get('request_count', 0):,}")
    summary_table.add_row("Successful", f"{total.get('success_count', 0):,}")
    summary_table.add_row("Total Tokens", f"{total.get('total_tokens', 0):,}")
    summary_table.add_row("  Prompt", f"{total.get('prompt_tokens', 0):,}")
    summary_table.add_row("  Completion", f"{total.get('completion_tokens', 0):,}")

    if total.get("avg_latency_ms"):
        summary_table.add_row("Avg Latency", f"{total.get('avg_latency_ms', 0):.0f} ms")

    console.print(summary_table)

    # By model table
    if by_model:
        console.print("\n[bold]Usage by Model[/bold]\n")

        model_table = Table()
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Provider", style="magenta")
        model_table.add_column("Requests", justify="right")
        model_table.add_column("Total Tokens", justify="right", style="green")
        model_table.add_column("Avg Latency", justify="right")

        for record in by_model:
            avg_latency = record.get("avg_latency_ms")
            latency = f"{avg_latency:.0f} ms" if avg_latency else "-"
            model_table.add_row(
                record["model"],
                record["provider"],
                f"{record['request_count']:,}",
                f"{record['total_tokens']:,}",
                latency,
            )

        console.print(model_table)

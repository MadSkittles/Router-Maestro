"""Model management commands."""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from router_maestro.cli.client import ServerNotRunningError, get_admin_client
from router_maestro.cli.client_configs.prompts import select_dropdown, supports_dropdowns
from router_maestro.config import AutoCapabilityPolicy, AutoMode, AutoTaskType
from router_maestro.routing.model_ref import qualify_model_id

app = typer.Typer(no_args_is_help=True)
console = Console()

_AUTO_SHIMMER_STOPS = (
    (27, 151, 255),
    (37, 217, 255),
    (160, 241, 255),
    (255, 255, 255),
    (119, 228, 255),
    (74, 132, 255),
)


def _model_key(model: dict) -> str:
    """Return one provider-qualified model key without duplicating its prefix."""
    if model.get("virtual") is True:
        return model["id"]
    return qualify_model_id(model["provider"], model["id"])


def _format_token_count(tokens: int) -> str:
    if tokens >= 1_000_000:
        value = int(tokens / 100_000) / 10
        return f"{value:g}M"
    if tokens > 900_000:
        return "1M"
    if tokens >= 1_000:
        return f"{round(tokens / 1_000)}K"
    return str(tokens)


def _context_windows_label(model: dict) -> str:
    limits: list[int] = []
    options = model.get("context_window_options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            value = option.get("max_prompt_tokens")
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                and value not in limits
            ):
                limits.append(value)
    if not limits:
        fallback = model.get("max_context_window_tokens") or model.get("max_prompt_tokens")
        if isinstance(fallback, int) and not isinstance(fallback, bool) and fallback > 0:
            limits.append(fallback)
    return " / ".join(_format_token_count(limit) for limit in limits) if limits else "unknown"


def _auto_shimmer_text(value: str) -> Text:
    """Render Auto branding as a terminal-safe cyan-to-blue light sweep."""
    result = Text()
    last_index = max(len(value) - 1, 1)
    last_stop = len(_AUTO_SHIMMER_STOPS) - 1
    for index, character in enumerate(value):
        position = (index / last_index) * last_stop
        stop_index = min(int(position), last_stop - 1)
        blend = position - stop_index
        start = _AUTO_SHIMMER_STOPS[stop_index]
        end = _AUTO_SHIMMER_STOPS[stop_index + 1]
        red, green, blue = (
            round(start[channel] + (end[channel] - start[channel]) * blend) for channel in range(3)
        )
        result.append(character, style=f"bold italic rgb({red},{green},{blue})")
    return result


def _handle_server_error(e: Exception) -> None:
    """Handle server connection errors."""
    if isinstance(e, ServerNotRunningError):
        console.print(f"[red]{e}[/red]")
    else:
        console.print(f"[red]Error: {e}[/red]")
    raise typer.Exit(1)


def _select_catalog_model(models: list[dict], prompt: str, *, default: str | None = None) -> str:
    concrete = [model for model in models if model.get("virtual") is not True]
    if not concrete:
        raise ValueError("No concrete models are available")
    choices = [
        (
            f"{_model_key(model)} — {model['name']} — {_context_windows_label(model)}",
            _model_key(model),
        )
        for model in concrete
    ]
    if supports_dropdowns():
        return select_dropdown(prompt, choices, default=default, searchable=True)
    from rich.prompt import Prompt

    for index, (label, _value) in enumerate(choices, 1):
        console.print(f"  {index}. {label}")
    default_index = next(
        (str(index) for index, (_label, value) in enumerate(choices, 1) if value == default),
        "1",
    )
    answer = Prompt.ask(
        prompt, choices=[str(i) for i in range(1, len(choices) + 1)], default=default_index
    )
    return choices[int(answer) - 1][1]


auto_app = typer.Typer(no_args_is_help=True, help="Configure the virtual Auto model")
app.add_typer(auto_app, name="auto")


@auto_app.command(name="show")
def auto_show() -> None:
    """Show the active Auto mode and both retained profiles."""
    client = get_admin_client()
    try:
        auto = asyncio.run(client.get_runtime_config()).get("auto", {})
    except Exception as error:
        _handle_server_error(error)
        return
    console.print("\n[bold]Auto Model Configuration[/bold]")
    console.print(f"  Mode: [cyan]{auto.get('mode', AutoMode.TASK_ROUTER.value)}[/cyan]")
    console.print(f"  Capability policy: [cyan]{auto.get('capability_policy', 'strict')}[/cyan]")
    task_router = auto.get("task_router", {})
    console.print(
        f"  Router model: [cyan]{task_router.get('router_model', 'not configured')}[/cyan]"
    )
    for task in AutoTaskType:
        model = task_router.get("task_models", {}).get(task.value, "not configured")
        console.print(f"  {task.value}: [cyan]{model}[/cyan]")
    chain = auto.get("priority_chain", [])
    console.print(f"  Priority chain: [cyan]{' → '.join(chain) if chain else 'empty'}[/cyan]\n")


@auto_app.command(name="configure")
def auto_configure() -> None:
    """Interactively configure Smart Auto or a strict priority fallback chain."""
    client = get_admin_client()
    try:
        data = asyncio.run(client.get_runtime_config())
        models = asyncio.run(client.list_models())
        revision = data.pop("revision")
        auto = dict(data.get("auto", {}))
        current_mode = auto.get("mode", AutoMode.TASK_ROUTER.value)
        mode_choices = [
            ("Smart Auto — classify each request by task", AutoMode.TASK_ROUTER.value),
            ("Priority Chain — use a strict ordered fallback chain", AutoMode.PRIORITY_CHAIN.value),
        ]
        mode = (
            select_dropdown("Auto mode", mode_choices, default=current_mode)
            if supports_dropdowns()
            else typer.prompt(
                "Auto mode (task-router/priority-chain)",
                default=current_mode,
            )
        )
        if mode not in {item.value for item in AutoMode}:
            raise ValueError("Auto mode must be task-router or priority-chain")
        auto["mode"] = mode

        if mode == AutoMode.TASK_ROUTER.value:
            policy_choices = [
                (
                    "Require confirmed support — exclude unknown capabilities",
                    AutoCapabilityPolicy.STRICT.value,
                ),
                (
                    "Allow unknown support — exclude only confirmed incompatibility",
                    AutoCapabilityPolicy.OPTIMISTIC.value,
                ),
            ]
            current_policy = auto.get("capability_policy", AutoCapabilityPolicy.STRICT.value)
            policy = (
                select_dropdown(
                    "Unknown capability handling",
                    policy_choices,
                    default=current_policy,
                )
                if supports_dropdowns()
                else typer.prompt(
                    "Unknown capability handling (strict/optimistic)",
                    default=current_policy,
                )
            )
            if policy not in {item.value for item in AutoCapabilityPolicy}:
                raise ValueError("Capability policy must be strict or optimistic")
            auto["capability_policy"] = policy
            task_router = dict(auto.get("task_router", {}))
            task_router["router_model"] = _select_catalog_model(
                models,
                "Router model",
                default=task_router.get("router_model"),
            )
            current_tasks = dict(task_router.get("task_models", {}))
            task_router["task_models"] = {
                task.value: _select_catalog_model(
                    models,
                    f"{task.value.replace('_', ' ').title()} model",
                    default=current_tasks.get(task.value),
                )
                for task in AutoTaskType
            }
            auto["task_router"] = task_router
        else:
            chain: list[str] = []
            current = list(auto.get("priority_chain", []))
            while True:
                default = current[len(chain)] if len(chain) < len(current) else None
                selected = _select_catalog_model(
                    models,
                    f"Priority {len(chain) + 1}",
                    default=default,
                )
                if selected in chain:
                    console.print("[yellow]That model is already in the chain.[/yellow]")
                    continue
                chain.append(selected)
                if not Confirm.ask("Add another fallback model?", default=len(chain) == 1):
                    break
            auto["priority_chain"] = chain

        data["auto"] = auto
        asyncio.run(client.patch_runtime_config(config=data, revision=revision))
        console.print("[green]Auto model configuration updated[/green]")
    except Exception as error:
        _handle_server_error(error)


@app.command(name="list")
def list_models() -> None:
    """List all available models with their priorities."""
    client = get_admin_client()

    # Get models and priorities
    try:
        models = asyncio.run(client.list_models())
        priorities_data = asyncio.run(client.get_priorities())
        priorities_list = priorities_data.get("priorities", [])
    except Exception as e:
        _handle_server_error(e)
        return

    if not models:
        console.print("[dim]No models available.[/dim]")
        console.print("[dim]Make sure you have authenticated with at least one provider.[/dim]")
        return

    table = Table(title="Available Models")
    table.add_column("Priority", style="cyan", justify="right")
    table.add_column("Model Key", style="green")
    table.add_column("Display Name", style="white")
    table.add_column("Provider", style="magenta")
    table.add_column("Contexts", style="cyan")

    for model in models:
        model_key = _model_key(model)
        is_auto = model.get("virtual") is True
        # Check if this model is in the priority list
        try:
            priority_idx = priorities_list.index(model_key)
            priority_str = str(priority_idx + 1)
        except ValueError:
            priority_str = "-"

        table.add_row(
            priority_str,
            _auto_shimmer_text(model_key) if is_auto else model_key,
            _auto_shimmer_text(model["name"]) if is_auto else model["name"],
            model["provider"],
            _context_windows_label(model),
        )

    console.print(table)


@app.command(name="refresh")
def refresh_models() -> None:
    """Refresh the models cache from all providers."""
    client = get_admin_client()

    console.print("[dim]Refreshing models cache...[/dim]")

    try:
        success = asyncio.run(client.refresh_models())
        if success:
            console.print("[green]Models cache refreshed successfully[/green]")
        else:
            console.print("[red]Failed to refresh models cache[/red]")
            raise typer.Exit(1)
    except Exception as e:
        _handle_server_error(e)


# Priority subcommand group
priority_app = typer.Typer(no_args_is_help=True, help="Manage model priorities")
app.add_typer(priority_app, name="priority")


@priority_app.command(name="list")
def priority_list() -> None:
    """List current model priorities."""
    client = get_admin_client()

    try:
        data = asyncio.run(client.get_priorities())
        priorities = data.get("priorities", [])
    except Exception as e:
        _handle_server_error(e)
        return

    if not priorities:
        console.print("[dim]No priorities configured.[/dim]")
        console.print(
            "[dim]Use 'router-maestro model priority add <provider/model>' to add priorities.[/dim]"
        )
        return

    table = Table(title="Model Priorities")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Model Key", style="green")

    for idx, model_key in enumerate(priorities):
        table.add_row(str(idx + 1), model_key)

    console.print(table)


@priority_app.command(name="add")
def priority_add(
    model_key: Annotated[str, typer.Argument(help="Model key in format 'provider/model'")],
    position: Annotated[
        int | None,
        typer.Option("--position", "-p", help="Position in priority list (1-based)"),
    ] = None,
) -> None:
    """Add or move a model in the priority list."""
    if "/" not in model_key:
        console.print("[red]Model key must be in format 'provider/model'[/red]")
        raise typer.Exit(1)

    client = get_admin_client()

    try:
        data = asyncio.run(client.get_runtime_config())
        revision = data.pop("revision")
        priorities = data.get("priorities", [])

        # Remove if already exists
        if model_key in priorities:
            priorities.remove(model_key)

        # Insert at position
        if position is None:
            priorities.append(model_key)
        else:
            pos = position - 1  # Convert 1-based to 0-based
            priorities.insert(pos, model_key)

        data["priorities"] = priorities
        asyncio.run(client.patch_runtime_config(config=data, revision=revision))

        if position:
            console.print(f"[green]Added '{model_key}' at position {position}[/green]")
        else:
            console.print(f"[green]Added '{model_key}' to end of priority list[/green]")

    except Exception as e:
        _handle_server_error(e)


@priority_app.command(name="remove")
def priority_remove(
    model_key: Annotated[str, typer.Argument(help="Model key in format 'provider/model'")],
) -> None:
    """Remove a model from the priority list."""
    if "/" not in model_key:
        console.print("[red]Model key must be in format 'provider/model'[/red]")
        raise typer.Exit(1)

    client = get_admin_client()

    try:
        data = asyncio.run(client.get_runtime_config())
        revision = data.pop("revision")
        priorities = data.get("priorities", [])

        if model_key in priorities:
            priorities.remove(model_key)
            data["priorities"] = priorities
            asyncio.run(client.patch_runtime_config(config=data, revision=revision))
            console.print(f"[green]Removed '{model_key}' from priority list[/green]")
        else:
            console.print(f"[yellow]'{model_key}' was not in the priority list[/yellow]")

    except Exception as e:
        _handle_server_error(e)


@priority_app.command(name="clear")
def priority_clear() -> None:
    """Clear all priorities."""
    client = get_admin_client()

    try:
        data = asyncio.run(client.get_runtime_config())
        revision = data.pop("revision")
        data["priorities"] = []
        asyncio.run(client.patch_runtime_config(config=data, revision=revision))
        console.print("[green]Cleared all priorities[/green]")
    except Exception as e:
        _handle_server_error(e)


# Fallback subcommand group
fallback_app = typer.Typer(no_args_is_help=True, help="Manage fallback configuration")
app.add_typer(fallback_app, name="fallback")

VALID_STRATEGIES = ["priority", "same-model", "none"]


@fallback_app.command(name="show")
def fallback_show() -> None:
    """Show current fallback configuration."""
    client = get_admin_client()

    try:
        data = asyncio.run(client.get_priorities())
        fallback = data.get("fallback", {})
    except Exception as e:
        _handle_server_error(e)
        return

    strategy = fallback.get("strategy", "priority")
    max_retries = fallback.get("maxRetries", 2)

    console.print()
    console.print("[bold]Fallback Configuration[/bold]")
    console.print(f"  Strategy:    [cyan]{strategy}[/cyan]")
    console.print(f"  Max Retries: [cyan]{max_retries}[/cyan]")
    console.print()


@fallback_app.command(name="set")
def fallback_set(
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", "-s", help="Fallback strategy (priority, same-model, none)"),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", "-r", help="Maximum number of fallback retries (0-10)"),
    ] = None,
) -> None:
    """Set fallback configuration."""
    # Validate that at least one option is provided
    if strategy is None and max_retries is None:
        console.print("[red]At least one of --strategy or --max-retries must be provided[/red]")
        raise typer.Exit(1)

    # Validate strategy
    if strategy is not None and strategy not in VALID_STRATEGIES:
        console.print(f"[red]Invalid strategy '{strategy}'[/red]")
        console.print(f"[dim]Valid strategies: {', '.join(VALID_STRATEGIES)}[/dim]")
        raise typer.Exit(1)

    # Validate max_retries
    if max_retries is not None and (max_retries < 0 or max_retries > 10):
        console.print("[red]max-retries must be between 0 and 10[/red]")
        raise typer.Exit(1)

    client = get_admin_client()

    try:
        data = asyncio.run(client.get_runtime_config())
        revision = data.pop("revision")
        fallback = data.get("fallback", {})

        # Update fallback config
        if strategy is not None:
            fallback["strategy"] = strategy
        if max_retries is not None:
            fallback["maxRetries"] = max_retries

        data["fallback"] = fallback
        asyncio.run(client.patch_runtime_config(config=data, revision=revision))

        console.print("[green]Fallback configuration updated[/green]")

        # Show updated config
        console.print(f"  Strategy:    [cyan]{fallback.get('strategy', 'priority')}[/cyan]")
        console.print(f"  Max Retries: [cyan]{fallback.get('maxRetries', 2)}[/cyan]")

    except Exception as e:
        _handle_server_error(e)

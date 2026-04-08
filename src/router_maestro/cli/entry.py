"""Lightweight CLI entry point that suppresses KeyboardInterrupt tracebacks.

Wraps the heavy import of cli.main (which triggers pydantic, httpx, etc.)
inside a try/except so that Ctrl+C during startup exits cleanly instead of
dumping a traceback.  Once app() is running, Click/Typer's own handler
converts KeyboardInterrupt to Abort and prints "Aborted!" to stderr.
"""

import sys


def cli() -> None:
    """Entry point wrapper — catches KeyboardInterrupt during module import."""
    try:
        from router_maestro.cli.main import app

        app()
    except KeyboardInterrupt:
        sys.exit(130)

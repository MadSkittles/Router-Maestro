# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run the CLI
uv run router-maestro --help

# Start the API server
uv run router-maestro server start --port 8080

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_auth.py -v

# Run a specific test
uv run pytest tests/test_auth.py::TestAuthStorage::test_empty_storage -v

# Lint code
uv run ruff check src/

# Format code
uv run ruff format src/
```

## Architecture Overview

Router-Maestro is a multi-model routing system that exposes both OpenAI-compatible and Anthropic-compatible APIs. It routes requests to various LLM providers (GitHub Copilot, OpenAI, Anthropic, custom) with priority-based routing and fallback support.

### Core Components

**Router (`src/router_maestro/routing/router.py`)**
- Central routing logic that selects providers based on model priorities
- Handles auto-routing when model is `router-maestro`
- Implements fallback to alternative providers on failure
- Caches model availability from all providers

**Providers (`src/router_maestro/providers/`)**
- `BaseProvider` - Abstract base class defining the provider interface
- `CopilotProvider` - GitHub Copilot integration with OAuth
- `OpenAIProvider` - Native OpenAI API
- `AnthropicProvider` - Native Anthropic API
- `OpenAICompatibleProvider` - Custom providers with OpenAI-compatible APIs

**Server (`src/router_maestro/server/`)**
- FastAPI application with two API flavors:
  - OpenAI-compatible: `/v1/chat/completions`, `/v1/models`
  - Anthropic-compatible: `/v1/messages`, `/api/anthropic/v1/messages`
- `translation.py` - Converts between Anthropic and OpenAI request/response formats
- `schemas/` - Pydantic models for both API formats

**CLI (`src/router_maestro/cli/`)**
- Typer-based CLI with subcommands: `server`, `auth`, `model`, `context`, `config`
- Each subcommand in its own module registered in `main.py`

### Data Flow

1. Request arrives at API endpoint (OpenAI or Anthropic format)
2. Anthropic requests are translated to internal OpenAI format
3. Router selects provider based on model key and priorities
4. Provider makes upstream API call
5. Response is translated back if needed (for Anthropic API)

### File Locations

Configuration and data files follow XDG conventions:
- **Config** (`~/.config/router-maestro/`): `providers.json`, `priorities.json`, `contexts.json`
- **Data** (`~/.local/share/router-maestro/`): `auth.json`, `server.json`

### Model Identification

Models are identified by `provider/model-id` format (e.g., `github-copilot/gpt-4o`). The special model name `router-maestro` triggers auto-routing based on priority configuration.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run the CLI
uv run router-maestro --help

# Start the API server (requires ROUTER_MAESTRO_API_KEY env var)
ROUTER_MAESTRO_API_KEY="sk-rm-..." uv run router-maestro server start --port 8080

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_auth.py -v

# Run a specific test
uv run pytest tests/test_auth.py::TestAuthStorage::test_empty_storage -v

# Lint code
uv run ruff check src/

# Format code
uv run ruff format src/ tests/

# Build and push multi-arch Docker image
make build-multiarch
```

### Local Server Startup

The API key is configured via `ROUTER_MAESTRO_API_KEY` env var. For VS Code debugging, see `.vscode/launch.json` which has the key pre-configured. The server runs on uvicorn and requires authenticated GitHub Copilot (via `router-maestro auth login copilot`) for most models.

### Docker Deployment

Production deployment uses `docker-compose.yml` with Traefik reverse proxy. Use `make build-multiarch` to build for both amd64 and arm64 (requires the `multiarch` buildx builder).

## Architecture Overview

Router-Maestro is a multi-model routing system that exposes both OpenAI-compatible and Anthropic-compatible APIs. It routes requests to various LLM providers (GitHub Copilot, OpenAI, Anthropic, custom) with priority-based routing and fallback support.

## Project Structure

```
src/router_maestro/
├── __init__.py              # Package root, exports __version__
├── __main__.py              # Entry point for `python -m router_maestro`
├── config/                  # Configuration loading and models
│   ├── paths.py             # XDG config/data paths (AUTH_FILE, PRIORITIES_FILE, etc.)
│   ├── priorities.py        # PrioritiesConfig, FallbackConfig, ThinkingBudgetConfig
│   └── contexts.py          # Context-based server configuration
├── auth/                    # Authentication management
│   ├── manager.py           # AuthManager — credential CRUD
│   ├── storage.py           # JSON file-based credential storage
│   └── github_oauth.py      # GitHub Copilot OAuth device flow
├── providers/               # LLM provider implementations
│   ├── base.py              # BaseProvider ABC, ChatRequest/ChatResponse/Message dataclasses
│   ├── copilot.py           # CopilotProvider — GitHub Copilot API (primary provider)
│   ├── openai_base.py       # OpenAIChatProvider — shared logic for OpenAI-compatible providers
│   ├── openai.py            # OpenAIProvider — native OpenAI API
│   ├── openai_compat.py     # OpenAICompatibleProvider — custom OpenAI-compatible endpoints
│   ├── anthropic.py         # AnthropicProvider — native Anthropic API
│   └── tool_parsing.py      # XML tool call recovery utility (Copilot quirk workaround)
├── routing/
│   └── router.py            # Router — provider selection, fallback, model resolution
├── server/                  # FastAPI application
│   ├── app.py               # FastAPI app factory, lifespan, route registration
│   ├── middleware/           # Auth middleware (API key verification)
│   ├── routes/
│   │   ├── anthropic.py     # /api/anthropic/v1/messages — Anthropic Messages API proxy
│   │   ├── chat.py          # /api/openai/v1/chat/completions — OpenAI Chat API proxy
│   │   ├── models.py        # /api/openai/v1/models — model listing
│   │   ├── responses.py     # /api/openai/v1/responses — OpenAI Responses API (Codex)
│   │   └── admin.py         # /api/admin/* — auth, config, priorities management
│   ├── schemas/
│   │   ├── openai.py        # Pydantic models for OpenAI API (ChatMessage, ChatCompletionRequest, etc.)
│   │   ├── anthropic.py     # Pydantic models for Anthropic API (AnthropicMessagesRequest, etc.)
│   │   └── responses.py     # Pydantic models for Responses API
│   ├── translation.py       # Bidirectional Anthropic ↔ OpenAI format translation
│   └── streaming.py         # SSE streaming helpers
├── cli/                     # Typer CLI
│   ├── main.py              # CLI app, registers subcommands
│   ├── server.py            # `router-maestro server start/stop`
│   ├── auth.py              # `router-maestro auth login/status`
│   ├── model.py             # `router-maestro model list/info`
│   ├── config.py            # `router-maestro config show/set`
│   ├── context.py           # `router-maestro context` management
│   └── client.py            # HTTP client for CLI → server communication
└── utils/
    ├── logging.py           # Logging setup (get_logger, setup_logging)
    ├── tokens.py            # Token estimation, stop reason mapping
    ├── token_config.py      # Provider-aware token counting config
    ├── cache.py             # TTLCache generic utility
    ├── context_window.py    # Thinking budget resolution logic
    ├── model_match.py       # Fuzzy model ID matching (rapidfuzz)
    └── model_sort.py        # Model list sorting by provider/family/version

tests/                       # pytest test suite (508+ tests)
Makefile                     # Dev commands: test, lint, build, build-multiarch, publish
Dockerfile                   # Multi-stage Docker build
docker-compose.yml           # Production: Traefik + Router-Maestro
docker-compose.dev.yml       # Dev: local source build
```

## Key Concepts

### Provider System

All providers implement `BaseProvider` (defined in `providers/base.py`) which defines:
- `chat_completion(ChatRequest) -> ChatResponse` — non-streaming
- `chat_completion_stream(ChatRequest) -> AsyncIterator[ChatStreamChunk]` — streaming
- `list_models() -> list[ModelInfo]`
- `is_authenticated() -> bool`

The internal format uses OpenAI-style structures (`ChatRequest`, `ChatResponse` with `tool_calls` as `list[dict]`). Anthropic requests are translated to this format in `translation.py`.

### Copilot Provider Quirks

GitHub Copilot API has non-standard behaviors that require special handling:
- **Multi-choice tool calls**: Returns multiple `choices[]` — one with text content and separate ones each with a single tool_call. The provider merges all choices into a single response.
- **XML tool calls (fallback)**: `tool_parsing.py` provides recovery for tool calls embedded as `<tool_call>` XML in content text.
- **Token refresh**: Copilot tokens expire; `ensure_token()` handles automatic refresh via GitHub OAuth.

### API Endpoints

| Endpoint | Format | Handler |
|----------|--------|---------|
| `POST /api/anthropic/v1/messages` | Anthropic Messages API | `routes/anthropic.py` |
| `POST /api/openai/v1/chat/completions` | OpenAI Chat API | `routes/chat.py` |
| `GET /api/openai/v1/models` | OpenAI Models API | `routes/models.py` |
| `POST /api/openai/v1/responses` | OpenAI Responses API | `routes/responses.py` |
| `GET /api/anthropic/v1/models` | Anthropic Models API | `routes/anthropic.py` |
| `POST /api/anthropic/v1/messages/count_tokens` | Token counting | `routes/anthropic.py` |
| `/api/admin/*` | Admin/management | `routes/admin.py` |

### Data Flow

1. Request arrives at API endpoint (OpenAI or Anthropic format)
2. Anthropic requests are translated to internal OpenAI format via `translation.py`
3. Router selects provider based on model key and priorities (`routing/router.py`)
4. Provider makes upstream API call and returns `ChatResponse`
5. Response is translated back if needed (for Anthropic API)
6. Tool calls from all provider choices are merged; XML recovery runs as fallback

### Configuration Files

Runtime configuration follows XDG conventions:
- **Config** (`~/.config/router-maestro/`):
  - `providers.json` — custom provider endpoints
  - `priorities.json` — model routing priorities, fallback strategy, thinking budget config
  - `contexts.json` — context-based server configuration
- **Data** (`~/.local/share/router-maestro/`):
  - `auth.json` — stored credentials (OAuth tokens, API keys)
  - `server.json` — server state

### Model Identification

Models are identified by `provider/model-id` format (e.g., `github-copilot/gpt-4o`). The special model name `router-maestro` triggers auto-routing based on priority configuration. Fuzzy matching (`utils/model_match.py`) handles minor ID variations (e.g., `claude-opus-4-6` → `claude-opus-4.6`).

### Search Tools

When `ripgrep` (`rg`) is available on the system, prefer it over `grep` for faster and more ergonomic code search.

### Branch Workflow

Never commit directly to `master`. Always create a feature branch for changes:

```bash
git checkout -b feat/description   # new feature
git checkout -b fix/description    # bug fix
git checkout -b chore/description  # maintenance
git checkout -b docs/description   # documentation
```

**Important:** Before making any changes, check the current branch with `git branch`. If on `master`, switch to an existing relevant branch or create a new one first. Do not stage, commit, or modify files while on `master`.

After work is complete, open a PR to merge into `master`.

### Pre-Commit Workflow

Run `/lint` and let Codex review the code before committing.

### Version Updates

When releasing a new version, update these files:

1. `pyproject.toml` - `version = "x.x.x"`
2. `src/router_maestro/__init__.py` - `__version__ = "x.x.x"`
3. Run `uv lock` to update `uv.lock`
4. Create git tag: `git tag vx.x.x`

### GitHub Operations

Use the `gh` CLI for GitHub operations like creating PRs, issues, and reviews.

```bash
# Create a pull request
gh pr create --title "Title" --body "Description"

# View PR details
gh pr view <number>

# Merge a PR
gh pr merge <number>

# Create an issue
gh issue create --title "Title" --body "Description"

# List issues
gh issue list
```

# AGENTS.md

This file gives coding agents instructions for working in this repository.
It applies to the entire `Router-Maestro` tree unless a more specific
`AGENTS.md` exists in a subdirectory.

## Project Summary

Router-Maestro is a Python 3.11+ multi-model routing proxy. It exposes
OpenAI-compatible, Anthropic-compatible, and Gemini-compatible APIs and routes
requests across GitHub Copilot, OpenAI, Anthropic, and custom OpenAI-compatible
providers with priority-based fallback.

The package uses:

- `uv` for dependency management and command execution
- FastAPI and uvicorn for the API server
- Typer and Rich for the CLI
- Pydantic v2 for schemas and config models
- pytest for tests
- Ruff for linting and formatting

## Required Workflow

- Before making code or documentation changes, run `git branch --show-current`.
- Do not edit, stage, or commit while on `master`.
- If the current branch is `master`, create or switch to a relevant branch first:

```bash
git checkout -b feat/description
git checkout -b fix/description
git checkout -b chore/description
git checkout -b docs/description
```

- Do not revert user changes. If the worktree is dirty, preserve unrelated
  changes and work around them.
- Do not commit unless the user explicitly asks for a commit.
- Use `rg` instead of `grep` for repository searches when available.
- Prefer existing project patterns over new abstractions.
- Keep changes scoped to the requested behavior.

## Common Commands

Install dependencies:

```bash
uv pip install -e ".[dev]"
```

Run the CLI:

```bash
uv run router-maestro --help
```

Start the API server:

```bash
ROUTER_MAESTRO_API_KEY="sk-rm-..." uv run router-maestro server start --port 8080
```

Run all tests:

```bash
uv run pytest tests/ -v
```

Run the local live-backend integration tests:

```bash
make integration-test
```

Run a bounded local integration test model matrix:

```bash
RM_INTEGRATION_MAX_MODELS=8 make integration-test
```

The integration tests start a local Router-Maestro server, reuse the existing
local config/auth files, and send model-call requests to the real GitHub
Copilot backend. They require GitHub Copilot auth and are intentionally
local-only, not part of GitHub Actions:

```bash
uv run router-maestro auth login github-copilot
```

Run a single test file:

```bash
uv run pytest tests/test_auth.py -v
```

Run a single test:

```bash
uv run pytest tests/test_auth.py::TestAuthStorage::test_empty_storage -v
```

Lint:

```bash
uv run ruff check src/ tests/
```

Format:

```bash
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
```

Build and push the multi-arch Docker image:

```bash
make build-multiarch
```

Other useful Make targets include `make test`, `make lint`, `make format`,
`make run`, `make run-debug`, `make docker-up`, `make docker-down`,
`make dev-up`, and `make dev-down`.

## Repository Layout

```text
src/router_maestro/
|-- __init__.py              # Package root, exports __version__
|-- __main__.py              # Entry point for python -m router_maestro
|-- auth/                    # Credential storage and GitHub OAuth flow
|-- cli/                     # Typer CLI commands
|-- config/                  # Runtime config models and XDG paths
|-- providers/               # Provider implementations
|-- routing/                 # Provider/model selection and fallback
|-- server/                  # FastAPI app, routes, schemas, translation, SSE
`-- utils/                   # Shared helpers for tokens, models, caching, etc.

tests/                       # pytest suite
docs/                        # Design and behavior documentation
scripts/                     # Utility scripts
Makefile                     # Common development, Docker, and release commands
Dockerfile                   # Multi-stage Docker build
docker-compose.yml           # Production compose setup
docker-compose.dev.yml       # Development compose setup
```

## Architecture Notes

Provider implementations live under `src/router_maestro/providers/` and should
implement the `BaseProvider` contract from `providers/base.py`:

- `chat_completion(ChatRequest) -> ChatResponse`
- `chat_completion_stream(ChatRequest) -> AsyncIterator[ChatStreamChunk]`
- `list_models() -> list[ModelInfo]`
- `is_authenticated() -> bool`

Internal request/response objects are OpenAI-style dataclasses. Anthropic and
Gemini wire formats are translated at the server boundary before routing and
translated back when needed.

Primary API route modules:

- `server/routes/chat.py` handles OpenAI chat completions
- `server/routes/responses.py` handles OpenAI Responses API compatibility
- `server/routes/anthropic.py` handles Anthropic Messages API compatibility
- `server/routes/gemini.py` handles Gemini API compatibility
- `server/routes/models.py` handles model listing
- `server/routes/admin.py` handles management APIs

Translation and streaming helpers:

- `server/translation.py` for Anthropic/OpenAI conversion
- `server/translation_gemini.py` for Gemini conversion
- `server/streaming.py` for SSE streaming behavior
- `utils/responses_bridge.py` for Responses API bridging helpers

Routing behavior is centered in `routing/router.py`. Model matching and sorting
helpers live in `utils/model_match.py` and `utils/model_sort.py`.

## Provider and Routing Behavior

Models use the `provider/model-id` form, such as:

- `github-copilot/gpt-4o`
- `github-copilot/claude-sonnet-4`
- `openai/gpt-4-turbo`
- `anthropic/claude-3-5-sonnet`

The special model name `router-maestro` triggers automatic routing based on the
priority configuration.

GitHub Copilot provider quirks are important:

- Copilot can return separate choices for text and tool calls; merge behavior
  must preserve all tool calls.
- Some tool calls may appear as XML in text; `providers/tool_parsing.py`
  contains recovery logic.
- Copilot tokens expire; use the existing token refresh flow instead of adding
  duplicate token logic.
- Reasoning tiers and large-context variants have compatibility behavior that
  should remain transparent to clients.

## Runtime Configuration

Runtime state follows XDG paths:

- Config files under `~/.config/router-maestro/`
  - `providers.json`
  - `priorities.json`
  - `contexts.json`
- Data files under `~/.local/share/router-maestro/`
  - `auth.json`
  - `server.json`

Do not hardcode user-specific paths. Use the helpers in
`src/router_maestro/config/paths.py`.

The API server requires `ROUTER_MAESTRO_API_KEY` for authenticated access.
Most provider-backed requests also require appropriate provider credentials,
for example GitHub Copilot OAuth via:

```bash
uv run router-maestro auth login github-copilot
```

## Testing Guidance

- Add or update focused tests for behavior changes.
- Prefer testing translation, routing, provider normalization, and streaming at
  the boundary where behavior is observable.
- For route behavior, use the existing FastAPI test patterns in `tests/`.
- For provider behavior, mock upstream HTTP calls rather than calling real
  provider APIs.
- For local live-backend validation, use `make integration-test`; it runs the
  complete Copilot model matrix by default. To intentionally run a bounded
  subset, use `RM_INTEGRATION_MAX_MODELS=<N> make integration-test`.
- Run the narrowest relevant pytest target first, then broaden to the full
  suite when the change has wider risk.

## Style and Code Quality

- Target Python 3.11+.
- Keep line length at 100 characters.
- Follow Ruff rules from `pyproject.toml`: `E`, `F`, `I`, `N`, `W`, and `UP`.
- Use Pydantic models for request and response schemas.
- Use dataclasses already defined in provider base modules for internal
  provider-facing structures.
- Prefer explicit, typed functions for shared behavior.
- Keep comments sparse and useful; explain non-obvious protocol or compatibility
  behavior, not simple assignments.
- Avoid broad exception handling unless the caller needs fallback behavior and
  the error is logged or surfaced appropriately.

## Documentation

Update docs when behavior changes client-visible APIs, routing semantics,
configuration formats, Docker usage, release behavior, or provider compatibility.
Important docs include:

- `README.md`
- `CHANGELOG.md`
- `docs/api-translation.md`
- `docs/copilot-context-limits.md`
- `docs/deployment.md`
- `docs/token-calculation.md`
- `docs/tool-choice-behavior.md`

## Release and Version Updates

For a release, update all version-bearing files together:

1. `pyproject.toml`
2. `src/router_maestro/__init__.py`
3. `uv.lock` after running `uv lock`

Then create the tag:

```bash
git tag vX.Y.Z
```

The Makefile also has release-related targets, including `make dist`,
`make publish`, `make publish-test`, and `make release`.

## GitHub Operations

Use the `gh` CLI for GitHub operations when needed:

```bash
gh pr create --title "Title" --body "Description"
gh pr view <number>
gh pr merge <number>
gh issue create --title "Title" --body "Description"
gh issue list
```

Open pull requests against `master`.

# Router-Maestro

[![CI](https://github.com/MadSkittles/Router-Maestro/actions/workflows/ci.yml/badge.svg)](https://github.com/MadSkittles/Router-Maestro/actions/workflows/ci.yml)
[![Release](https://github.com/MadSkittles/Router-Maestro/actions/workflows/release.yml/badge.svg)](https://github.com/MadSkittles/Router-Maestro/actions/workflows/release.yml)

Multi-model routing router with OpenAI-compatible and Anthropic-compatible APIs. Route LLM requests across GitHub Copilot, OpenAI, Anthropic, and custom providers with intelligent fallback and priority-based selection.

## TL;DR

**Use GitHub Copilot's models (Claude, GPT-4o, o3-mini) with Claude Code or any OpenAI/Anthropic-compatible client.**

Router-Maestro acts as a proxy that gives you access to models from multiple providers through a unified API. Authenticate once with GitHub Copilot, and use its models anywhere that supports OpenAI or Anthropic APIs.

## Features

- **1M context support**: Activate Opus 4.6 with 1M context window via GitHub Copilot — just select `claude-opus-4-6[1m]` during `config claude-code` setup
- **Fuzzy model matching**: No need to type exact model IDs. Subagents, agent teams, and tools that hardcode model names (e.g. `opus-4-6`, `claude-sonnet-4.5`) are resolved automatically to the correct provider model
- **Multi-provider support**: GitHub Copilot (OAuth), OpenAI, Anthropic, and custom OpenAI-compatible endpoints
- **Intelligent routing**: Priority-based model selection with automatic fallback on failure
- **Dual API compatibility**: Both OpenAI (`/v1/...`) and Anthropic (`/v1/messages`) API formats
- **Gemini API compatibility**: Gemini REST API format (`/api/gemini/v1beta/...`) for Gemini CLI/SDK
- **Cross-provider translation**: Seamlessly route OpenAI requests to Anthropic providers and vice versa
- **Configuration hot-reload**: Auto-reload config files every 5 minutes without server restart
- **CLI management**: Full command-line interface for configuration and server control
- **Docker ready**: Production-ready Docker images with Traefik integration

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Model Identification](#model-identification)
  - [Auto-Routing](#auto-routing)
  - [Priority & Fallback](#priority--fallback)
  - [Cross-Provider Translation](#cross-provider-translation)
  - [Contexts](#contexts)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Deployment](#deployment)
  - [Architecture](#architecture)
  - [Option A: Simple Docker (No HTTPS)](#option-a-simple-docker-no-https)
  - [Option B: Production (Docker Compose + Traefik + HTTPS)](#option-b-production-docker-compose--traefik--https)
  - [Remote Management](#remote-management)
  - [Advanced Configuration](#advanced-configuration)
- [License](#license)
- [Changelog](#changelog)

## Quick Start

Get up and running in 4 steps:

<https://github.com/user-attachments/assets/8f60ec7a-4fbe-4342-9408-084073a4d48d>

### 1. Start the Server

#### Docker (recommended)

```bash
docker run -d -p 8080:8080 \
  -v ~/.local/share/router-maestro:/home/maestro/.local/share/router-maestro \
  -v ~/.config/router-maestro:/home/maestro/.config/router-maestro \
  likanwen/router-maestro:latest
```

#### Install locally

```bash
pip install router-maestro
router-maestro server start --port 8080
```

### 2. Set Context (for Docker or Remote)

When running via Docker in remote VPS, set up a context to communicate with the containerized server:

```bash
pip install router-maestro  # Install CLI locally
router-maestro context add docker --endpoint http://localhost:8080
router-maestro context set docker
```

> **What's a context?** A context is a named connection profile (endpoint + API key) that lets you manage local or remote Router-Maestro servers. See [Contexts](#contexts) for details.

### 3. Authenticate with GitHub Copilot

```bash
router-maestro auth login github-copilot

# Follow the prompts:
#   1. Visit https://github.com/login/device
#   2. Enter the displayed code
#   3. Authorize "GitHub Copilot Chat"
```

### 4. Configure Your CLI Tool

#### Claude Code

```bash
router-maestro config claude-code
# Follow the wizard to select models
```

#### OpenAI Codex (CLI, Extension, App)

```bash
router-maestro config codex
# Follow the wizard to select models
```

#### Gemini CLI

```bash
router-maestro config gemini
# Follow the wizard to select models
```

After configuration, set the API key environment variable:

```bash
# Get your API key
router-maestro server show-key

# Set the environment variable (add to your shell profile)
export ROUTER_MAESTRO_API_KEY="your-api-key-here"
```

**Done!** Now run `claude` or `codex` and your requests will route through Router-Maestro.

> **For production deployment**, see the [Deployment](#deployment) section.

## Core Concepts

### Model Identification

Models are identified using the format `{provider}/{model-id}`:

| Example                           | Description                         |
| --------------------------------- | ----------------------------------- |
| `github-copilot/gpt-4o` | GPT-4o via GitHub Copilot |
| `github-copilot/claude-sonnet-4` | Claude Sonnet 4 via GitHub Copilot |
| `openai/gpt-4-turbo` | GPT-4 Turbo via OpenAI |
| `anthropic/claude-3-5-sonnet` | Claude 3.5 Sonnet via Anthropic |

**Fuzzy matching**: You don't need to type exact model IDs. Router-Maestro will fuzzy-match common variations:

| You type              | Resolves to                      |
| --------------------- | -------------------------------- |
| `Opus 4.6`            | `claude-opus-4-6-20250617`       |
| `opus-4-6`            | `claude-opus-4-6-20250617`       |
| `claude-sonnet-4.5`   | `claude-sonnet-4-5-20250929`     |
| `anthropic/sonnet-4-5`| Sonnet 4.5 via Anthropic only    |

When multiple versions match, the newest (by date suffix) is selected automatically.

### Auto-Routing

Use the special model name `router-maestro` for automatic provider selection:

```json
{"model": "router-maestro", "messages": [...]}
```

The router will try models in priority order and fall back to the next on failure.

### Priority & Fallback

**Priority** determines which model is tried first when using auto-routing.

```bash
# Set priorities
router-maestro model priority github-copilot/claude-sonnet-4 --position 1
router-maestro model priority github-copilot/gpt-4o --position 2

# View priorities
router-maestro model priority list
```

**Fallback** triggers when a request fails with a retryable error (429, 5xx):

| Strategy     | Behavior                             |
| ------------ | ------------------------------------ |
| `priority` | Try next model in priorities list |
| `same-model` | Try same model on different provider |
| `none` | Fail immediately |

Configure in `~/.config/router-maestro/priorities.json`:

```json
{
  "priorities": ["github-copilot/claude-sonnet-4", "github-copilot/gpt-4o"],
  "fallback": {"strategy": "priority", "maxRetries": 2}
}
```

### Cross-Provider Translation

Router-Maestro automatically translates between OpenAI and Anthropic formats:

```bash
# Use Anthropic API with OpenAI provider
POST /v1/messages  {"model": "openai/gpt-4o", ...}

# Use OpenAI API with Anthropic provider
POST /v1/chat/completions  {"model": "anthropic/claude-3-5-sonnet", ...}
```

### Contexts

A **context** is a named connection profile that stores an endpoint URL and API key. Contexts let you manage multiple Router-Maestro deployments from a single CLI.

| Context  | Use Case                                   |
| -------- | ------------------------------------------ |
| `local` | Default context for `router-maestro server start` |
| `docker` | Connect to a local Docker container |
| `my-vps` | Connect to a remote VPS deployment |

```bash
# Add a context
router-maestro context add my-vps --endpoint https://api.example.com --api-key xxx

# Switch contexts
router-maestro context set my-vps

# All CLI commands now target the remote server
router-maestro model list
```

## CLI Reference

### Server

| Command                    | Description        |
| -------------------------- | ------------------ |
| `server start --port 8080` | Start the server   |
| `server stop` | Stop the server |
| `server info` | Show server status |

### Authentication

| Command                 | Description                    |
| ----------------------- | ------------------------------ |
| `auth login [provider]` | Authenticate with a provider   |
| `auth logout <provider>` | Remove authentication |
| `auth list` | List authenticated providers |

### Models

| Command                            | Description            |
| ---------------------------------- | ---------------------- |
| `model list`                       | List available models  |
| `model refresh` | Refresh models cache |
| `model priority list` | Show priorities |
| `model priority <model> --position <n>` | Set priority |
| `model fallback show` | Show fallback config |

### Contexts (Remote Management)

| Command                                              | Description          |
| ---------------------------------------------------- | -------------------- |
| `context show`                                       | Show current context |
| `context list` | List all contexts |
| `context set <name>` | Switch context |
| `context add <name> --endpoint <url> --api-key <key>` | Add remote context |
| `context test` | Test connection |

### Other

| Command              | Description                   |
| -------------------- | ----------------------------- |
| `config claude-code` | Generate Claude Code settings |
| `config codex`       | Generate Codex config (CLI/Extension/App) |
| `config gemini`      | Generate Gemini CLI .env      |

## API Reference

### OpenAI-Compatible

```bash
# Chat completions
POST /v1/chat/completions
{
  "model": "github-copilot/gpt-4o",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}

# List models
GET /v1/models
```

### Anthropic-Compatible

```bash
# Messages
POST /v1/messages
POST /api/anthropic/v1/messages
{
  "model": "github-copilot/claude-sonnet-4",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Hello"}]
}

# Count tokens
POST /v1/messages/count_tokens
```

### Admin

```bash
POST /api/admin/models/refresh   # Refresh model cache
```

### Gemini-Compatible

```bash
# Generate content (non-streaming)
POST /api/gemini/v1beta/models/{model}:generateContent
{
  "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
}

# Stream generate content (SSE)
POST /api/gemini/v1beta/models/{model}:streamGenerateContent?alt=sse
{
  "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
}

# Count tokens
POST /api/gemini/v1beta/models/{model}:countTokens
{
  "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
}
```

## Configuration

### File Locations

Following XDG Base Directory specification:

| Type       | Path                               | Contents                     |
| ---------- | ---------------------------------- | ---------------------------- |
| **Config** | `~/.config/router-maestro/` | |
| | `providers.json` | Custom provider definitions |
| | `priorities.json` | Model priorities and fallback |
| | `contexts.json` | Deployment contexts |
| **Data** | `~/.local/share/router-maestro/` | |
| | `auth.json` | OAuth tokens |
| | `server.json` | Server state |

### Custom Providers

Add OpenAI-compatible providers in `~/.config/router-maestro/providers.json`:

```json
{
  "providers": {
    "ollama": {
      "type": "openai-compatible",
      "baseURL": "http://localhost:11434/v1",
      "models": {
        "llama3": {"name": "Llama 3"},
        "mistral": {"name": "Mistral 7B"}
      }
    }
  }
}
```

Set API keys via environment variables (uppercase, hyphens → underscores):

```bash
export OLLAMA_API_KEY="sk-..."
```

### Hot-Reload

Configuration files are automatically reloaded every 5 minutes:

| File               | Auto-Reload      |
| ------------------ | ---------------- |
| `priorities.json` | ✓ (5 min) |
| `providers.json` | ✓ (5 min) |
| `auth.json` | Requires restart |

Force immediate reload:

```bash
router-maestro model refresh
```

## Deployment

### Architecture

```mermaid
graph TD
    Internet["🌐 Internet (HTTPS)"]
    subgraph VPS
        Traefik["Traefik (ports 80/443)\nAutomatic HTTPS · Let's Encrypt\nHTTP → HTTPS redirect"]
        RM["Router-Maestro (port 8080)\nOpenAI / Anthropic-compatible API\nMulti-provider routing"]
    end
    Providers["LLM Providers\nGitHub Copilot · OpenAI · Anthropic"]

    Internet -->|443| Traefik
    Traefik -->|8080| RM
    RM --> Providers
```

- **Traefik** — reverse proxy that handles TLS termination and auto-renews HTTPS certificates via Let's Encrypt. Only needed for public-facing deployments.
- **Router-Maestro** — the API server. Listens on port 8080, requires an API key for all requests, and routes them to configured LLM providers.

### Option A: Simple Docker (No HTTPS)

**Use when:** local testing, running behind an existing reverse proxy (Nginx, Caddy, etc.), or on an internal network.

**Prerequisites:** Docker installed.

**Step 1 — Generate an API key**

```bash
export ROUTER_MAESTRO_API_KEY=$(openssl rand -hex 32)
echo "Save this key: $ROUTER_MAESTRO_API_KEY"
```

**Step 2 — Start the container**

```bash
docker run -d --name router-maestro \
  -p 8080:8080 \
  -e ROUTER_MAESTRO_API_KEY="$ROUTER_MAESTRO_API_KEY" \
  -v ~/.local/share/router-maestro:/home/maestro/.local/share/router-maestro \
  -v ~/.config/router-maestro:/home/maestro/.config/router-maestro \
  likanwen/router-maestro:latest
```

**Step 3 — Authenticate with GitHub Copilot**

```bash
docker exec -it router-maestro router-maestro auth login github-copilot
# 1. Visit the URL shown
# 2. Enter the code
# 3. Authorize "GitHub Copilot Chat"
```

**Step 4 — Verify**

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}

curl http://localhost:8080/api/openai/v1/models \
  -H "Authorization: Bearer $ROUTER_MAESTRO_API_KEY"
# Expected: JSON list of available models
```

### Option B: Production (Docker Compose + Traefik + HTTPS)

**Use when:** deploying to a public-facing VPS with a domain name. Provides automatic HTTPS via Let's Encrypt with Cloudflare DNS challenge.

**Prerequisites:**
- A VPS with Docker and Docker Compose installed
- A domain name (e.g., `api.example.com`) with DNS pointing to your VPS
- A Cloudflare account managing your domain's DNS (for automatic HTTPS)

**Step 1 — Clone the repository**

```bash
git clone https://github.com/MadSkittles/Router-Maestro.git
cd Router-Maestro
```

**Step 2 — Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` with your values:

| Variable | Description | Example |
|----------|-------------|---------|
| `DOMAIN` | Your domain pointing to this VPS | `api.example.com` |
| `CF_DNS_API_TOKEN` | Cloudflare API token with `Zone:DNS:Edit` permission. [Generate here](https://dash.cloudflare.com/profile/api-tokens) | `abc123...` |
| `ACME_EMAIL` | Email for Let's Encrypt certificate expiry notifications | `you@example.com` |
| `ROUTER_MAESTRO_API_KEY` | API key clients use to authenticate. Generate with `openssl rand -hex 32` | `a1b2c3...` |
| `ROUTER_MAESTRO_LOG_LEVEL` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `TRAEFIK_DASHBOARD_AUTH` | (Optional) Basic auth for Traefik dashboard. Generate with `htpasswd -nB admin`, then escape `$` as `$$` | `admin:$$2y$$05$$...` |

**Step 3 — Start the services**

```bash
docker compose up -d
```

This starts both Traefik (reverse proxy) and Router-Maestro. Traefik will automatically obtain an HTTPS certificate for your domain.

**Step 4 — Authenticate with GitHub Copilot**

```bash
docker compose exec router-maestro router-maestro auth login github-copilot
# 1. Visit the URL shown
# 2. Enter the code
# 3. Authorize "GitHub Copilot Chat"
```

**Step 5 — Set up remote management (on your local machine)**

```bash
pip install router-maestro   # install CLI locally if not already installed

router-maestro context add my-vps \
  --endpoint https://api.example.com \
  --api-key YOUR_API_KEY

router-maestro context set my-vps
```

Now all CLI commands run against your VPS:

```bash
router-maestro model list          # list models on VPS
router-maestro auth list           # check auth status on VPS
router-maestro config claude-code  # configure Claude Code to use VPS
```

**Step 6 — Verify**

```bash
curl https://api.example.com/health
# Expected: {"status":"ok"}

curl https://api.example.com/api/openai/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
# Expected: JSON list of available models
```

### Remote Management

Contexts let you manage any Router-Maestro server (local or remote) from your local CLI:

```bash
# Add a remote server
router-maestro context add my-vps --endpoint https://api.example.com --api-key YOUR_KEY

# Switch between servers
router-maestro context set my-vps     # target remote VPS
router-maestro context set local      # target local server

# Test the connection
router-maestro context test

# All commands now target the active context
router-maestro model list
router-maestro auth login github-copilot
```

### Advanced Configuration

For additional deployment options, see [docs/deployment.md](docs/deployment.md):

- Alternative DNS providers (AWS Route53, DigitalOcean, GoDaddy, Namecheap, etc.)
- HTTP challenge setup (when DNS challenge is not available)
- Traefik dashboard configuration and security
- Complete environment variables reference

## License

MIT License - see [LICENSE](LICENSE) file.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Router-Maestro

Multi-model routing router with OpenAI-compatible and Anthropic-compatible APIs. Route LLM requests across GitHub Copilot, OpenAI, Anthropic, and custom providers with intelligent fallback and priority-based selection.

## TL;DR

**Use GitHub Copilot's models (Claude, GPT-4o, o3-mini) with Claude Code or any OpenAI/Anthropic-compatible client.**

Router-Maestro acts as a proxy that gives you access to models from multiple providers through a unified API. Authenticate once with GitHub Copilot, and use its models anywhere that supports OpenAI or Anthropic APIs.

## Features

- **Multi-provider support**: GitHub Copilot (OAuth), OpenAI, Anthropic, and custom OpenAI-compatible endpoints
- **Intelligent routing**: Priority-based model selection with automatic fallback on failure
- **Dual API compatibility**: Both OpenAI (`/v1/...`) and Anthropic (`/v1/messages`) API formats
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
- [License](#license)

## Quick Start

Get up and running in 4 steps:

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

### 4. Configure Claude Code

```bash
router-maestro config claude-code
# Follow the wizard to select models
```

**Done!** Now run `claude` and your requests will route through Router-Maestro.

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

### Docker Deployment

Deploy to a VPS with Docker Compose:

```bash
# On your VPS
git clone https://github.com/likanwen/router-maestro.git
cd router-maestro
cp .env.example .env  # Edit with your domain
docker compose up -d
```

Configure `.env`:

```bash
DOMAIN=api.example.com
CF_DNS_API_TOKEN=your_cloudflare_token  # For HTTPS
ACME_EMAIL=your@email.com
ROUTER_MAESTRO_API_KEY=$(openssl rand -hex 32)
```

Authenticate inside the container:

```bash
docker compose exec router-maestro /bin/sh
router-maestro auth login github-copilot
# Follow OAuth flow, then exit
```

### Remote Management

Manage your VPS deployment from your local machine using contexts:

```bash
# Add remote context
router-maestro context add my-vps \
  --endpoint https://api.example.com \
  --api-key your_api_key

# Switch to remote
router-maestro context set my-vps

# Now all commands target the VPS
router-maestro model list
```

### HTTPS with Traefik

The Docker Compose setup includes Traefik for automatic HTTPS via Let's Encrypt with DNS challenge.

For detailed configuration options including:

- Other DNS providers (Route53, DigitalOcean, etc.)
- HTTP challenge setup
- Traefik dashboard configuration

See [docs/deployment.md](docs/deployment.md).

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

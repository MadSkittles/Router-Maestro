# Router-Maestro

Multi-model routing router with OpenAI-compatible and Anthropic-compatible APIs. Route LLM requests across GitHub Copilot, OpenAI, Anthropic, and custom providers with intelligent fallback and priority-based selection.

## Features

- **Multi-provider support**: GitHub Copilot (OAuth), OpenAI, Anthropic, and custom OpenAI-compatible endpoints
- **Intelligent routing**: Priority-based model selection with automatic fallback on failure
- **Dual API compatibility**: Both OpenAI (`/v1/...`) and Anthropic (`/v1/messages`) API formats
- **Cross-provider translation**: Seamlessly route OpenAI requests to Anthropic providers and vice versa
- **Usage tracking**: Token usage statistics with heatmap visualization
- **CLI management**: Full command-line interface for configuration and server control
- **Hot-reload cache**: 5-minute TTL model cache with admin refresh endpoint
- **Docker ready**: Production-ready Docker images with Traefik integration

## Installation

### pip/uv

```bash
pip install router-maestro
# or
uv pip install router-maestro
```

### uvx (run without installing)

```bash
uvx --from router-maestro router-maestro --help
```

### Development

```bash
git clone https://github.com/likanwen/router-maestro.git
cd router-maestro
uv pip install -e ".[dev]"
```

## Quick Start

### 1. Start the server

```bash
# Start API server on port 8080
router-maestro server start --port 8080
```

The server will:
- Auto-generate an API key (displayed on startup)
- Pre-warm model cache if providers are already authenticated
- Listen on `http://localhost:8080`

### 2. Authenticate with a provider

```bash
# Interactive provider selection
router-maestro auth login

# Or specify provider directly
router-maestro auth login github-copilot
router-maestro auth login openai
router-maestro auth login anthropic
```

### 3. List available models

```bash
router-maestro model list
```

### 4. Make API requests

```bash
# Using the OpenAI-compatible endpoint
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "github-copilot/gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## CLI Commands

### Server Management

```bash
# Start server
router-maestro server start --port 8080

# Stop server
router-maestro server stop

# Get server info
router-maestro server info
```

### Authentication

```bash
# Login (interactive)
router-maestro auth login

# Login to specific provider
router-maestro auth login github-copilot
router-maestro auth login openai
router-maestro auth login anthropic

# Logout
router-maestro auth logout github-copilot

# List authenticated providers
router-maestro auth list
```

### Model Management

```bash
# List all available models
router-maestro model list

# Set model priority (higher priority = tried first)
router-maestro model priority github-copilot/gpt-4o --position 1
router-maestro model priority openai/gpt-4-turbo --position 2

# Remove from priority
router-maestro model priority remove github-copilot/gpt-4o

# Show current priorities
router-maestro model priority list
```

### Context Management

```bash
# Set deployment API key
router-maestro context set-key YOUR_API_KEY

# Get current API key
router-maestro context get-key

# Show current context
router-maestro context show
```

### Statistics

```bash
# Show token usage for last 7 days
router-maestro stats --days 7

# Show heatmap visualization
router-maestro stats --days 30 --heatmap
```

### Configuration

```bash
# Generate Claude Code settings
router-maestro config claude-code
```

## API Endpoints

### OpenAI-Compatible API

#### Chat Completions

```bash
POST /v1/chat/completions
```

Request format (standard OpenAI):

```json
{
  "model": "github-copilot/gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

Streaming:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "github-copilot/gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

#### List Models

```bash
GET /v1/models
```

### Anthropic-Compatible API

#### Messages

```bash
POST /v1/messages
POST /api/anthropic/v1/messages
```

Request format (Anthropic):

```json
{
  "model": "github-copilot/claude-3-5-sonnet",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
```

Streaming with Anthropic format:

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "github-copilot/claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Count Tokens

```bash
POST /v1/messages/count_tokens
POST /api/anthropic/v1/messages/count_tokens
```

### Admin API

```bash
# Refresh model cache
POST /api/admin/models/refresh
```

## Model Identification

Models are identified using the format: `{provider}/{model-id}`

Examples:
- `github-copilot/gpt-4o` - GPT-4o via GitHub Copilot
- `openai/gpt-4-turbo` - GPT-4 Turbo via OpenAI
- `anthropic/claude-3-5-sonnet` - Claude 3.5 Sonnet via Anthropic
- `my-llm/llama-3` - Custom model via OpenAI-compatible provider

### Auto-Routing

Use the special model name `router-maestro` for automatic provider selection:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router-maestro",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

The router will:
1. Try providers in priority order
2. Use the first authenticated provider with available models
3. Fall back to next provider on failure (if configured)

## Configuration

### Provider Configuration

Edit `~/.config/router-maestro/providers.json` to add custom providers:

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
    },
    "vllm": {
      "type": "openai-compatible",
      "baseURL": "http://localhost:8000/v1ories": {}
    }
  }
}
```

Set API keys via environment variables (uppercase, underscore for hyphens):

```bash
export OLLAMA_API_KEY="sk-..."  # For ollama provider
export VLLM_API_KEY="sk-..."    # For vllm provider
```

### Priority Configuration

Edit `~/.config/router-maestro/priorities.json`:

```json
{
  "priorities": [
    "github-copilot/gpt-4o",
    "openai/gpt-4-turbo",
    "anthropic/claude-3-5-sonnet"
  ],
  "fallback": {
    "strategy": "priority",
    "maxRetries": 2
  }
}
```

Fallback strategies:
- `priority` - Try next model in priorities list
- `same-model` - Try same model on different providers
- `none` - No fallback, fail immediately

### Data Locations

Following XDG Base Directory specification:

| Type | Location |
|-------|-----------|
| Config | `~/.config/router-maestro/` |
| - providers.json | Custom provider definitions |
| - priorities.json | Model priorities and fallback config |
| - contexts.json | Deployment context and API keys |
| Data | `~/.local/share/router-maestro/` |
| - auth.json | Stored OAuth tokens and API keys |
| - server.json | Server state |
| - stats.db | Token usage statistics |

## Docker Deployment

### Using docker-compose

```bash
# Copy .env.example to .env and configure
cp .env.example .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f router-maestro
```

The docker-compose includes:
- **Traefik**: Reverse proxy with automatic HTTPS via Let's Encrypt (Cloudflare DNS)
- **Router-Maestro**: Application server with persistent volumes

### Standalone container

```bash
docker run -d --name router-maestro -p 8080:8080 \
  -e ROUTER_MAESTRO_API_KEY=your_secure_key \
  -e ROUTER_MAESTRO_LOG_LEVEL=INFO \
  -v ~/.local/share/router-maestro:/home/maestro/.local/share/router-maestro \
  -v ~/.config/router-maestro:/home/maestro/.config/router-maestro \
  likanwen/router-maestro:latest
```

## Cross-Provider Request Translation

Router-Maestro automatically translates requests between OpenAI and Anthropic formats:

- **OpenAI request → Anthropic provider**: Messages, roles, and parameters are translated
- **Anthropic request → OpenAI provider**: Blocks, roles, and system prompts are translated

This means you can use either API format with any provider:

```bash
# Use Anthropic API format with OpenAI provider
POST /v1/messages
{
  "model": "openai/gpt-4o",  # OpenAI provider
  "messages": [{"role": "user", "content": "Hello"}]
}

# Use OpenAI API format with Anthropic provider
POST /v1/chat/completions
{
  "model": "anthropic/claude-3-5-sonnet",  # Anthropic provider
  "messages": [{"role": "user", "content": "Hello"}]
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

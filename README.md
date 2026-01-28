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

### Quick Deploy to VPS

This section covers deploying Router-Maestro to a VPS with HTTPS support via Traefik.

#### 1. Prerequisites

- A VPS with Docker and Docker Compose installed
- A domain pointing to your VPS (e.g., `api.example.com`)
- A Cloudflare account with your domain configured (for DNS challenge)

#### 2. Clone and Configure

```bash
# On your VPS
git clone https://github.com/likanwen/router-maestro.git
cd router-maestro

# Copy environment template
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Your domain (must point to this VPS)
DOMAIN=api.example.com

# Cloudflare API token (Zone:DNS:Edit permission)
# Generate at: https://dash.cloudflare.com/profile/api-tokens
CF_DNS_API_TOKEN=your_cloudflare_api_token

# Email for Let's Encrypt notifications
ACME_EMAIL=your-email@example.com

# API key for Router-Maestro (generate a secure random string)
ROUTER_MAESTRO_API_KEY=$(openssl rand -hex 32)

# Traefik dashboard auth (optional, generate with: htpasswd -nB admin)
# Note: Escape $ as $$ in .env file
TRAEFIK_DASHBOARD_AUTH=admin:$$2y$$05$$your_bcrypt_hash_here
```

#### 3. Deploy

```bash
# Start services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f router-maestro
```

Your Router-Maestro API will be available at `https://api.example.com`.

### HTTPS with Traefik and Let's Encrypt

The included `docker-compose.yml` uses [Traefik](https://traefik.io/) as a reverse proxy with automatic HTTPS certificate management via [Let's Encrypt](https://letsencrypt.org/).

#### How It Works

1. **Traefik** listens on ports 80 and 443
2. **Let's Encrypt** issues free SSL certificates automatically
3. **DNS Challenge** verifies domain ownership without opening additional ports
4. **Auto-renewal** happens before certificates expire

#### Default: Cloudflare DNS Challenge

The default configuration uses Cloudflare for DNS challenge. This is the recommended approach because:
- Works even if port 80 is blocked
- Supports wildcard certificates
- No downtime during certificate renewal

Required Cloudflare API token permissions:
- `Zone:DNS:Edit` - to create TXT records for verification

#### Using Other DNS Providers

Traefik supports 100+ DNS providers. To switch from Cloudflare:

1. **Update `docker-compose.yml`** - change the DNS challenge provider:

```yaml
# In traefik service command section, replace:
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=cloudflare"
# With your provider, e.g.:
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=route53"      # AWS
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=digitalocean" # DigitalOcean
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=godaddy"      # GoDaddy
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=namecheap"    # Namecheap
```

2. **Update environment variables** - each provider requires different credentials:

```yaml
# In traefik service environment section, replace CF_DNS_API_TOKEN with:
# AWS Route53
- AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
- AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# DigitalOcean
- DO_AUTH_TOKEN=${DO_AUTH_TOKEN}

# GoDaddy
- GODADDY_API_KEY=${GODADDY_API_KEY}
- GODADDY_API_SECRET=${GODADDY_API_SECRET}
```

3. **Update `.env`** with your provider's credentials

See the [Traefik DNS Challenge documentation](https://doc.traefik.io/traefik/https/acme/#providers) for the full list of supported providers and their required environment variables.

#### Using HTTP Challenge (Alternative)

If you don't want to use DNS challenge, you can use HTTP challenge instead. This requires port 80 to be accessible:

```yaml
# Replace dnschallenge lines with:
- "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
- "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
```

#### 4. Authenticate with GitHub Copilot (on VPS)

You need to authenticate with providers on the VPS. The easiest way is to run CLI commands inside the container:

```bash
# Enter the container
docker compose exec router-maestro /bin/sh

# Login to GitHub Copilot (follow the OAuth flow)
router-maestro auth login github-copilot

# Verify authentication
router-maestro auth list

# Exit container
exit
```

### Managing Remote Deployments from Local Machine

Once your VPS is deployed, you can manage it from your local machine using **contexts**.

#### 1. Install Router-Maestro Locally

```bash
pip install router-maestro
# or
uvx --from router-maestro router-maestro --help
```

#### 2. Add Your VPS as a Context

```bash
# Add the remote deployment
router-maestro context add my-vps \
  --endpoint https://api.example.com \
  --api-key your_router_maestro_api_key

# Switch to the remote context
router-maestro context set my-vps

# Verify connection
router-maestro context test
```

#### 3. Manage the Remote Server

Now all CLI commands will target your VPS:

```bash
# List models available on the remote server
router-maestro model list

# Check authenticated providers
router-maestro auth list

# View statistics
router-maestro stats --days 7

# Set model priorities
router-maestro model priority github-copilot/claude-sonnet-4 --position 1
```

#### 4. Switch Between Contexts

```bash
# List all contexts
router-maestro context list

# Switch back to local
router-maestro context set local

# Switch to VPS
router-maestro context set my-vps

# Show current context
router-maestro context current
```

### GitHub Copilot OAuth Authentication

Router-Maestro uses GitHub's OAuth Device Flow for Copilot authentication:

#### On Local Server

```bash
# Start the server first
router-maestro server start --port 8080

# In another terminal, initiate OAuth
router-maestro auth login github-copilot
```

You'll see:
```
Please visit the following URL and enter the code:
  URL: https://github.com/login/device
  Code: ABCD-1234

Waiting for authorization...
```

1. Open the URL in your browser
2. Enter the code
3. Authorize "GitHub Copilot Chat"
4. The CLI will confirm: "Successfully authenticated!"

#### On Remote VPS

```bash
# Enter the container
docker compose exec router-maestro /bin/sh

# Run OAuth flow
router-maestro auth login github-copilot
# Follow the same steps as above

# Exit container
exit
```

The OAuth token is stored in `/home/maestro/.local/share/router-maestro/auth.json` inside the container (persisted via Docker volume).

### Generating Claude Code settings.json

Router-Maestro can generate a `settings.json` file for Claude Code CLI to use your deployment.

#### Interactive Generation

```bash
# Make sure you're connected to the right context
router-maestro context current

# Generate settings.json
router-maestro config claude-code
```

The wizard will:
1. Ask for configuration level (user `~/.claude/` or project `./.claude/`)
2. Show available models from your server
3. Let you select main and fast models
4. Generate the settings file

#### Example Generated settings.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.example.com/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "your_router_maestro_api_key",
    "ANTHROPIC_MODEL": "github-copilot/claude-sonnet-4",
    "ANTHROPIC_SMALL_FAST_MODEL": "github-copilot/gpt-4o-mini",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

#### Manual Configuration

If you prefer to create the file manually:

```bash
mkdir -p ~/.claude

cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.example.com/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "your_router_maestro_api_key",
    "ANTHROPIC_MODEL": "github-copilot/claude-sonnet-4",
    "ANTHROPIC_SMALL_FAST_MODEL": "github-copilot/gpt-4o-mini",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
EOF
```

Now Claude Code will route requests through your Router-Maestro deployment.

### Using docker-compose (Reference)

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

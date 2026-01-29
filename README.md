# Router-Maestro

Multi-model routing router with OpenAI-compatible and Anthropic-compatible APIs. Route LLM requests across GitHub Copilot, OpenAI, Anthropic, and custom providers with intelligent fallback and priority-based selection.

## Features

- **Multi-provider support**: GitHub Copilot (OAuth), OpenAI, Anthropic, and custom OpenAI-compatible endpoints
- **Intelligent routing**: Priority-based model selection with automatic fallback on failure
- **Dual API compatibility**: Both OpenAI (`/v1/...`) and Anthropic (`/v1/messages`) API formats
- **Cross-provider translation**: Seamlessly route OpenAI requests to Anthropic providers and vice versa
- **Configuration hot-reload**: Auto-reload config files every 5 minutes without server restart
- **Usage tracking**: Token usage statistics with heatmap visualization
- **CLI management**: Full command-line interface for configuration and server control
- **Docker ready**: Production-ready Docker images with Traefik integration

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Model Identification & Routing](#model-identification--routing)
- [Priority Configuration](#priority-configuration)
- [Configuration Hot-Reload](#configuration-hot-reload)
- [API Endpoints](#api-endpoints)
- [Provider Configuration](#provider-configuration)
- [Docker Deployment](#docker-deployment)
- [Claude Code Integration](#claude-code-integration)
- [License](#license)

## Installation

### pip/uv

```bash
pip install router-maestro
# or
uv pip install router-maestro
# or
uv tool install router-maestro
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
router-maestro server start --port 8080   # Start server
router-maestro server stop                 # Stop server
router-maestro server info                 # Get server info
```

### Authentication

```bash
router-maestro auth login                  # Interactive login
router-maestro auth login github-copilot   # Login to specific provider
router-maestro auth logout github-copilot  # Logout from provider
router-maestro auth list                   # List authenticated providers
```

### Model Management

```bash
router-maestro model list                  # List all available models
router-maestro model refresh               # Refresh models cache immediately
router-maestro model priority list         # Show current priorities
router-maestro model priority <model> --position <n>  # Set priority
router-maestro model priority remove <model>          # Remove from priority
router-maestro model fallback show         # Show fallback configuration
router-maestro model fallback set --strategy <s> --max-retries <n>  # Set fallback
```

### Context Management

```bash
router-maestro context show                # Show current context
router-maestro context list                # List all contexts
router-maestro context set <name>          # Switch context
router-maestro context add <name> --endpoint <url> --api-key <key>  # Add context
router-maestro context set-key <key>       # Set API key for current context
router-maestro context get-key             # Get current API key
router-maestro context test                # Test connection to current context
```

### Statistics

```bash
router-maestro stats --days 7              # Show token usage for last 7 days
router-maestro stats --days 30 --heatmap   # Show heatmap visualization
```

### Configuration

```bash
router-maestro config claude-code          # Generate Claude Code settings.json
```

## Model Identification & Routing

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
1. Try models in priority order (from `priorities.json`)
2. If priorities list is empty, use the first available model from any authenticated provider
3. Fall back to next model on failure (if configured)

**Note**: The special model name `router-maestro` does not appear in `model list` or `/v1/models` API responses. It is only used in requests to trigger auto-routing.

### Cross-Provider Translation

Router-Maestro automatically translates requests between OpenAI and Anthropic formats:

```bash
# Use Anthropic API format with OpenAI provider
POST /v1/messages
{"model": "openai/gpt-4o", "messages": [...]}

# Use OpenAI API format with Anthropic provider
POST /v1/chat/completions
{"model": "anthropic/claude-3-5-sonnet", "messages": [...]}
```

## Priority Configuration

Priority determines which model is tried first when using auto-routing (`model: "router-maestro"`).

### Setting Priorities

```bash
# Set model as highest priority (position 1)
router-maestro model priority github-copilot/claude-sonnet-4 --position 1

# Set model as second priority
router-maestro model priority openai/gpt-4o --position 2

# View current priorities
router-maestro model priority list
```

### How Priority Insertion Works

When you set a model's priority, existing models shift to make room (they don't get replaced):

**Example**: Starting with priorities `[A, B, C]` (positions 1, 2, 3)

```bash
# Insert D at position 2
router-maestro model priority D --position 2
# Result: [A, D, B, C] (positions 1, 2, 3, 4)
# - A stays at position 1
# - D is inserted at position 2
# - B shifts from position 2 to 3
# - C shifts from position 3 to 4
```

### Priority Configuration File

Edit `~/.config/router-maestro/priorities.json` directly:

```json
{
  "priorities": [
    "github-copilot/claude-sonnet-4",
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

### Fallback Behavior

Fallback only triggers when:
1. The error is **retryable** (see Retryable Errors below)
2. The fallback strategy is not `none`
3. There are valid fallback candidates

**Important**: Fallback behavior depends on how you specify the model:

| Model Specification | Fallback Behavior |
|---------------------|-------------------|
| `router-maestro` (auto-routing) | Tries models in priority order; on failure, continues down the list |
| `provider/model` in priorities list | On failure, tries models **after** it in the priorities list |
| `provider/model` NOT in priorities list | **No fallback** - fails immediately even if retryable |

**When priorities list is empty**:
- Auto-routing (`router-maestro`) will use the first available model from any authenticated provider
- Since there's no priorities list to follow, **no fallback occurs** on failure (even with `priority` strategy)
- To enable fallback, add models to your priorities list

### Fallback Strategies

Configure in `priorities.json`:

```json
{
  "fallback": {
    "strategy": "priority",
    "maxRetries": 2
  }
}
```

#### `priority` Strategy

On failure, try the next model in the priorities list (in order).

**Example**: priorities = `[A, B, C, D]`
- Request with `model: "router-maestro"` → tries A, if fails tries B, then C...
- Request with `model: "B"` → tries B, if fails tries C, then D (skips A)
- Request with `model: "X"` (not in list) → tries X, if fails → **error** (no fallback)
- **priorities = `[]` (empty)** → auto-routing picks first available model, if fails → **error** (no fallback)

#### `same-model` Strategy

On failure, try the **same model** on a different provider.

**Example**: You have `gpt-4o` available on both `github-copilot` and `openai`
- Request with `model: "github-copilot/gpt-4o"` fails
- Router tries `openai/gpt-4o` as fallback

This strategy is useful when:
- Multiple providers offer the same model
- You want provider redundancy without changing the model

**Note**: If no other provider has the same model, no fallback occurs.

#### `none` Strategy

Never fallback. Any failure returns an error immediately.

### Retryable Errors

Fallback only occurs for these HTTP status codes:
- `429` - Rate limited
- `500`, `502`, `503`, `504` - Server errors
- `529` - Anthropic overloaded

Non-retryable errors (e.g., `400 Bad Request`, `401 Unauthorized`, `404 Not Found`) always fail immediately without fallback.

## Configuration Hot-Reload

Router-Maestro automatically reloads configuration files without requiring a server restart.

### Auto-Reload Behavior

| File | Auto-Reload | TTL |
|------|-------------|-----|
| `priorities.json` | Yes | 5 minutes |
| `providers.json` | Yes | 5 minutes |
| `auth.json` | No | Requires restart |

### When Changes Take Effect

1. **Within 5 minutes**: Configuration changes are automatically detected and applied
2. **Immediately**: Use CLI commands to force immediate reload:
   ```bash
   # Refresh models cache (reloads providers.json and priorities.json)
   router-maestro model refresh
   ```

### Manual Refresh via API

```bash
POST /api/admin/models/refresh
```

This endpoint:
- Reloads `providers.json` and `priorities.json`
- Clears the models cache
- Re-fetches models from all providers

## API Endpoints

### OpenAI-Compatible API

#### Chat Completions

```bash
POST /v1/chat/completions
```

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

```json
{
  "model": "github-copilot/claude-3-5-sonnet",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

#### Count Tokens

```bash
POST /v1/messages/count_tokens
POST /api/anthropic/v1/messages/count_tokens
```

### Admin API

```bash
POST /api/admin/models/refresh   # Refresh model cache
```

## Provider Configuration

### Custom Providers

Edit `~/.config/router-maestro/providers.json` to add custom OpenAI-compatible providers:

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
      "baseURL": "http://localhost:8000/v1",
      "models": {}
    }
  }
}
```

### API Keys for Custom Providers

Set API keys via environment variables (uppercase, replace hyphens with underscores):

```bash
export OLLAMA_API_KEY="sk-..."  # For ollama provider
export VLLM_API_KEY="sk-..."    # For vllm provider
```

### Data Locations

Following XDG Base Directory specification:

| Type | Location |
|------|----------|
| **Config** | `~/.config/router-maestro/` |
| - providers.json | Custom provider definitions |
| - priorities.json | Model priorities and fallback config |
| - contexts.json | Deployment contexts and API keys |
| **Data** | `~/.local/share/router-maestro/` |
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

## Claude Code Integration

Router-Maestro can generate a `settings.json` file for Claude Code CLI to use your deployment.

### Interactive Generation

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

### Example Generated settings.json

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

### Manual Configuration

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

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

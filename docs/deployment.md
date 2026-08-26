# Advanced Deployment Guide

This guide covers advanced deployment options for Router-Maestro, including HTTPS configuration with various DNS providers.

## Table of Contents

- [HTTPS with Traefik](#https-with-traefik)
- [DNS Challenge Providers](#dns-challenge-providers)
- [HTTP Challenge (Alternative)](#http-challenge-alternative)
- [Traefik Dashboard](#traefik-dashboard)
- [Environment Variables Reference](#environment-variables-reference)
- [Reasoning Capsule Keys](#reasoning-capsule-keys)

## HTTPS with Traefik

The included `docker-compose.yml` uses [Traefik](https://traefik.io/) as a reverse proxy with automatic HTTPS certificate management via [Let's Encrypt](https://letsencrypt.org/).

### How It Works

1. **Traefik** listens on ports 80 and 443
2. **Let's Encrypt** issues free SSL certificates automatically
3. **DNS Challenge** verifies domain ownership without opening additional ports
4. **Auto-renewal** happens before certificates expire

### Default: Cloudflare DNS Challenge

The default configuration uses Cloudflare for DNS challenge. This is the recommended approach because:

- Works even if port 80 is blocked
- Supports wildcard certificates
- No downtime during certificate renewal

Required Cloudflare API token permissions:

- `Zone:DNS:Edit` - to create TXT records for verification

Generate a token at: https://dash.cloudflare.com/profile/api-tokens

## DNS Challenge Providers

Traefik supports 100+ DNS providers. Below are common configurations.

### AWS Route53

Update `docker-compose.yml`:

```yaml
# In traefik service command section, replace cloudflare with:
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=route53"

# In traefik service environment section:
environment:
  - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
  - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
  - AWS_REGION=${AWS_REGION}
```

Update `.env`:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

### DigitalOcean

Update `docker-compose.yml`:

```yaml
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=digitalocean"

environment:
  - DO_AUTH_TOKEN=${DO_AUTH_TOKEN}
```

Update `.env`:

```bash
DO_AUTH_TOKEN=your_digitalocean_token
```

### GoDaddy

Update `docker-compose.yml`:

```yaml
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=godaddy"

environment:
  - GODADDY_API_KEY=${GODADDY_API_KEY}
  - GODADDY_API_SECRET=${GODADDY_API_SECRET}
```

Update `.env`:

```bash
GODADDY_API_KEY=your_api_key
GODADDY_API_SECRET=your_api_secret
```

### Namecheap

Update `docker-compose.yml`:

```yaml
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=namecheap"

environment:
  - NAMECHEAP_API_USER=${NAMECHEAP_API_USER}
  - NAMECHEAP_API_KEY=${NAMECHEAP_API_KEY}
```

Update `.env`:

```bash
NAMECHEAP_API_USER=your_username
NAMECHEAP_API_KEY=your_api_key
```

### Other Providers

See the [Traefik DNS Challenge documentation](https://doc.traefik.io/traefik/https/acme/#providers) for the full list of 100+ supported providers and their required environment variables.

## HTTP Challenge (Alternative)

If you don't want to use DNS challenge, you can use HTTP challenge instead. This requires port 80 to be accessible from the internet.

Update `docker-compose.yml`:

```yaml
# Replace these lines:
- "--certificatesresolvers.letsencrypt.acme.dnschallenge=true"
- "--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=cloudflare"

# With:
- "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
- "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
```

Remove the DNS provider environment variables from the traefik service.

**Limitations of HTTP Challenge:**

- Port 80 must be accessible from the internet
- Does not support wildcard certificates
- Brief downtime during initial certificate issuance

## Traefik Dashboard

The Docker Compose setup includes an optional Traefik dashboard for monitoring.

### Enable the Dashboard

The dashboard is configured in `docker-compose.yml`. To access it:

1. Generate a password hash:

```bash
htpasswd -nB admin
# Enter password when prompted
# Output: admin:$2y$05$...
```

2. Add to `.env` (escape `$` as `$$`):

```bash
TRAEFIK_DASHBOARD_AUTH=admin:$$2y$$05$$your_hash_here
```

3. Access at `https://traefik.your-domain.com` (configure the domain in docker-compose.yml)

### Dashboard Security

- Always use HTTPS for the dashboard
- Use strong passwords
- Consider IP whitelisting for additional security

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `DOMAIN` | Your domain (e.g., `api.example.com`) | Yes |
| `ACME_EMAIL` | Email for Let's Encrypt notifications | Yes |
| `ROUTER_MAESTRO_API_KEY` | Optional fixed Router-Maestro server API key. Leave blank, and do not set it in the shell running Docker Compose, to auto-generate on first start. | No |
| `ROUTER_MAESTRO_REASONING_CAPSULE_KEY` | Current unpadded base64url-encoded 32-byte AES key used to seal reasoning continuation capsules. Required for multi-instance deployments. | Conditional |
| `ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS` | Optional comma-separated decrypt-only old capsule keys used during rotation. Valid only when the current key is set. | No |
| `CF_DNS_API_TOKEN` | Cloudflare API token (if using Cloudflare) | Conditional |
| `TRAEFIK_DASHBOARD_AUTH` | Basic auth for Traefik dashboard | No |

### Router-Maestro API Key

Router-Maestro protects API routes with one server API key. If `ROUTER_MAESTRO_API_KEY` is blank or unset in both `.env` and the shell running Docker Compose, the server generates a `sk-rm-...` key on first start and stores it in the mounted Router-Maestro config. Read it from inside the container:

```bash
docker compose exec router-maestro router-maestro server show-key
```

Use that same key in remote client contexts and raw API calls. If deployment automation needs a stable pre-provisioned value, set `ROUTER_MAESTRO_API_KEY` in `.env` before starting the service.

This is currently also the administrator credential: `/api/admin/*` and
inference routes use the same `ROUTER_MAESTRO_API_KEY`, and remote CLI
management reads that key from the active context. There is no separate
administrator-key environment variable in the current release. Protect this
key as both inference and configuration authority, expose admin routes only on
trusted networks or behind an appropriate access-control layer, and rotate the
single key consistently across all clients when required.

### Reasoning Capsule Keys

Cross-protocol reasoning replay seals provider-owned opaque state in an
authenticated `rmr1` capsule. A capsule is bound to its provider, upstream
model, transport binding, and reasoning item ID. Tampering, an unknown key,
unsupported version, or provenance mismatch fails closed before upstream I/O;
Router-Maestro does not log keys, capsule plaintext, or decrypted provider
state.

For a single instance with no capsule environment variables, Router-Maestro
atomically generates and reuses:

```text
~/.local/share/router-maestro/reasoning-capsule-keys.json
```

The file must be owner-only (`0600` on POSIX). An invalid environment value or
a corrupt, unreadable, or overly permissive key file causes startup to fail.
The server never silently replaces an unusable key, because doing so would
invalidate active conversations.

The included production and development Compose files forward both capsule
variables from the host only when they are set. They intentionally use bare
environment-list entries rather than an empty-value default: injecting an
empty current key is invalid and would prevent startup instead of selecting the
single-instance XDG key file.

Multi-instance deployments must provide the same
`ROUTER_MAESTRO_REASONING_CAPSULE_KEY` to every instance through the deployment
secret manager. The value is an unpadded URL-safe base64 encoding of exactly 32
random bytes. Do not rely on separate container-local XDG files behind a load
balancer: a later reasoning turn may reach another instance.

Rotate without breaking existing capsules in this order:

1. Deploy the new current key to every instance and include the former current
   key in the comma-separated
   `ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS` list.
2. Confirm every instance has the identical current and previous-key set before
   issuing new traffic.
3. After the maximum conversation/replay lifetime, remove expired previous
   keys. Previous keys are decrypt-only; new capsules always use the current
   key.

### Complete .env Example

```bash
# Domain configuration
DOMAIN=api.example.com
ACME_EMAIL=admin@example.com

# Router-Maestro
ROUTER_MAESTRO_API_KEY=
# Required and shared when more than one Router-Maestro instance serves traffic
# ROUTER_MAESTRO_REASONING_CAPSULE_KEY=<unpadded-base64url-32-byte-key>
# ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS=<old-key-1>,<old-key-2>

# Cloudflare (default DNS provider)
CF_DNS_API_TOKEN=your_cloudflare_api_token

# Traefik dashboard (optional)
# Note: $ must be escaped as $$ in .env files
TRAEFIK_DASHBOARD_AUTH=admin:$$2y$$05$$your_bcrypt_hash
```

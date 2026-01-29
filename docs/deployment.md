# Advanced Deployment Guide

This guide covers advanced deployment options for Router-Maestro, including HTTPS configuration with various DNS providers.

## Table of Contents

- [HTTPS with Traefik](#https-with-traefik)
- [DNS Challenge Providers](#dns-challenge-providers)
- [HTTP Challenge (Alternative)](#http-challenge-alternative)
- [Traefik Dashboard](#traefik-dashboard)
- [Environment Variables Reference](#environment-variables-reference)

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
| `ROUTER_MAESTRO_API_KEY` | API key for Router-Maestro | Yes |
| `CF_DNS_API_TOKEN` | Cloudflare API token (if using Cloudflare) | Conditional |
| `TRAEFIK_DASHBOARD_AUTH` | Basic auth for Traefik dashboard | No |

### Generating a Secure API Key

```bash
openssl rand -hex 32
```

### Complete .env Example

```bash
# Domain configuration
DOMAIN=api.example.com
ACME_EMAIL=admin@example.com

# Router-Maestro
ROUTER_MAESTRO_API_KEY=your_secure_random_key

# Cloudflare (default DNS provider)
CF_DNS_API_TOKEN=your_cloudflare_api_token

# Traefik dashboard (optional)
# Note: $ must be escaped as $$ in .env files
TRAEFIK_DASHBOARD_AUTH=admin:$$2y$$05$$your_bcrypt_hash
```

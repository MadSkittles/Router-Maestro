# Deployment Guide

This guide covers native installation, local Docker, production HTTPS with
Docker Compose and Traefik, remote client setup, upgrades, rollback,
multi-instance reasoning keys, and AI-assisted deployment.

For model, provider, Auto, and client settings, see the
[Configuration Guide](configuration.md). For metrics and request diagnosis,
see [Metrics and Observability](observability.md).

## Choose a Deployment

| Goal | Recommended path | Exposure |
| --- | --- | --- |
| Try Router-Maestro on one machine | Native server | Loopback only |
| Run the packaged image locally | Local Docker | Loopback only |
| Serve multiple trusted client machines | Docker Compose + Traefik | HTTPS |
| Run multiple Router-Maestro replicas | Orchestrator + shared secrets | HTTPS/load balancer |

Do not expose an unauthenticated or plain-HTTP Router-Maestro endpoint to the
internet. The Router-Maestro server key protects both inference and
administration, so possession of it grants configuration authority.

## Prerequisites

Native installation requires:

- Python 3.14;
- [uv](https://docs.astral.sh/uv/) or a compatible installer; and
- a provider credential, such as an active GitHub Copilot subscription.

Container installation requires:

- Docker Engine or Docker Desktop; and
- Docker Compose v2 for the production stack.

Production HTTPS additionally requires:

- a domain whose DNS you control;
- inbound TCP 80 and 443, unless your network design terminates TLS elsewhere;
- a Let's Encrypt contact email; and
- credentials for the configured ACME challenge provider.

## Native Local Server

Install Router-Maestro as an isolated tool:

```bash
uv tool install --python 3.14 router-maestro
```

Start the server on its loopback default:

```bash
router-maestro server start
```

The first start generates and persists a `sk-rm-...` server key in the local
context. Keep the terminal open, then use a second terminal:

```bash
curl http://localhost:8080/health
router-maestro context test
router-maestro auth login github-copilot
router-maestro model list
```

Bind another loopback port with `--port`. Avoid `--host 0.0.0.0` unless a
trusted reverse proxy, firewall, and TLS boundary are already in place.

## Local Docker

Create the bind-mount directories before starting the non-root container:

```bash
mkdir -p ~/.config/router-maestro ~/.local/share/router-maestro
```

Start the published image on IPv4 loopback:

```bash
docker run -d \
  --name router-maestro \
  --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -v ~/.config/router-maestro:/home/maestro/.config/router-maestro \
  -v ~/.local/share/router-maestro:/home/maestro/.local/share/router-maestro \
  likanwen/router-maestro:latest
```

Both mounts matter:

- `.config/router-maestro` stores contexts, the generated server key, providers,
  and routing policy;
- `.local/share/router-maestro` stores provider credentials, the single-instance
  reasoning key file, logs, and optional audit traces.

Because the host and container share these directories, the host CLI can use
the generated `local` context directly:

```bash
curl http://localhost:8080/health
router-maestro context test
router-maestro auth login github-copilot
router-maestro model list
```

If a Linux host uses a different UID and the container cannot write the mounts,
fix ownership narrowly on these two Router-Maestro directories. Do not make the
home directory or config tree world-writable.

To build and run the current checkout instead:

```bash
docker compose -f docker-compose.dev.yml up --build -d
docker compose -f docker-compose.dev.yml logs -f router-maestro
```

The development Compose file also publishes only to `127.0.0.1`.

## Production HTTPS with Docker Compose

The repository's `docker-compose.yml` runs:

```text
Internet → Traefik :443 → router-maestro:8080
```

Traefik handles HTTP-to-HTTPS redirection and Let's Encrypt. Router-Maestro is
reachable only on the internal `web` network and is not directly published on
the host.

### 1. Prepare the host

Clone or copy the repository into a dedicated deployment directory. Create the
persistent directories as the deployment user:

```bash
mkdir -p ~/.config/router-maestro ~/.local/share/router-maestro
```

For production, pin the `router-maestro` service to a tested version instead of
tracking `latest` indefinitely:

```yaml
services:
  router-maestro:
    image: likanwen/router-maestro:<version>
```

### 2. Create `.env`

Keep `.env` owner-readable only and out of version control:

```bash
chmod 600 .env
```

Example values:

```dotenv
DOMAIN=ai.example.com
ACME_EMAIL=admin@example.com

# Leave blank for single-instance first-start generation, or provision a
# stable secret for automation. Never reuse an OpenAI/GitHub/provider key.
ROUTER_MAESTRO_API_KEY=
ROUTER_MAESTRO_LOG_LEVEL=INFO

# Required only when multiple Router-Maestro instances may serve one
# conversation. Use the same value on every replica.
# ROUTER_MAESTRO_REASONING_CAPSULE_KEY=<unpadded-base64url-32-byte-key>
# ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS=<old-key-1>,<old-key-2>

# Default compose ACME provider
CF_DNS_API_TOKEN=<cloudflare-token-with-zone-dns-edit>

# The included compose exposes the Traefik dashboard at traefik.${DOMAIN}.
# Generate with `htpasswd -nB admin` and escape each $ as $$ in this file.
TRAEFIK_DASHBOARD_AUTH=admin:$$2y$$05$$replace-with-bcrypt-hash
```

The Cloudflare token needs `Zone:DNS:Edit` only for the relevant zone. If the
dashboard is not required, remove its Traefik router/middleware labels instead
of publishing it with weak or empty authentication.

### 3. Validate before changing runtime state

Render the fully interpolated Compose model and inspect the image, mounts,
networks, routes, and unresolved variables:

```bash
docker compose config --quiet
docker compose config
```

Do not paste the rendered output into an issue: depending on the Compose
version and configuration, it can include environment values.

### 4. Start and verify

```bash
docker compose pull
docker compose up -d
docker compose ps
docker compose logs --tail=100 router-maestro
```

Wait for certificate issuance and the container health check, then verify:

```bash
curl --fail --show-error https://ai.example.com/health
```

If the server generated its own API key, a human operator can retrieve it from
inside the container:

```bash
docker compose exec router-maestro router-maestro server show-key
```

Treat the output as a secret. Do not include it in logs, screenshots, shell
history, issues, or chat transcripts.

### 5. Add the remote context from a client machine

Install the Router-Maestro CLI on the client machine, then add the HTTPS
deployment using a secure key handoff:

```bash
router-maestro context add production \
  --endpoint https://ai.example.com \
  --api-key 'sk-rm-...'
router-maestro context set production
router-maestro context test
```

Provider authentication happens on the selected server even though the command
runs on the client:

```bash
router-maestro auth login github-copilot
router-maestro model refresh
router-maestro model list
```

Finally, configure the intended clients with the
[Configuration Guide](configuration.md).

## ACME Challenge Options

### Cloudflare DNS challenge (included default)

The included Compose configuration uses:

```yaml
--certificatesresolvers.letsencrypt.acme.dnschallenge.provider=cloudflare
```

This works without exposing a challenge file and supports wildcard
certificates. Supply `CF_DNS_API_TOKEN` with least-privilege DNS edit access.

### Other DNS providers

Traefik uses the lego DNS provider implementations. Replace the provider name
and pass only that provider's required environment values. Examples:

| Provider | Traefik provider | Typical environment variables |
| --- | --- | --- |
| AWS Route53 | `route53` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| DigitalOcean | `digitalocean` | `DO_AUTH_TOKEN` |
| GoDaddy | `godaddy` | `GODADDY_API_KEY`, `GODADDY_API_SECRET` |
| Namecheap | `namecheap` | `NAMECHEAP_API_USER`, `NAMECHEAP_API_KEY` |

Consult the
[Traefik ACME DNS provider documentation](https://doc.traefik.io/traefik/https/acme/#providers)
for exact current variables. Do not grant broader DNS or account permissions
than the provider requires.

### HTTP challenge

When DNS automation is unavailable and public port 80 reaches Traefik, replace
the DNS challenge flags with:

```yaml
- "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
- "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
```

Remove the unused DNS-provider credential from the service. HTTP challenge does
not support wildcard certificates.

For repeated deployment testing, use Let's Encrypt's staging endpoint first to
avoid production rate limits:

```yaml
- "--certificatesresolvers.letsencrypt.acme.caserver=https://acme-staging-v02.api.letsencrypt.org/directory"
```

Remove that line only after the staging flow succeeds.

## Environment Variables

| Variable | Purpose | Required |
| --- | --- | --- |
| `ROUTER_MAESTRO_API_KEY` | Fixed server inference/admin key; blank or unset generates and persists one | No |
| `ROUTER_MAESTRO_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` | No |
| `ROUTER_MAESTRO_REASONING_CAPSULE_KEY` | Current 32-byte unpadded base64url capsule key | Multi-instance only |
| `ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS` | Comma-separated decrypt-only old capsule keys | No |
| `ROUTER_MAESTRO_METRICS_TOKEN` | Independent bearer token for `/metrics` | No |
| `ROUTER_MAESTRO_TRACE` | Set to `1` to enable audit tracing for the process | Debugging only |
| `DOMAIN` | Public hostname used by the included Compose labels | Compose only |
| `ACME_EMAIL` | Let's Encrypt contact | Compose only |
| `CF_DNS_API_TOKEN` | Cloudflare DNS challenge credential | Default Compose only |
| `TRAEFIK_DASHBOARD_AUTH` | bcrypt basic-auth entry for the included dashboard | Included Compose |

Custom-provider API-key environment variables are documented in the
[Configuration Guide](configuration.md#custom-openai-compatible-providers).
The included Compose file forwards the server key, log level, and reasoning
capsule variables. Add `ROUTER_MAESTRO_METRICS_TOKEN`, a temporary
`ROUTER_MAESTRO_TRACE`, or custom-provider credential variable explicitly to
the `router-maestro.environment` list when that deployment needs it. Values in
Compose `.env` are interpolation inputs; they are not injected into a container
unless the service declares them.

## Reasoning Capsule Keys

Cross-protocol reasoning replay can contain opaque provider state. Router-Maestro
seals that state in an authenticated `rmr1` capsule bound to its provider,
model, transport, and reasoning item. Invalid keys, corrupted key files,
tampering, and provenance mismatch fail closed before upstream I/O.

### Single instance

When no capsule environment variable is configured, Router-Maestro atomically
generates and reuses:

```text
~/.local/share/router-maestro/reasoning-capsule-keys.json
```

It must remain owner-only (`0600` on POSIX). An invalid or unreadable file
causes startup to fail; Router-Maestro does not silently replace it and break
active conversations. Persist the data directory across container replacement.

### Multiple instances

Every replica behind one load balancer must receive the same
`ROUTER_MAESTRO_REASONING_CAPSULE_KEY` from the deployment secret manager. It is
the unpadded URL-safe base64 encoding of exactly 32 random bytes. Do not rely on
separate container-local key files: a later turn may reach another replica.

Rotate keys in this order:

1. Install the new current key on every replica.
2. Add the former current key to
   `ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS` everywhere.
3. Verify the entire replica set has an identical key set before new traffic.
4. After the maximum replay lifetime, remove expired previous keys.

Previous keys decrypt only; new capsules always use the current key. Never log
the keys, capsule plaintext, or decrypted provider state.

## Upgrade

Before upgrading:

1. read [CHANGELOG.md](../CHANGELOG.md);
2. record the current image tag and digest;
3. make a protected backup of Router-Maestro config and data;
4. retain the existing server and reasoning keys; and
5. validate the target image in a test context when protocol behavior changed.

Before upgrading from 0.9 to 1.0.0, scan client and reverse-proxy
configuration for the three deprecated Router-Maestro aliases and replace them:

| Removed in 1.0.0 | Stable replacement |
| --- | --- |
| `/api/openai/beta/v1/responses` | `/api/openai/v1/responses` |
| `/api/anthropic/beta/v1/messages` | `/api/anthropic/v1/messages` |
| `/api/anthropic/beta/v1/messages/count_tokens` | `/api/anthropic/v1/messages/count_tokens` |

Gemini's `/api/gemini/v1beta` path is unaffected because `v1beta` identifies
the Gemini API version.

For a pinned Compose deployment, update only the image tag, then:

```bash
docker compose pull router-maestro
docker compose up -d --no-deps router-maestro
docker compose ps
docker compose logs --tail=100 router-maestro
curl --fail --show-error https://ai.example.com/health
```

From a client machine:

```bash
router-maestro context set production
router-maestro context test
router-maestro model refresh
router-maestro model list
```

Open fresh Claude Code, Codex, or Gemini sessions after client config or Codex
catalog changes. For a release or protocol migration, follow the live-client
validation process in [CONTRIBUTING.md](../CONTRIBUTING.md#live-provider-and-client-validation).

## Rollback

Rollback should preserve the mounted config and data directories:

1. restore the previous image tag in `docker-compose.yml`;
2. run `docker compose up -d --no-deps router-maestro`;
3. verify container health and public `/health`;
4. run `router-maestro context test` and `router-maestro model list`; and
5. verify one fresh client request.

Do not delete the data volume as part of an image rollback. If a release changed
an on-disk format, follow its changelog migration notes and restore the protected
backup only when required.

## Operational Verification Checklist

A deployment is not complete merely because the container is running. Verify:

- the intended image tag and digest are active;
- the container reports healthy;
- public `/health` returns 200 through the real hostname;
- the remote context succeeds;
- provider authentication is present;
- the authenticated model catalog loads;
- a fresh client performs a text request through the stable endpoint;
- tool and multi-turn behavior are tested when the release changed protocol
  conversion or reasoning replay; and
- logs contain no startup key-file, provider, or routing errors.

Use audit only for a scoped investigation, then disable it. Traces can contain
prompts and outputs. See [Contributing: Audit tracing](../CONTRIBUTING.md#audit-tracing-for-development)
and [Metrics and Observability](observability.md#request-lifecycle-and-audit-traces).

## AI-Assisted Deployment

These prompts are intended for a coding agent with terminal access. They make
the expected authority and verification explicit. Replace every placeholder
before use.

### Local Docker prompt

```text
Deploy Router-Maestro <version> locally with Docker on this machine and
configure <claude-code|codex|gemini> at <user|project> scope.

Requirements:
1. Inspect the OS/architecture, Docker and Compose versions, occupied ports,
   existing Router-Maestro containers, current contexts, and existing config
   directories before changing anything. Preserve unrelated files and changes.
2. Bind the API only to 127.0.0.1:8080. Persist both Router-Maestro config and
   data directories. Use a version-pinned image unless I explicitly request
   latest.
3. Never print, log, commit, or send API keys, OAuth tokens, .env contents,
   reasoning keys, or complete secret-bearing config files. Mask secrets in the
   report. Do not create broad filesystem permissions.
4. Start the container, wait for its health check, and verify /health. Then run
   `router-maestro context test`.
5. If GitHub Copilot is not authenticated, pause for me to complete the device
   authorization. Do not attempt to bypass browser or account security.
6. Refresh and inspect the live model catalog, run the official
   `router-maestro config <client>` flow, preserve/backup existing client
   config, and prefer provider-qualified IDs. Refresh the Codex model catalog
   when Codex is selected.
7. Start a fresh client session and run one harmless text request. Do not enable
   audit unless a failure needs scoped diagnosis.
8. Report the image tag/digest, bind address, context, model, config target,
   backup path, and every verification result without exposing secrets.
```

### Remote HTTPS prompt

```text
Deploy Router-Maestro <version> to <ssh-host> as the <context-name> instance,
using <deployment-directory>, Docker Compose, Traefik, and HTTPS for
<domain>. Use <cloudflare|route53|digitalocean|godaddy|namecheap|http> ACME
validation. Do not modify any other Router-Maestro instance.

Requirements:
1. Verify the SSH host identity. If its host key changed, stop and ask me; do
   not bypass the warning. Inspect OS/architecture, Docker/Compose, disk space,
   ports 80/443, DNS resolution, current containers, image tag/digest,
   deployment files, mounts, networks, and health before changing state.
2. Preserve the existing .env, compose file, config, data, server API key,
   reasoning key, and unrelated user changes. Create protected backups before
   editing. Never print or return secret values or complete secret-bearing
   files. Use the existing secret manager or request a non-echoing handoff.
3. Pin `likanwen/router-maestro:<version>`. Ensure Traefik and Router-Maestro
   share the configured Docker network and route to container port 8080. Do not
   publish Router-Maestro directly. Use least-privilege ACME credentials.
4. Run `docker compose config --quiet`, pull/build the image as requested, and
   update only the Router-Maestro service. Do not delete volumes or the old
   image needed for rollback.
5. Wait for container health and HTTPS certificate readiness. Verify the real
   domain /health endpoint, the selected client context, provider auth status,
   and the authenticated model catalog. If Copilot authorization is required,
   pause for my device-flow action.
6. Configure <claude-code|codex|gemini> on <client-machine> only after the
   server checks pass. Use stable Router-Maestro endpoints and refresh the
   Codex model catalog when applicable.
7. Run one fresh client smoke request. For protocol/dispatcher changes, run the
   repository live-validation workflow and correlate request IDs. Enable audit
   only if needed, disable it immediately afterward, and retain or remove
   traces only with explicit scope.
8. On failure, collect sanitized diagnostics and either leave the existing
   healthy instance untouched or roll back to the recorded image. Report the
   host, image tag/digest, public endpoint, context, health, catalog/client test,
   and rollback readiness without secrets.
```

An AI agent still needs a human for OAuth authorization, DNS/account access not
already available, and any ambiguous destructive or production-wide action.

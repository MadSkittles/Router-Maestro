# Configuration Guide

This guide covers Router-Maestro contexts, provider authentication, model IDs,
client configuration, the local Web portal, Auto routing, runtime policy, and
custom providers. For installation and server operations, see the
[Deployment Guide](deployment.md).

The recommended workflow is:

1. Add or select a deployment context.
2. Test the context before changing a client.
3. Authenticate providers on that server.
4. Inspect the server's live model catalog.
5. Generate user- or project-level client configuration.
6. Start a fresh client session and run a real request.

## Configuration Model

A **context** is a named Router-Maestro server:

```text
context = endpoint + Router-Maestro server API key
```

The active context determines which server receives CLI administration,
provider login, model-list, Auto, and client-configuration operations. Provider
credentials and routing configuration belong to the selected server; client
configuration belongs to the machine where the client runs.

Router-Maestro's server key is not an OpenAI, Anthropic, Gemini, or GitHub
credential. It protects both inference and `/api/admin/*`, so treat it as
configuration authority as well as inference access.

## Contexts

The first native or volume-mounted local server start creates a `local` context
at `http://localhost:8080` and persists its generated server key. Inspect it
without exposing the key:

```bash
router-maestro context list
router-maestro context current
router-maestro context test
```

Add a remote deployment, switch to it, and verify it:

```bash
router-maestro context add production \
  --endpoint https://ai.example.com \
  --api-key 'sk-rm-...'
router-maestro context set production
router-maestro context test
```

Other context operations:

```bash
router-maestro context update production --endpoint https://new-ai.example.com
router-maestro context update production --api-key 'sk-rm-...'
router-maestro context set local
router-maestro context remove production
```

Prefer HTTPS for a server reached over a network. Do not put the API key in
shell history on shared systems; use an environment variable, password manager,
or a non-echoing secret handoff when possible.

## Provider Authentication

Authentication is performed against the active context:

```bash
router-maestro auth list
router-maestro auth login github-copilot
router-maestro auth login openai
router-maestro auth login anthropic
router-maestro auth login deepseek
```

GitHub Copilot uses an OAuth device flow. The terminal prints a verification URL
and short-lived user code; complete that step in a browser. OpenAI, Anthropic,
DeepSeek, and authenticated custom providers prompt for an API key.

After login, refresh and inspect the live catalog:

```bash
router-maestro model refresh
router-maestro model list
```

The Copilot catalog is account- and rollout-dependent. “Full catalog” means all
models the GitHub Copilot service currently exposes to the authenticated
account; Router-Maestro does not invent or unlock models absent from that
catalog.

## Model IDs and Context Windows

Provider-qualified IDs are the safest public form:

```text
github-copilot/gpt-5.6-terra
openai/gpt-5.6-terra
anthropic/claude-opus-4.6
deepseek/deepseek-v4-pro
```

They remain unambiguous when two providers expose the same upstream ID. The
client wizards ask whether to keep the provider prefix as their final model-ID
choice. Bare IDs are convenient but can become ambiguous as providers change.

The live catalog may advertise more than one prompt window, for example:

```text
272K / 1M
```

For Claude Code, selecting the 1M choice adds the client-side `[1m]` model hint
and configures its auto-compact window. The suffix does not select a different
upstream model and is not used by Router-Maestro's server-side Auto routing.
Codex obtains window metadata through its generated model catalog; it does not
need `[1m]` in the model ID.

## Configure a Client with the CLI

Run one of the interactive wizards:

```bash
router-maestro config claude-code
router-maestro config codex
router-maestro config gemini
router-maestro config dsh
```

Or run `router-maestro config` to choose the client interactively. Each wizard:

- chooses a supported scope (DSH is user-only; the other clients support user or project);
- offers to back up an existing target file;
- fetches the active server's current model catalog;
- selects a model and relevant context window;
- asks whether to keep the provider-qualified model ID when the client supports both forms; and
- preserves unrelated configuration fields where the client format permits it.

The optional `--id-style` flag makes model-ID spelling non-interactive:

```bash
router-maestro config codex --id-style qualified
router-maestro config claude-code --id-style bare
```

`qualified` is recommended. `official` exists for legacy native-vendor spelling
compatibility and is not needed for normal Router-Maestro use.

### Claude Code

Targets:

| Scope | File |
| --- | --- |
| User | `~/.claude/settings.json` |
| Project | `./.claude/settings.json` |

The generated environment points Claude Code at the stable Anthropic surface:

```text
<context endpoint>/api/anthropic
```

The main model is stored in `ANTHROPIC_MODEL`. When the live catalog has no
Claude-family models, the wizard also asks for Fable, Opus, Sonnet, Haiku, and
subagent mappings so Claude Code can use non-Claude models consistently. These
mappings work at user or project scope. The obsolete
`ANTHROPIC_SMALL_FAST_MODEL` entry is removed.

Choosing a 1M context window affects Claude Code's client hint and compacting
behavior. It does not create a synthetic model in Router-Maestro.

### OpenAI Codex

Targets:

| Scope | File |
| --- | --- |
| User | `~/.codex/config.toml` |
| Project | `./.codex/config.toml` |

User-level configuration defines the Router-Maestro provider using the stable
Responses base URL:

```text
<context endpoint>/api/openai/v1
```

The generated provider reads the server key from
`ROUTER_MAESTRO_API_KEY`. Make it available to Codex through its launch
environment or existing secret-management mechanism. Do not hardcode it into a
committed project file.

Each configuration run can refresh
`~/.codex/router-maestro-models.json` from the active server. Keep the default
**Yes** unless a deliberately pinned catalog is required. Codex reads
`model_catalog_json` at startup, so open a new session after an update.

Generated catalog entries keep Codex's two window fields distinct:

- `context_window` is the upstream model's maximum prompt/input budget;
- `max_context_window` is its combined prompt + output capacity.

For example, a 1.05M GPT model with a 128K output allowance is written as
`context_window: 922000` and `max_context_window: 1050000`. User-level
configuration removes stale global `model_context_window` and
`model_auto_compact_token_limit` overrides so they cannot mask per-model
catalog values. Project-level overrides are left untouched.

Generated Router-Maestro GPT display names use a space after `GPT` instead of a
hyphen, for example `GPT 5.6 Sol`. This keeps their GPT prefix visible and
distinguishes them from Codex's bundled GPT labels. Model IDs and bundled model
entries remain unchanged. Generated display names and descriptions also omit a
trailing `(Internal only)` marker.

Codex Auto-review (guardian) requests use the internal model ID
`codex-auto-review`. The Copilot provider maps that ID to the bare upstream
model ID `gpt-6-luna` (without a provider prefix). This provider-owned mapping is
defined in `COPILOT_MODEL_ALIASES`, not a runtime configuration option; it
does not follow the client's selected conversation model. The alias is not
advertised in Router-Maestro's public model catalog.

Current Codex versions accept the provider definition only at user scope.
Project configuration can override `model` and the model-catalog path, but it
inherits the user-level provider. Therefore:

1. configure the intended Router-Maestro context at user scope first;
2. configure the project model second; and
3. keep the project context consistent with the inherited user provider.

### Gemini CLI

Targets:

| Scope | File |
| --- | --- |
| User | `~/.gemini/.env` |
| Project | `./.gemini/.env` |

The generator sets the stable Router-Maestro Gemini base, server key, selected
model, and disables Gemini telemetry in the generated file:

```text
GOOGLE_GEMINI_BASE_URL=<context endpoint>/api/gemini
GEMINI_MODEL=<selected model>
```

The API path still contains `v1beta` when Gemini CLI calls generation methods;
that is the Gemini protocol version, not a Router-Maestro beta endpoint.

### 0.9 endpoint migration

0.9 is the last minor release that retains Router-Maestro's old beta aliases.
Before 1.0.0, migrate any manually maintained client configuration as follows:

| Deprecated alias | Stable replacement |
| --- | --- |
| `/api/openai/beta/v1/responses` | `/api/openai/v1/responses` |
| `/api/anthropic/beta/v1/messages` | `/api/anthropic/v1/messages` |
| `/api/anthropic/beta/v1/messages/count_tokens` | `/api/anthropic/v1/messages/count_tokens` |

Current CLI and Web configuration already generate stable paths. Gemini's
`/api/gemini/v1beta` remains because `v1beta` is the Gemini API version.

### DeepSeek Harness (DSH)

DSH configuration targets only `~/.dsh/settings.yaml`; DSH does not currently
load a project-level settings file. The CLI and Web Portal preserve unrelated
top-level settings and replace only:

- `llm-pi-ai.providers.router-maestro`, containing the current context's full
  model catalog; and
- `agent-default-model`, containing the selected Router-Maestro model.

The generated provider uses `api: openai-responses`, the stable
`<context endpoint>/api/openai/v1` base URL, and resolves its key through
`ROUTER_MAESTRO_API_KEY`. Model IDs stay provider-qualified to avoid collisions.
DSH defines `contextWindow` as the combined request + response capacity, so a
GPT model with a 922K prompt budget and 128K output allowance is written as
`contextWindow: 1050000`. Router-Maestro does not set DSH `maxTokens`, because
that field would also become the default output limit on requests.

The same rule applies to DeepSeek: its 1M total window remains
`contextWindow: 1000000` in DSH. The 616K value obtained by reserving the full
384K maximum output allowance is a Codex prompt-budget convention only; Codex
catalog generation writes `context_window: 616000` together with
`max_context_window: 1000000`.

#### Use DSH's native DeepSeek adapter through Router-Maestro

Router-Maestro also has a built-in `deepseek` provider. Its OpenAI Chat,
OpenAI Responses, and Anthropic Messages bindings use DeepSeek's corresponding
native upstream endpoint. Matching ingress protocols stay on the raw identity
path; Gemini ingress is converted to DeepSeek Chat Completions.

First save the DeepSeek upstream key on the Router-Maestro server:

```bash
router-maestro auth login deepseek
```

Then edit DSH's `DeepSeek` (`deepseek-official`) provider and set:

```text
API key:  <the Router-Maestro server API key, not the DeepSeek key>
Base URL: https://<router-maestro-host>/api/openai/v1
Models:   deepseek-flash
          deepseek-v4-pro
```

`deepseek-flash` is the canonical API ID for multimodal DeepSeek-V4.1-Flash.
The retired `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` IDs remain
accepted as compatibility aliases and route to `deepseek-flash`, but are not
listed as current models. DSH appends `/chat/completions` to the Base URL.
Router-Maestro pins these official bare IDs to its `deepseek` provider,
replaces the downstream authorization header with the server-side DeepSeek
credential, and preserves DeepSeek-specific Chat fields such as
`reasoning_content`, `thinking`, and future extension fields. The provider
catalog also exposes the unambiguous `deepseek/<model>` form to other clients.
The Anthropic-compatible
`/messages/count_tokens` route also calls DeepSeek's native exact-count endpoint
and falls back to local estimation only if that upstream helper is unavailable.

When DeepSeek rejects a request for exceeding the model context window,
Router-Maestro returns a safe `context_length_exceeded` error and the
`X-Router-Maestro-Error-Signal: context_window_exceeded` header. The provider's
request-specific token counts are not copied downstream. DSH uses the
structured classification to run its context-overflow compaction and retry
flow.

The same Base URL exposes DeepSeek's OpenAI-compatible Files API at
`/api/openai/v1/files`. DSH can upload a vision image, reuse its `file-api-*`
ID, list or inspect stored files, and delete files without falling back to
inline base64. These operations always target the built-in DeepSeek provider
and its server-side credential; they do not participate in model routing or
provider fallback. DeepSeek currently accepts JPEG, PNG, GIF, and WebP uploads
up to 64 MiB, with either permanent storage or an optional lifetime from one
hour to 30 days.

## Configure with the Local Web Portal

Start the portal on the client machine:

```bash
router-maestro web
```

It binds to `127.0.0.1:8765` by default and opens a browser. Use
`--no-open` or a different loopback port when needed:

```bash
router-maestro web --no-open --port 8876
```

The portal can:

- switch among local and remote contexts;
- measure public `/health` round-trip time;
- load the selected context's authenticated model catalog;
- show model provider, context windows, and transport capability summary;
- configure Claude Code, Codex, or Gemini CLI at user or project scope;
- configure DSH at its supported user scope;
- discover project roots from client trust stores and explicit additions;
- preview changes without writing;
- back up an existing configuration before Apply;
- refresh the Codex model catalog; and
- configure Smart Auto or a Priority Chain.

The portal is deliberately local-only. It is not served by the Router-Maestro
API server and does not expose a remote management website. **Copy Key** reads a
context key only for the explicit clipboard action.

## Router-Maestro Auto

The virtual public model ID is:

```text
router-maestro
```

Configure it with:

```bash
router-maestro model auto show
router-maestro model auto configure
```

The same settings are available after selecting Router-Maestro Auto in the Web
portal.

### Smart Auto

Smart Auto is the default. It has one router model and four task-model slots:

| Task | Intended use |
| --- | --- |
| Fast | Short, low-latency requests |
| General | Everyday analysis and conversation |
| Coding | Implementation, debugging, and code review |
| Deep Reasoning | Hard, multi-step reasoning |

The router model is used only to classify a bounded task type. It cannot invent
a model ID; Router-Maestro selects the configured model for that task.

Before dispatch, candidates are filtered against required request features and
estimated input size. The default context policy treats a candidate as safe
only below 70% of its advertised prompt capacity. If the safety filter would
remove every candidate, Router-Maestro retains all hard-compatible configured
models tied for the largest context window. A precise Copilot prompt-overflow
response may trigger a larger configured fallback before the first response
frame. No candidate or model is replayed after stream commitment.

The capability policy controls unknown catalog claims:

- **Exclude unknown models (`strict`)**: a required capability must be
  explicitly supported.
- **Allow unknown models (`optimistic`)**: only explicitly unsupported models
  are removed.

### Priority Chain

Priority Chain mode follows one user-configured ordered list. The chain must
contain at least one concrete provider-qualified model and cannot contain
duplicates. Router-Maestro does not append unconfigured catalog models.

## Direct Priorities and Fallback

For explicit-model routing outside the virtual Auto model:

```bash
router-maestro model priority list
router-maestro model priority add github-copilot/gpt-5.6-terra
router-maestro model priority remove github-copilot/gpt-5.6-terra
router-maestro model fallback show
router-maestro model fallback set --strategy priority --max-retries 2
```

Supported fallback strategies are `priority`, `same-model`, and `none`.
Transport switching for one model is not counted as a model fallback.

## Stream Guards and Anthropic Beta Header Filtering

Stream guards are enabled by default in runtime configuration:

- **Leak Guard** detects known provider control envelopes and leaked XML tool
  invocations. Control-envelope leaks terminate the stream; recoverable invoke
  leaks are projected back into structured tool calls.
- **Runaway Guard** terminates degenerate streams that exceed the configured
  byte ceiling or sustain excessive tiny fragments.

`beta_strip` is separate from Router-Maestro endpoint stability. It removes
matching tokens from an inbound `anthropic-beta` header before provider I/O;
the Gemini `/v1beta` protocol path is unrelated.

The runtime fields are:

```json
{
  "guards": {
    "leak_guard": {"enabled": true},
    "runaway_guard": {
      "enabled": true,
      "max_bytes": 10000000,
      "max_deltas": 50000
    }
  },
  "beta_strip": ["output-128k-*"]
}
```

Update these fields through the revision-aware admin configuration rather than
replacing `priorities.json` underneath a running server. Preserve all unrelated
Auto, fallback, model override, thinking, and audit members. Contributors can
reuse the compare-and-swap pattern in
[Audit tracing for development](../CONTRIBUTING.md#revision-aware-runtime-method).

## Custom OpenAI-Compatible Providers

Custom providers are server-side configuration in
`~/.config/router-maestro/providers.json`:

```json
{
  "providers": {
    "ollama": {
      "type": "openai-compatible",
      "baseURL": "http://localhost:11434/v1",
      "models": {
        "llama3": {"name": "Llama 3"},
        "mistral": {"name": "Mistral 7B"}
      },
      "options": {
        "allow_unauthenticated": true
      }
    }
  }
}
```

For an authenticated custom provider, credentials resolve in this order:

1. the configured environment variable;
2. a key saved with `router-maestro auth login <provider>`; or
3. no key only when `allow_unauthenticated` is explicitly `true`.

The default environment name is the provider ID converted to uppercase and
underscores plus `_API_KEY` (`my-provider` becomes
`MY_PROVIDER_API_KEY`). Override it with `options.api_key_env`.

Custom OpenAI-compatible providers expose Chat Completions by default. If the
upstream implements native `/responses`, set `options.responses` to `true` to
register that transport as well. This preserves native Responses items, unknown
wire fields, and continuation state instead of converting them to Chat. Do not
enable it for a service that only implements Chat. The built-in OpenAI provider
registers both transports without additional configuration.

Application-defined provider types and resource endpoints can be registered
through the [Provider Plugins](provider-plugins.md) API. Registration is explicit
at application startup; a `providers.json` entry does not load arbitrary Python
code or install a package.

When Router-Maestro runs in Docker, the variable must be passed into the
`router-maestro` container; defining it only in the host shell or Compose
`.env` file does not automatically expose it to the service. For example:

```yaml
services:
  router-maestro:
    environment:
      - MY_PROVIDER_API_KEY
```

After changing providers, force immediate model rediscovery:

```bash
router-maestro model refresh
```

In a managed server, runtime Auto/priorities changes should go through the Web
portal, CLI, or revision-aware admin API so the active Router generation is
atomically replaced. Editing `priorities.json` behind a running process is not
the recommended control path.

## Files and Ownership

Router-Maestro follows XDG paths on Unix-like systems:

| Path | Purpose | Contains secrets |
| --- | --- | --- |
| `~/.config/router-maestro/contexts.json` | Deployment endpoints and server keys | Yes |
| `~/.config/router-maestro/providers.json` | Custom provider definitions | Usually no |
| `~/.config/router-maestro/priorities.json` | Auto, fallback, guards, audit policy | No prompts, but operationally sensitive |
| `~/.config/router-maestro/projects.json` | Projects explicitly added to the portal | No |
| `~/.local/share/router-maestro/auth.json` | Provider OAuth/API credentials | Yes |
| `~/.local/share/router-maestro/reasoning-capsule-keys.json` | Single-instance reasoning capsule keys | Yes |
| `~/.local/share/router-maestro/traces/` | Opt-in request audit traces | Yes—may contain prompts and outputs |

On Windows, Router-Maestro uses `%LOCALAPPDATA%\router-maestro`. Respect owner-only
permissions and never commit contexts, credentials, capsule keys, or traces.

## AI-Assisted Client Configuration

The following prompt is designed for a coding agent with terminal access. Fill
in the placeholders, then give the prompt to the agent from the machine where
the client runs.

```text
Configure Router-Maestro for <claude-code|codex|gemini|dsh> at <user|project>
scope using context <context-name>. Work in <project-path> when project scope is
requested.

Requirements:
1. Inspect `router-maestro --version`, the active context, and existing target
   config before changing anything. Preserve unrelated settings and user edits.
2. Never print, log, commit, or paste API keys, OAuth tokens, capsule keys, or
   complete secret-bearing config files. Mask secrets in all summaries.
3. If the context does not exist, ask me for the endpoint and arrange a
   non-echoing secret handoff for its Router-Maestro API key. Do not invent a
   key or put it in a tracked file.
4. Run `router-maestro context set <context-name>`, `router-maestro context
   test`, and `router-maestro model list`. Stop if the server or catalog is not
   healthy.
5. Run the official `router-maestro config <client>` flow. Prefer
   provider-qualified model IDs, back up an existing target, and for Codex
   refresh `router-maestro-models.json`. Configure Codex user scope before
   project scope when the Router-Maestro provider is not already defined.
6. Do not enable audit unless diagnosing a specific failure. Do not modify the
   server deployment or another context.
7. Start a fresh client process and perform one harmless text request. Report
   the selected context, model ID, target file, backup path, and verification
   result without exposing secrets.
```

For end-to-end server installation, use the prompts in the
[Deployment Guide](deployment.md#ai-assisted-deployment).

## Troubleshooting

### `401` from inference or admin APIs

The client is sending a Router-Maestro server key that does not match the
selected deployment. Check `router-maestro context current`, update the context
key, then regenerate the client configuration. Do not substitute a provider API
key.

### A model appears in one context but not another

Catalogs are server- and account-scoped. Check the active context and its
authenticated providers, then run `router-maestro model refresh`.

### Codex warns that model metadata is missing

Re-run `router-maestro config codex`, allow the catalog update, and open a fresh
Codex session. Confirm its `model_catalog_json` points to the generated
`router-maestro-models.json`.

### A project-level Codex model uses the wrong server

Codex inherits the provider endpoint from user scope. Configure the intended
Router-Maestro context at user level, then apply the project model override.

### Cross-protocol input is rejected as unrepresentable

The target transports cannot preserve one or more explicit request fields.
Choose a model with a compatible upstream transport or remove the unsupported
feature. Router-Maestro intentionally does not silently discard it. For a
protocol investigation, follow the safe audit procedure in
[CONTRIBUTING.md](../CONTRIBUTING.md#audit-tracing-for-development).

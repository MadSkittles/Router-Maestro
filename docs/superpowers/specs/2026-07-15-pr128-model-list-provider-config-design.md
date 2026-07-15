# PR #128 Model Listing and Provider Config Compatibility Design

## Context

PR #128 introduced application-owned `RouterOwner` generations for request
lifecycle and credential/configuration mutations. Inference requests and Admin
model listing use those generations, but the public OpenAI and Anthropic model
listing routes still resolve the legacy module singleton. A successful Admin
login, logout, priority update, or model refresh therefore leaves those two
catalogs stale until independent caches expire.

The same PR replaced the previously open-ended custom-provider `options` map
with two typed fields. Unknown keys in an existing `providers.json` now fail
whole-document validation; the generic loader then returns the empty default,
temporarily removing every custom provider from routing and discovery.

## Requirements

1. OpenAI, Anthropic, and Admin model listing must read one application-owned
   Router generation under a lease and must never consult the module singleton
   in normal FastAPI operation.
2. A listing that began before a generation swap may finish on its leased old
   generation; a listing beginning after the swap must see the new generation.
   Provider resources close only after the last old lease is released.
3. Listing success and failure must release the lease exactly once.
4. Unknown legacy custom-provider option keys must survive load and save without
   becoming active runtime behavior or silently disappearing.
5. One invalid custom-provider entry must not erase unrelated valid entries.
   Diagnostics identify provider and field paths without logging input values.
6. Existing typed option validation, reserved-name validation, credential
   precedence, inference behavior, and public response formats remain intact.
7. The fix lands through a pull request but is not merged as part of this task.

## Design

### Application-owned model catalog dependency

Create a small shared server dependency module. `get_router_owner()` reads the
owner from `request.app.state`; `get_app_router()` acquires the current Router
lease and yields its Router, releasing the lease in `finally`. Move the runtime
configuration dependency there as well so Admin routes do not own a dependency
needed by other route modules.

Inject this Router into all three model-list endpoints. Do not add model listing
to `RequestContextMiddleware`: listing is not inference and should not acquire
inference audit/terminal state. Do not reset the module singleton from
`RouterOwner.rebuild()`: the singleton has no lease/refcount relationship with
the application owner and cannot safely retire provider resources.

Keep `get_router()` and `reset_router()` as legacy/standalone compatibility APIs
for now. Normal server model listing no longer uses their fallback branch.

### Compatible custom-provider loading

Keep `api_key_env` and `allow_unauthenticated` typed, but configure
`CustomProviderOptions` to retain unknown extras. Runtime credential resolution
continues to read only the two known fields. Model serialization includes the
extras, so a future save cannot silently delete legacy data.

Specialize `load_providers_config()` instead of weakening the generic config
loader used by unrelated files. It reads the top-level document, validates each
provider independently, skips invalid entries, and returns all healthy entries.
Warnings contain only provider names, validation locations/codes, and unknown
option key names—never option values or Pydantic input echoes. Invalid JSON or
an invalid top-level shape still falls back to the empty default without
overwriting the source file.

## Verification

- Red/green route tests prove both public catalogs switch from generation A to
  B immediately and never use `_router_instance`.
- Concurrency tests prove an old generation remains alive until its listing
  lease finishes while a new request observes the replacement.
- Dependency tests prove release on listing exceptions.
- Config tests prove unknown option round-trip, known-field validation, one bad
  provider isolation, safe diagnostics, and unchanged source files.
- Run focused suites, all unit tests, Ruff/format, compile, lock, build, wheel
  install smoke, and a controlled real-app Admin-login/listing regression.

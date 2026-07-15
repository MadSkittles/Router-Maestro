# PR #128 Model Listing and Provider Config Compatibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the split Router ownership from model listing and restore safe compatibility for existing custom-provider option data.

**Architecture:** FastAPI model-list routes acquire the active application-owned Router through one yield dependency, preserving generation leases and resource retirement. Provider configuration loading retains unknown legacy option data and validates providers independently so one invalid entry cannot erase healthy entries.

**Tech Stack:** Python 3.11+, FastAPI dependency injection, Pydantic v2, pytest/pytest-asyncio, Ruff, uv.

---

### Task 1: Make every server model listing use a RouterOwner lease

**Files:**
- Create: `src/router_maestro/server/dependencies.py`
- Modify: `src/router_maestro/server/routes/models.py`
- Modify: `src/router_maestro/server/routes/anthropic.py`
- Modify: `src/router_maestro/server/routes/admin.py`
- Modify: `tests/test_models_route.py`
- Modify: `tests/test_anthropic_models.py`
- Modify: `tests/test_admin_routes.py`

- [x] **Step 1: Write failing generation-consistency tests**

Add HTTP-level tests with an application-owned `RouterOwner`. Prime a legacy
singleton with stale data, assert both public routes return generation A, call
`owner.rebuild()` with generation B, and assert both routes immediately return
B while the legacy singleton is never called.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/test_models_route.py tests/test_anthropic_models.py -q
```

Expected: the new tests fail because both public endpoints still call
`get_router()` and ignore `app.state.router_owner`.

- [x] **Step 3: Add the shared lease dependency and inject it**

Implement `get_router_owner()`, `get_runtime_config_repository()`, and an async
generator `get_app_router()` that acquires a lease and releases it in `finally`.
Inject its yielded Router into the OpenAI, Anthropic, and Admin listing routes.
Keep Admin refresh and mutation operations on the owner itself.

- [x] **Step 4: Add lifecycle/error coverage**

Prove that an old listing blocked on generation A keeps A open while a rebuild
installs B, a new listing sees B, and A closes exactly once after release. Prove
that `list_models()` exceptions still release the lease.

- [x] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_models_route.py tests/test_anthropic_models.py tests/test_admin_routes.py tests/test_router_lifecycle.py -q
```

Expected: all selected tests pass with no singleton use in server listing.

### Task 2: Preserve legacy provider options and isolate invalid providers

**Files:**
- Modify: `src/router_maestro/config/providers.py`
- Modify: `src/router_maestro/config/settings.py`
- Modify: `tests/test_custom_provider_auth.py`
- Modify: `tests/test_config.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [x] **Step 1: Write failing compatibility tests**

Write tests that load two custom providers from disk where one contains an
unknown option. Assert both load, the unknown key survives `model_dump()` and
save/reload, and the file is not changed merely by loading. Add a second test
where one provider has an invalid known field; assert the healthy provider is
retained and logs contain the provider/field path but not the invalid value.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/test_custom_provider_auth.py tests/test_config.py -q
```

Expected: unknown options raise `extra_forbidden` and a single validation error
causes `providers={}`.

- [x] **Step 3: Retain compatibility extras**

Set `CustomProviderOptions` to allow and retain extras while continuing to
validate `api_key_env` and `allow_unauthenticated`. Runtime code must continue
to consume only the typed fields.

- [x] **Step 4: Add a provider-specific resilient loader**

Implement `load_providers_config()` as a safe per-provider parser. Validate
top-level structure and provider identifiers, skip only invalid entries, and
log sanitized field locations/codes. Preserve missing-file creation and corrupt
JSON fallback behavior without rewriting existing invalid files.

- [x] **Step 5: Document the compatibility contract**

Document that unknown legacy option keys are preserved but ignored at runtime,
and record both regression fixes under `CHANGELOG.md` `Unreleased`.

- [x] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_custom_provider_auth.py tests/test_config.py tests/test_admin_routes.py tests/test_integration_harness.py -q
```

Expected: all selected tests pass, including safe diagnostics and round-trip.

### Task 3: Review, full verification, and PR handoff

**Files:**
- Review all files changed by Tasks 1–2.

- [x] **Step 1: Run independent specification and quality reviews**

Review the complete diff against the design. Fix all Critical/Important issues
and re-review affected boundaries.

- [x] **Step 2: Run full local gates**

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/ integration_tests/
uv run ruff format --check src/ tests/ integration_tests/
uv run python -m compileall -q src/router_maestro
uv lock --check
uv run router-maestro --help
uv build
git diff --check
```

- [x] **Step 3: Run controlled end-to-end regression**

Start a real temporary-XDG Router-Maestro app, prime public model listing,
perform Admin login for a configured custom provider, and prove OpenAI,
Anthropic, and Admin listings all immediately expose the same model. Verify the
real user auth file hash is unchanged.

- [x] **Step 4: Verify the built wheel**

Install the wheel into a fresh CPython 3.11 environment, run CLI help, and
import the shared server dependency and provider config models from installed
`site-packages`.

- [ ] **Step 5: Commit, push, and open the PR without merging**

Explicitly stage only reviewed paths, commit, push the fix branch, and create a
PR against `master` containing the two root causes, compatibility behavior, and
exact verification results. Confirm the PR remains `OPEN`; do not merge, tag,
publish, or release.

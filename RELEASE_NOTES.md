# Release Notes

## v0.1.19 (2026-03-09)

### Features
- Add Gemini-compatible API routes (PR #26)
- Add tool calling support to OpenAI-compatible endpoint
- Add Gemini CLI config command with model selection

### Bug Fixes
- Merge tool_calls from all Copilot response choices (fixes multi-choice handling)
- Use multiarch builder for multi-platform Docker builds
- Rename gemini-cli to gemini, strip provider prefix from model
- Ruff format fixes for Gemini route files

### Documentation & Tests
- Expand CLAUDE.md with full project structure and key concepts
- Update README with Gemini support
- Add auth and tool parameter tests

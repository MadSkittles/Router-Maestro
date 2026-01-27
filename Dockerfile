# Multi-stage build for minimal final image
# Stage 1: Builder with compilation tools
FROM python:3.11-alpine AS builder

WORKDIR /app

# Install build dependencies for native extensions
# - gcc, musl-dev: C compiler for native extensions
# - libffi-dev: Required by cffi/cryptography
# - cargo, rust: Required by tiktoken
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    cargo \
    rust

# Install uv for fast package installation
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .

# Stage 2: Minimal runtime image
FROM python:3.11-alpine

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apk add --no-cache \
    libffi \
    curl

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN adduser -D -u 1000 maestro
USER maestro

# Create data and logs directories
RUN mkdir -p /home/maestro/.local/share/router-maestro/logs \
    && mkdir -p /home/maestro/.config/router-maestro

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["router-maestro", "server", "start", "--host", "0.0.0.0", "--port", "8080"]

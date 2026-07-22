# Multi-stage build for minimal final image
# Stage 1: Builder with compilation tools
FROM python:3.14-alpine AS builder

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
# Strip build-time-only tooling not needed at runtime (~26MB): pip, setuptools,
# pkg_resources. Verified no runtime dependency imports them. Also remove the
# now-dangling pip console scripts in the venv bin (their package is gone, so
# they would crash if invoked). The base image's own /usr/local pip is left
# alone — deleting from a base layer only adds a whiteout and reclaims nothing.
RUN rm -rf /opt/venv/lib/python*/site-packages/pip \
           /opt/venv/lib/python*/site-packages/pip-*.dist-info \
           /opt/venv/lib/python*/site-packages/setuptools \
           /opt/venv/lib/python*/site-packages/setuptools-*.dist-info \
           /opt/venv/lib/python*/site-packages/pkg_resources \
           /opt/venv/lib/python*/site-packages/_distutils_hack \
           /opt/venv/bin/pip /opt/venv/bin/pip3 /opt/venv/bin/pip3.* \
    && find /opt/venv -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# Stage 2: Minimal runtime image
FROM python:3.14-alpine

WORKDIR /app

# Install runtime dependencies only (no build tools).
# curl removed: healthcheck uses Python stdlib urllib instead (~4MB saved).
RUN apk add --no-cache libffi

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
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=5).status==200 else 1)"]

# Run the server
CMD ["router-maestro", "server", "start", "--host", "0.0.0.0", "--port", "8080"]

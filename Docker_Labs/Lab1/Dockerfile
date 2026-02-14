# Stage 1: Builder
FROM python:3.10 AS builder

# Set the working directory in the container
WORKDIR /app

# Copy the model training script into the container
COPY src/ .

# Copy uv project files
COPY pyproject.toml uv.lock ./

# Install uv package manager using curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Sync dependencies using uv
RUN uv sync --all-groups --no-dev

# Set up PATH for the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run tests
RUN python -m pytest test.py -v

# Stage 2: Runtime
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the trained model and dependencies from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/main.py .

# Set up PATH for the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run the script when the container launches
CMD ["python", "main.py"]

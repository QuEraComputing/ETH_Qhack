# Start with a Python base image
FROM python:3.12-slim

# Install Java (JRE)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get install -y uv && \
    apt-get clean;

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Make the virtual environment active by adding its binary path to the top of $PATH
# All subsequent Python commands (like gunicorn) will use this environment
# ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of your app's code
COPY . .
RUN uv sync --no-cache --active

# Expose the port Render uses (default is 10000 or 8000 depending on your framework)
EXPOSE 10000

# Replace with your actual start command (e.g., for FastAPI, Flask, etc.)
CMD ["py server.py",]
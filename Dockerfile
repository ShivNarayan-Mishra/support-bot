FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# llama-cpp-python compiles from C++ source if no matching pre-built wheel exists -
# slim doesn't ship a compiler by default, so this is needed for a reliable build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt
COPY main.py db.py ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
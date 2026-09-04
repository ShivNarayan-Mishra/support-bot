# slim is very lightweight with just the amount of debian required to run python
FROM python:3.11-slim

# uv was used due to its efficiency. This pulls the pre compiled binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# specify the working directory inside the container
WORKDIR /app

# since docker caches layers the copies are staggered
COPY requirements.txt .

# Install dependencies using the pre-built CPU wheel for llama-cpp-python
RUN uv pip install --system --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY main.py db.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#majority of choices are made to accomodate local memory constraints
#slim is very lightweight with just the amount of debian required to run python
FROM python:3.11-slim
#uv was used due to its efficiency. This pulls the pre compiled binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
#specify the working directory inside the container
WORKDIR /app
#pytorch installed first to satisfy the requirement compatibilty 
#system wide installations due to separated virtual enviroments
#cpu only versions are used due to system constrains and the wheel reduces the costs drastically
RUN uv pip install --system  --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
#since docker caches layers the copies are staggered
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt
COPY main.py db.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# uvicorn: the ASGI server running our FastAPI app
# --host 0.0.0.0: listen on all interfaces inside the container
#   (not just localhost, which would be unreachable from outside)
# --port 8000: the port Docker will map to the host
# --reload: auto-restart on code changes (dev only — remove in production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
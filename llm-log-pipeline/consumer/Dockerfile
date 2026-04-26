FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for psycopg2 compilation if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# --no-install-recommends: skip optional packages (saves ~200MB)
# rm -rf /var/lib/apt/lists/*: delete apt cache after install (smaller image)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
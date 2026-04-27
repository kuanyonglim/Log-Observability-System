FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# --server.address=0.0.0.0: listen on all interfaces (not just localhost)
# --server.port=8501: Streamlit's default port
# --server.headless=true: disable the "open browser" prompt in Docker
# --server.fileWatcherType=none: disable file watcher (not needed in container)
CMD [ "streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none" ]


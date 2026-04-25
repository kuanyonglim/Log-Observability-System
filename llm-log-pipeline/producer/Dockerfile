# Use slim variant — full Python image is ~900MB, slim is ~130MB.
# We don't need compilers or dev headers for this service.
FROM python:3.11-slim

# Set working directory inside the container.
# All subsequent commands run relative to this path.
WORKDIR /app

# Copy requirements FIRST — before copying source code.
# Why? Docker caches each layer. If requirements.txt hasn't changed,
# Docker skips the pip install step entirely on rebuild. This saves
# 30-60 seconds per rebuild during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: don't store pip's download cache inside the image.
# Keeps the image smaller — we don't need to re-download in production.

# Now copy the source code (changes more often than requirements)
COPY . .

# The command that runs when the container starts
CMD ["python", "main.py"]
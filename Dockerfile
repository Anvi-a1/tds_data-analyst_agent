FROM python:3.11-slim

WORKDIR /app

# Install uv for fast Python package management
RUN pip install uv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps

# Copy source
COPY . /app

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

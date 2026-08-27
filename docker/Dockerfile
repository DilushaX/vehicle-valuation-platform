FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose FastAPI port (8000) and Streamlit port (8501)
EXPOSE 8000 8501

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

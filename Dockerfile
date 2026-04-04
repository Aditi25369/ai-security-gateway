FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY gateway.py .
COPY dashboard.py .
COPY evaluation.py .

# Create directories
RUN mkdir -p logs models results

# Expose ports
EXPOSE 8080 8501

# Default command (can be overridden)
CMD ["python", "gateway.py"]

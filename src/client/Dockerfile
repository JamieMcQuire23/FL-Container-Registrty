FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port for the API
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn on ARM architecture
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
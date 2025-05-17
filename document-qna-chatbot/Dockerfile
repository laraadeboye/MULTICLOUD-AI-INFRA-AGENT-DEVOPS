FROM python:3.10-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]

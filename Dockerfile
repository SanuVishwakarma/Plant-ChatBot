FROM python:3.9-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Create a script to run the API
RUN echo '#!/bin/bash\nuvicorn app:app --host 0.0.0.0 --port 7860' > /code/run.sh
RUN chmod +x /code/run.sh

# Command to run the application
CMD ["/code/run.sh"]
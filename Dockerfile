FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Copy the app directory
COPY app/ /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the transformer models
RUN python download_models.py

# Set the entrypoint to run main_challenge1b.py with collection argument
ENTRYPOINT ["python", "main_challenge1b.py", "--collection"]

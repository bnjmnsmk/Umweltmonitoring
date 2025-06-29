FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set timezone to UTC explicitly
ENV TZ=UTC

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Dash's default port
EXPOSE 8050

# Run the app
CMD ["python", "app.py"]

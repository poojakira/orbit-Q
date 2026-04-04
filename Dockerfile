FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from current folder into the container
COPY . .

# Install the orbit-q package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8501

# Start the dashboard (default; override in docker-compose per service)
CMD ["orbit-q", "dashboard"]

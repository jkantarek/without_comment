# Use the official Playwright image which has all browser dependencies pre-installed
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only requirements first for caching
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the rest of the app
COPY . .

# Install Playwright Firefox specifically
RUN playwright install firefox

# Create a data directory for the SQLite database
RUN mkdir -p /data && chmod 777 /data
ENV DB_PATH=/data/cache.db
ENV ADMIN_USER=""
ENV ADMIN_PASS=""
ENV FEED_USER=""
ENV FEED_PASS=""

# Expose port
EXPOSE 8000

# Run the application

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

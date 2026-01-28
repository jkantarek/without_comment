#!/bin/bash
set -e

# Requirement check for Auth
if [ -z "$ADMIN_USER" ] || [ -z "$ADMIN_PASS" ]; then
    echo "ERROR: ADMIN_USER and ADMIN_PASS environment variables must be set."
    echo "Example: ADMIN_USER=admin ADMIN_PASS=password123 FEED_USER=reader FEED_PASS=secret ./run.sh"
    echo "(FEED_USER and FEED_PASS are optional; if set, the /rss feed will be protected)"
    exit 1
fi

# Check if uv is installed
if command -v uv >/dev/null 2>&1; then
    echo "Using uv to manage the app..."
    uv run playwright install firefox
    echo "Starting app with uv..."
    uv run main.py
else
    echo "uv not found, falling back to standard venv..."
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Installing Playwright Firefox..."
    python3 -m playwright install firefox
    echo "Starting app..."
    python main.py
fi

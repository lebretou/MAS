#!/bin/bash

# Start script for Data Analysis Multi-Agent System

echo "=========================================="
echo "Data Analysis Multi-Agent System"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "Please create a .env file with your API keys."
    echo "See .env.example for required variables."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ℹ️  No virtual environment detected."
    echo "Consider activating one for isolated dependencies."
    echo ""
fi

# Install dependencies if needed
echo "Checking dependencies..."
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting server..."
echo ""
echo "🚀 Server will be available at:"
echo "   → http://localhost:8000"
echo "   → Web UI: http://localhost:8000/app/"
echo ""
echo "📊 Telemetry Dashboards:"
echo "   → LangSmith: https://smith.langchain.com/"
echo "   → Langfuse: https://cloud.langfuse.com/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Start the server
python -m backend.main

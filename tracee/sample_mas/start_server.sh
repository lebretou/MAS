#!/bin/bash

# start script for Data Analysis Multi-Agent System

echo "=========================================="
echo "Data Analysis Multi-Agent System"
echo "=========================================="
echo ""

# check if .env file exists
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

# check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ℹ️  No virtual environment detected."
    echo "Consider activating one for isolated dependencies."
    echo ""
fi

# install dependencies if needed
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
echo "📊 Telemetry:"
echo "   → LangSmith: https://smith.langchain.com/"
echo "   → MAS backbone traces: outputs/traces/<trace_id>/trace_events.jsonl"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# start the server
python -m backend.main

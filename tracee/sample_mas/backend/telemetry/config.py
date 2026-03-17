"""Telemetry helpers for LangSmith and tracee."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
workspace_root = project_root.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

load_dotenv()

TRACEE_SERVER_URL = os.getenv("TRACE_API_URL", "http://localhost:8000")


def get_langsmith_config():
    """configure LangSmith tracing from dot env file (you should have this locally)."""
    return {
        "project_name": os.getenv("LANGSMITH_PROJECT", "data-analysis-agents"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
    }


def setup_langsmith():
    """setup LangSmith environment variables if configured."""
    config = get_langsmith_config()
    
    if config["tracing_enabled"] and config["api_key"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config["project_name"]
        os.environ["LANGCHAIN_API_KEY"] = config["api_key"]
        print(f"✓ LangSmith tracing enabled for project: {config['project_name']}")
    else:
        print("⚠ LangSmith tracing disabled - API key not found")


def setup_telemetry():
    """setup telemetry for application startup (LangSmith only, tracing done per-run)."""
    setup_langsmith()

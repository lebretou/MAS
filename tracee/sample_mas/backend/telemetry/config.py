"""Telemetry helpers for LangSmith and MAS backbone."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv

# add backbone to path (sibling directory inside tracee/)
backbone_path = Path(__file__).parent.parent.parent.parent / "backbone"
if str(backbone_path.parent) not in sys.path:
    sys.path.insert(0, str(backbone_path.parent))

from backbone.sdk.tracing import enable_tracing, TracingContext

# load environment variables
load_dotenv()

# default output directory for traces
TRACES_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "server" / "data" / "traces"


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


@contextmanager
def tracing_session(session_id: str = "default") -> Generator[TracingContext, None, None]:
    """context manager for a tracing session using the new enable_tracing API.
    
    Usage:
        with tracing_session("my-session") as ctx:
            result = app.invoke(state, config={"callbacks": ctx.callbacks})
    """
    with enable_tracing(output_dir=TRACES_OUTPUT_DIR) as ctx:
        print(f"✓ MAS Backbone tracing enabled")
        print(f"  trace id: {ctx.trace_id}")
        print(f"  session: {session_id}")
        yield ctx


def setup_telemetry():
    """setup telemetry for application startup (LangSmith only, tracing done per-run)."""
    setup_langsmith()

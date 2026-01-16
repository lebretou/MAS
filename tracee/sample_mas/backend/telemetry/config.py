"""Telemetry helpers for LangSmith and MAS backbone."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# add backbone to path (sibling directory inside tracee/)
backbone_path = Path(__file__).parent.parent.parent.parent / "backbone"
if str(backbone_path.parent) not in sys.path:
    sys.path.insert(0, str(backbone_path.parent))

from backbone import Tracer
from backbone.adapters import EventEmitter

# load environment variables
load_dotenv()

# module-level storage for current tracer
_current_tracer: Tracer | None = None


def get_langsmith_config():
    """configure LangSmith tracing from dot env file (you shold have this locally)."""
    return {
        "project_name": os.getenv("LANGSMITH_PROJECT", "data-analysis-agents"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
    }


def get_mas_backbone_handler():
    """create MAS backbone handler and emitter for semantic tracing."""
    global _current_tracer
    
    # Write traces to the centralized server data directory
    output_dir = Path(__file__).parent.parent.parent.parent / "server" / "data" / "traces"
    _current_tracer = Tracer(output_dir=output_dir)
    
    print(f"✓ MAS Backbone tracing enabled")
    print(f"  trace id: {_current_tracer.trace_id}")
    print(f"  output: {_current_tracer.output_path}")
    
    return _current_tracer.callback, _current_tracer.emitter


def get_emitter() -> EventEmitter | None:
    """get the current EventEmitter for manual events."""
    return _current_tracer.emitter if _current_tracer else None


def get_trace_info() -> dict:
    """get current execution and trace ids."""
    if _current_tracer:
        return {
            "execution_id": _current_tracer.execution_id,
            "trace_id": _current_tracer.trace_id,
        }
    return {"execution_id": None, "trace_id": None}


def setup_telemetry():
    """setup telemetry for application startup."""
    config = get_langsmith_config()
    callbacks = []
    
    if config["tracing_enabled"] and config["api_key"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config["project_name"]
        os.environ["LANGCHAIN_API_KEY"] = config["api_key"]
        print(f"✓ LangSmith tracing enabled for project: {config['project_name']}")
    else:
        print("⚠ LangSmith tracing disabled - API key not found")
    
    mas_callback, emitter = get_mas_backbone_handler()
    callbacks.append(mas_callback)
    
    return {
        "langsmith": config,
        "callbacks": callbacks,
        "emitter": emitter,
    }


def get_callbacks(session_id: str = "default"):
    """return configured callback handlers (LangSmith + MAS backbone)."""
    return setup_telemetry()["callbacks"]

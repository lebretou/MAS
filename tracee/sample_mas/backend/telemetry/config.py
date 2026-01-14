"""Telemetry helpers for LangSmith and MAS backbone (no Langfuse)."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# add backbone to path (sibling directory inside tracee/)
backbone_path = Path(__file__).parent.parent.parent.parent / "backbone"
if str(backbone_path.parent) not in sys.path:
    sys.path.insert(0, str(backbone_path.parent))

from backbone.adapters import MASCallbackHandler, FileSink, EventEmitter
from backbone.utils import generate_execution_id, generate_trace_id

# load environment variables
load_dotenv()

# module-level storage for current execution context
_current_emitter: EventEmitter | None = None
_current_execution_id: str | None = None
_current_trace_id: str | None = None
_current_sink: FileSink | None = None


def get_langsmith_config():
    """configure LangSmith tracing."""
    return {
        "project_name": os.getenv("LANGSMITH_PROJECT", "data-analysis-agents"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
    }


def get_mas_backbone_handler():
    """create MAS backbone handler and emitter for semantic tracing."""
    global _current_emitter, _current_execution_id, _current_trace_id, _current_sink
    
    _current_execution_id = generate_execution_id()
    _current_trace_id = generate_trace_id()
    
    output_dir = Path(__file__).parent.parent / "outputs" / "traces" / _current_trace_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _current_sink = FileSink(output_dir / "trace_events.jsonl")
    
    callback = MASCallbackHandler(
        execution_id=_current_execution_id,
        trace_id=_current_trace_id,
        event_sink=_current_sink,
        default_agent_id="workflow",
    )
    
    _current_emitter = EventEmitter(_current_execution_id, _current_trace_id, _current_sink)
    
    print(f"✓ MAS Backbone tracing enabled")
    print(f"  trace id: {_current_trace_id}")
    print(f"  output: {output_dir}")
    
    return callback, _current_emitter


def get_emitter() -> EventEmitter | None:
    """get the current EventEmitter for manual events."""
    return _current_emitter


def get_trace_info() -> dict:
    """get current execution and trace ids."""
    return {
        "execution_id": _current_execution_id,
        "trace_id": _current_trace_id,
    }


def get_callbacks(session_id: str = "default"):
    """return configured callback handlers (LangSmith + MAS backbone)."""
    callbacks = []
    
    config = get_langsmith_config()
    if config["tracing_enabled"] and config["api_key"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config["project_name"]
        os.environ["LANGCHAIN_API_KEY"] = config["api_key"]
        print(f"✓ LangSmith tracing enabled for project: {config['project_name']}")
    else:
        print("⚠ LangSmith tracing disabled - API key not found")
    
    mas_callback, _ = get_mas_backbone_handler()
    callbacks.append(mas_callback)
    
    return callbacks


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

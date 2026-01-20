"""Adapters for integrating with LangChain/LangGraph and event capture."""

from backbone.adapters.langchain_callback import (
    EventSink,
    ListSink,
    RawCallbackHandler,
    MASCallbackHandler,  # alias for backwards compatibility
)
from backbone.adapters.event_api import (
    FileSink,
    EventEmitter,
)

__all__ = [
    "EventSink",
    "ListSink",
    "FileSink",
    "RawCallbackHandler",
    "MASCallbackHandler",
    "EventEmitter",
]

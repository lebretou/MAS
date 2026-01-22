"""Adapters for integrating with LangChain/LangGraph and event capture."""

from backbone.adapters.event_api import EventEmitter
from backbone.adapters.langchain_callback import (
    RawCallbackHandler,
    MASCallbackHandler,  # alias for backwards compatibility
)
from backbone.adapters.sinks import EventSink, FileSink, HttpSink, ListSink

__all__ = [
    "EventSink",
    "ListSink",
    "FileSink",
    "HttpSink",
    "RawCallbackHandler",
    "MASCallbackHandler",
    "EventEmitter",
]

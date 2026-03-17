"""Tests for the public tracee instrumentation API."""

import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


class DummyGraph:
    def __init__(self):
        self.invoke_calls = []
        self.ainvoke_calls = []

    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls.append({
            "input": input,
            "config": config,
            "kwargs": kwargs,
        })
        return config

    async def ainvoke(self, input, config=None, **kwargs):
        self.ainvoke_calls.append({
            "input": input,
            "config": config,
            "kwargs": kwargs,
        })
        return config


class TestTraceePublicApi:
    def test_tracee_exports_init_and_trace(self):
        import tracee

        assert callable(tracee.init)
        assert callable(tracee.trace)

    def test_tracee_imports_from_workspace_parent(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tracee; assert callable(tracee.init); assert callable(tracee.trace)",
            ],
            cwd=repo_root.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr


class TestInit:
    def test_init_registers_graph_and_patches_invoke(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        graph = DummyGraph()
        register = MagicMock()
        monkeypatch.setattr(instrument, "extract_and_register", register)

        tracee.init(
            graph,
            graph_id="graph-1",
            name="Graph One",
            description="desc",
            server_url="http://localhost:9000",
        )

        register.assert_called_once_with(
            graph,
            "graph-1",
            "Graph One",
            "desc",
            base_url="http://localhost:9000",
        )

        result = graph.invoke({"hello": "world"})
        assert result is None

    def test_init_without_graph_id_only_patches_invoke(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        graph = DummyGraph()
        register = MagicMock()
        monkeypatch.setattr(instrument, "extract_and_register", register)

        tracee.init(graph, server_url="http://localhost:9000")

        register.assert_not_called()
        assert callable(graph.invoke)

    def test_patched_invoke_merges_callbacks(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        graph = DummyGraph()
        ctx = SimpleNamespace(callbacks=["tracee-callback"])
        monkeypatch.setattr(instrument, "get_active_context", lambda: ctx)

        tracee.init(graph)

        result = graph.invoke(
            {"value": 1},
            config={"callbacks": ["existing"], "tags": ["demo"]},
            extra=True,
        )

        assert result == {
            "callbacks": ["existing", "tracee-callback"],
            "tags": ["demo"],
        }
        assert graph.invoke_calls[0]["kwargs"] == {"extra": True}

    def test_patched_ainvoke_merges_callbacks(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        graph = DummyGraph()
        ctx = SimpleNamespace(callbacks=["tracee-callback"])
        monkeypatch.setattr(instrument, "get_active_context", lambda: ctx)

        tracee.init(graph)

        result = asyncio.run(graph.ainvoke({"value": 1}, config={"callbacks": ["existing"]}))

        assert result == {"callbacks": ["existing", "tracee-callback"]}

    def test_patched_invoke_dedupes_existing_trace_callback(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        graph = DummyGraph()
        callback = object()
        ctx = SimpleNamespace(callbacks=[callback])
        monkeypatch.setattr(instrument, "get_active_context", lambda: ctx)

        tracee.init(graph)

        result = graph.invoke({"value": 1}, config={"callbacks": [callback]})

        assert result == {"callbacks": [callback]}


class TestTrace:
    def test_trace_uses_server_url_from_init(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        enable = MagicMock(return_value="ctx")
        monkeypatch.setattr(instrument, "enable_tracing", enable)

        tracee.init(DummyGraph(), server_url="http://localhost:8123")

        assert tracee.trace() == "ctx"
        enable.assert_called_once_with(base_url="http://localhost:8123")

    def test_trace_allows_explicit_override(self, monkeypatch):
        import tracee
        import backbone.sdk.instrument as instrument

        enable = MagicMock(return_value="ctx")
        monkeypatch.setattr(instrument, "enable_tracing", enable)

        assert tracee.trace(base_url="http://localhost:9999") == "ctx"
        enable.assert_called_once_with(base_url="http://localhost:9999")

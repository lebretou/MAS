"""Tests for the sample MAS tracee integration."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pandas as pd

sample_root = Path(__file__).resolve().parents[2] / "sample_mas"
if str(sample_root) not in sys.path:
    sys.path.insert(0, str(sample_root))

from backend.graph import workflow
from backend.agents import interaction, planner, coding, summary
from backend.telemetry.config import TRACEE_SERVER_URL


class DummyApp:
    def __init__(self):
        self.invoke_calls = []
        self.ainvoke_calls = []

    def invoke(self, state, config=None):
        self.invoke_calls.append({"state": state, "config": config})
        return {
            **state,
            "final_summary": "done",
            "relevance_decision": "chat_only",
            "generated_code": "",
            "execution_result": {"success": True},
            "analysis_plan": "",
            "rag_context": "",
            "retry_count": 0,
            "messages": state["messages"],
        }

    async def ainvoke(self, state, config=None):
        self.ainvoke_calls.append({"state": state, "config": config})
        return {
            **state,
            "final_summary": "done",
            "relevance_decision": "chat_only",
            "generated_code": "",
            "execution_result": {"success": True},
            "analysis_plan": "",
            "rag_context": "",
            "retry_count": 0,
            "messages": state["messages"],
        }


def _dataset():
    return pd.DataFrame({
        "age": [21, 35],
        "score": [0.1, 0.2],
    })


class TestCreateWorkflow:
    def test_create_workflow_registers_with_tracee_init(self, monkeypatch):
        init = MagicMock()
        monkeypatch.setattr(workflow.tracee, "init", init)

        app = workflow.create_workflow()

        init.assert_called_once_with(
            app,
            graph_id="data-analysis-mas",
            name="Data Analysis MAS",
            description="Multi-agent system for interactive data analysis",
            server_url="http://localhost:8000",
        )

    def test_agents_share_tracee_server_url_for_prompt_loading(self):
        assert interaction.loader.base_url == TRACEE_SERVER_URL
        assert planner.loader.base_url == TRACEE_SERVER_URL
        assert coding.loader.base_url == TRACEE_SERVER_URL
        assert summary.loader.base_url == TRACEE_SERVER_URL


class TestRunAnalysisWorkflow:
    def test_run_analysis_workflow_uses_trace_context_and_no_callback_threading(self, monkeypatch):
        app = DummyApp()
        state = {"workflow_created": False}

        @contextmanager
        def fake_trace():
            assert state["workflow_created"] is True
            yield "ctx"

        def fake_create_workflow():
            state["workflow_created"] = True
            return app

        monkeypatch.setattr(workflow, "create_workflow", fake_create_workflow)
        monkeypatch.setattr(workflow.tracee, "trace", fake_trace)

        result = workflow.run_analysis_workflow(_dataset(), "summarize the dataset")

        assert result["success"] is True
        assert app.invoke_calls[0]["config"] is None
        assert "callbacks" not in app.invoke_calls[0]["state"]

    def test_run_analysis_workflow_async_uses_trace_context_and_no_callback_threading(self, monkeypatch):
        app = DummyApp()
        state = {"workflow_created": False}

        @contextmanager
        def fake_trace():
            assert state["workflow_created"] is True
            yield "ctx"

        def fake_create_workflow():
            state["workflow_created"] = True
            return app

        monkeypatch.setattr(workflow, "create_workflow", fake_create_workflow)
        monkeypatch.setattr(workflow.tracee, "trace", fake_trace)

        result = asyncio.run(workflow.run_analysis_workflow_async(_dataset(), "summarize the dataset"))

        assert result["success"] is True
        assert app.ainvoke_calls[0]["config"] is None
        assert "callbacks" not in app.ainvoke_calls[0]["state"]

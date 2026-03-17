"""Tests for agent registry route merge behavior."""

from backbone.models.agent_registry import AgentRegistryEntry
from backbone.models.graph_topology import GraphNode


class TestUpsertAgent:
    def test_upsert_agent_preserves_existing_has_tools_when_not_provided(self, monkeypatch):
        from server import agent_routes

        existing = AgentRegistryEntry(
            agent_id="planner",
            prompt_id="old-prompt",
            prompt_version_id="v1",
            model="gpt-4o-mini",
            temperature=0.2,
            has_tools=True,
            metadata={"source": "graph"},
            updated_at="earlier",
        )
        saved = []

        monkeypatch.setattr(agent_routes, "db_get_agent", lambda agent_id: existing)
        monkeypatch.setattr(agent_routes, "db_upsert_agent", lambda entry: saved.append(entry))
        monkeypatch.setattr(agent_routes, "utc_timestamp", lambda: "now")

        entry = agent_routes.upsert_agent(
            "planner",
            agent_routes.UpsertAgentRequest(
                prompt_id="new-prompt",
                prompt_version_id="v2",
            ),
        )

        assert entry.prompt_id == "new-prompt"
        assert entry.prompt_version_id == "v2"
        assert entry.has_tools is True
        assert saved[0].has_tools is True


class TestUpsertGraph:
    def test_upsert_graph_preserves_existing_prompt_version(self, monkeypatch):
        from server import graph_routes

        existing_agent = AgentRegistryEntry(
            agent_id="planner",
            prompt_id="planner-prompt",
            prompt_version_id="v7",
            model="gpt-4o-mini",
            temperature=0.2,
            has_tools=True,
            metadata={"source": "runtime"},
            updated_at="earlier",
        )
        saved = []

        monkeypatch.setattr(graph_routes, "db_get_graph", lambda graph_id: None)
        monkeypatch.setattr(graph_routes, "db_upsert_graph", lambda graph: None)
        monkeypatch.setattr(graph_routes, "utc_timestamp", lambda: "now")

        import server.agent_db as agent_db

        monkeypatch.setattr(agent_db, "get_agent", lambda agent_id: existing_agent)
        monkeypatch.setattr(agent_db, "upsert_agent", lambda entry: saved.append(entry))

        request = graph_routes.UpsertGraphRequest(
            graph_id="graph-1",
            name="Graph One",
            nodes=[
                GraphNode(
                    node_id="planner",
                    label="planner",
                    node_type="agent",
                    prompt_id=None,
                    metadata={},
                )
            ],
            edges=[],
            state_schema=None,
        )

        graph_routes.upsert_graph("graph-1", request)

        assert saved[0].prompt_id == "planner-prompt"
        assert saved[0].prompt_version_id == "v7"
        assert saved[0].has_tools is True

    def test_upsert_graph_resets_prompt_version_when_prompt_changes(self, monkeypatch):
        from server import graph_routes

        existing_agent = AgentRegistryEntry(
            agent_id="planner",
            prompt_id="old-prompt",
            prompt_version_id="v7",
            model="gpt-4o-mini",
            temperature=0.2,
            has_tools=True,
            metadata={"source": "runtime"},
            updated_at="earlier",
        )
        saved = []

        monkeypatch.setattr(graph_routes, "db_get_graph", lambda graph_id: None)
        monkeypatch.setattr(graph_routes, "db_upsert_graph", lambda graph: None)
        monkeypatch.setattr(graph_routes, "utc_timestamp", lambda: "now")

        import server.agent_db as agent_db

        monkeypatch.setattr(agent_db, "get_agent", lambda agent_id: existing_agent)
        monkeypatch.setattr(agent_db, "upsert_agent", lambda entry: saved.append(entry))

        request = graph_routes.UpsertGraphRequest(
            graph_id="graph-1",
            name="Graph One",
            nodes=[
                GraphNode(
                    node_id="planner",
                    label="planner",
                    node_type="agent",
                    prompt_id="new-prompt",
                    metadata={},
                )
            ],
            edges=[],
            state_schema=None,
        )

        graph_routes.upsert_graph("graph-1", request)

        assert saved[0].prompt_id == "new-prompt"
        assert saved[0].prompt_version_id is None

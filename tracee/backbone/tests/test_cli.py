"""Tests for the tracee CLI and packaging config."""

from pathlib import Path
import sys
from types import SimpleNamespace
import tomllib

from fastapi.testclient import TestClient
import pytest
import cli
from server.app import app


class TestCli:
    def test_serve_subcommand_runs_uvicorn(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))))
        monkeypatch.setattr(sys, "argv", ["tracee", "serve", "--host", "127.0.0.1", "--port", "9001"])

        cli.main()

        assert calls == [
            (
                ("server.app:app",),
                {"host": "127.0.0.1", "port": 9001},
            )
        ]

    def test_serve_subcommand_requires_server_extras(self, monkeypatch):
        monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None if name == "uvicorn" else object())
        monkeypatch.setattr(sys, "argv", ["tracee", "serve"])

        with pytest.raises(SystemExit) as exc:
            cli.main()

        assert exc.value.code == 2


class TestServerApp:
    def test_server_serves_spa_deep_links(self):
        client = TestClient(app)
        response = client.get("/playground")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_server_blocks_ui_path_traversal(self):
        client = TestClient(app)
        response = client.get("/%2e%2e/%2e%2e/pyproject.toml")

        assert response.status_code == 404

    def test_server_preserves_api_404s(self):
        client = TestClient(app)
        response = client.get("/api/not-a-route")

        assert response.status_code == 404
        assert "application/json" in response.headers["content-type"]


class TestPackaging:
    def test_pyproject_declares_tracee_script_and_dependency_split(self):
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text())

        dependencies = data["project"]["dependencies"]
        optional = data["project"]["optional-dependencies"]
        scripts = data["project"]["scripts"]

        assert scripts["tracee"] == "tracee.cli:main"
        assert "httpx>=0.28" in dependencies
        assert "fastapi>=0.115" not in dependencies
        assert "server" in optional
        assert "dev" in optional
        assert "fastapi>=0.115" in optional["server"]
        assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]["playground-ui/dist"] == "playground-ui/dist"

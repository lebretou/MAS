"""Public tracee instrumentation helpers."""

from __future__ import annotations

from functools import wraps

from backbone.sdk.graph_extractor import extract_and_register
from backbone.sdk.tracing import enable_tracing, get_active_context

_config: dict[str, str | None] = {"server_url": None, "graph_id": None}


def init(
    compiled_graph,
    graph_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    server_url: str = "http://localhost:8000",
):
    """patch a compiled graph and optionally register its topology."""
    _config["server_url"] = server_url
    _config["graph_id"] = graph_id

    if graph_id:
        extract_and_register(
            compiled_graph,
            graph_id,
            name or graph_id,
            description,
            base_url=server_url,
        )

    _patch_invoke(compiled_graph)
    return compiled_graph


def trace(base_url: str | None = None, graph_id: str | None = None):
    """start a traced run using the configured server url."""
    return enable_tracing(
        base_url=base_url or _config.get("server_url"),
        graph_id=graph_id or _config.get("graph_id"),
    )


def _patch_invoke(compiled_graph) -> None:
    """patch invoke methods to inject callbacks from the active trace."""
    original_invoke = getattr(compiled_graph, "invoke", None)
    if original_invoke and not getattr(original_invoke, "__tracee_patched__", False):
        @wraps(original_invoke)
        def patched_invoke(input, config=None, **kwargs):
            ctx = get_active_context()
            if ctx:
                config = _merge_tracee_callbacks(config, ctx.callbacks)
            return original_invoke(input, config=config, **kwargs)

        patched_invoke.__tracee_patched__ = True
        compiled_graph.invoke = patched_invoke

    original_ainvoke = getattr(compiled_graph, "ainvoke", None)
    if original_ainvoke and not getattr(original_ainvoke, "__tracee_patched__", False):
        @wraps(original_ainvoke)
        async def patched_ainvoke(input, config=None, **kwargs):
            ctx = get_active_context()
            if ctx:
                config = _merge_tracee_callbacks(config, ctx.callbacks)
            return await original_ainvoke(input, config=config, **kwargs)

        patched_ainvoke.__tracee_patched__ = True
        compiled_graph.ainvoke = patched_ainvoke


def _merge_tracee_callbacks(config, callbacks):
    """merge trace callbacks into an existing config dict."""
    existing_config = dict(config or {})
    existing_callbacks = _to_list(existing_config.get("callbacks"))
    merged_callbacks = list(existing_callbacks)
    for callback in _to_list(callbacks):
        if any(existing is callback for existing in merged_callbacks):
            continue
        merged_callbacks.append(callback)
    existing_config["callbacks"] = merged_callbacks
    return existing_config


def _to_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]

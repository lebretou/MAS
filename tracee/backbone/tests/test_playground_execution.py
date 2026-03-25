"""tests for playground execution message compilation"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backbone.models.playground_run import PlaygroundRunCreate
from backbone.models.prompt_artifact import (
    PromptComponent,
    PromptComponentType,
    PromptMessageRole,
    PromptTool,
    PromptToolArgument,
    PromptVersion,
)
from backbone.utils.identifiers import utc_timestamp


def _make_version(*, with_tools: bool = False) -> PromptVersion:
    tools = []
    if with_tools:
        tools = [
            PromptTool(
                name="lookup_dataset",
                description="look up dataset metadata",
                arguments=[
                    PromptToolArgument(
                        name="dataset_name",
                        description="dataset to inspect",
                        required=True,
                    )
                ],
            )
        ]

    return PromptVersion(
        prompt_id="test-prompt",
        version_id="v1",
        name="Test prompt",
        components=[
            PromptComponent(
                type=PromptComponentType.role,
                name="Role",
                message_role=PromptMessageRole.system,
                content="You are a careful analyst.",
            ),
            PromptComponent(
                type=PromptComponentType.task,
                name="Task",
                message_role=PromptMessageRole.human,
                content="Review {{question}} and decide what to do.",
            ),
            PromptComponent(
                type=PromptComponentType.examples,
                name="Example reply",
                message_role=PromptMessageRole.ai,
                content="I will inspect the available data first.",
            ),
            PromptComponent(
                type=PromptComponentType.constraints,
                name="Constraints",
                content="Do not guess missing facts.",
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
        },
        tools=tools,
        created_at=utc_timestamp(),
    )


class TestPlaygroundMessageCompilation:
    def test_build_playground_messages_uses_component_roles_and_variables(self):
        from server.playground_routes import _build_playground_messages

        version = _make_version()

        messages = _build_playground_messages(
            version,
            {"question": "the latest release brief"},
        )

        assert messages == [
            {
                "role": "system",
                "content": "Role:\nYou are a careful analyst.",
            },
            {
                "role": "user",
                "content": "Task:\nReview the latest release brief and decide what to do.",
            },
            {
                "role": "assistant",
                "content": "Example reply:\nI will inspect the available data first.",
            },
            {
                "role": "system",
                "content": "Constraints:\nDo not guess missing facts.",
            },
        ]


class TestPlaygroundExecution:
    def test_execute_playground_run_passes_schema_and_tools_to_openai(self):
        from server.playground_routes import execute_playground_run

        version = _make_version(with_tools=True)
        request = PlaygroundRunCreate(
            prompt_id="test-prompt",
            version_id="v1",
            provider="openai",
            model="gpt-4o",
            temperature=0,
            input_variables={"question": "the latest release brief"},
        )

        call_mock = AsyncMock(return_value={
            "content": "",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "lookup_dataset",
                    "arguments": {"dataset_name": "release_briefs"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            "schema_enforced": False,
        })

        with (
            patch("server.playground_routes._load_prompt_version", return_value=version),
            patch("server.playground_routes.call_llm_messages", call_mock),
            patch("server.playground_routes.insert_run", MagicMock()),
        ):
            response = asyncio.run(execute_playground_run(request))

        kwargs = call_mock.await_args.kwargs
        assert kwargs["messages"] == [
            {
                "role": "system",
                "content": "Role:\nYou are a careful analyst.",
            },
            {
                "role": "user",
                "content": "Task:\nReview the latest release brief and decide what to do.",
            },
            {
                "role": "assistant",
                "content": "Example reply:\nI will inspect the available data first.",
            },
            {
                "role": "system",
                "content": "Constraints:\nDo not guess missing facts.",
            },
        ]
        assert kwargs["output_schema"] == version.output_schema
        assert kwargs["prompt_tools"] == version.tools
        assert response.run.tool_calls[0].name == "lookup_dataset"

    def test_execute_playground_run_uses_inline_draft_content_without_loading_a_saved_version(self):
        from server.playground_routes import execute_playground_run

        request = PlaygroundRunCreate(
            prompt_id="test-prompt",
            version_id="draft-v1",
            provider="openai",
            model="gpt-4o",
            temperature=0,
            input_variables={"question": "the latest release brief"},
            components=[
                PromptComponent(
                    type=PromptComponentType.role,
                    name="Role",
                    message_role=PromptMessageRole.system,
                    content="You are a draft reviewer.",
                ),
                PromptComponent(
                    type=PromptComponentType.task,
                    name="Task",
                    message_role=PromptMessageRole.human,
                    content="Summarize {{question}} with draft logic.",
                ),
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
            },
            tools=[
                PromptTool(
                    name="lookup_dataset",
                    description="look up dataset metadata",
                    arguments=[
                        PromptToolArgument(
                            name="dataset_name",
                            description="dataset to inspect",
                            required=True,
                        )
                    ],
                )
            ],
        )

        call_mock = AsyncMock(return_value={
            "content": '{"summary":"done"}',
            "tool_calls": None,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            "schema_enforced": True,
        })

        with (
            patch("server.playground_routes._load_prompt_version") as load_version_mock,
            patch("server.playground_routes.call_llm_messages", call_mock),
            patch("server.playground_routes.insert_run", MagicMock()),
        ):
            response = asyncio.run(execute_playground_run(request))

        load_version_mock.assert_not_called()
        kwargs = call_mock.await_args.kwargs
        assert kwargs["messages"] == [
            {
                "role": "system",
                "content": "Role:\nYou are a draft reviewer.",
            },
            {
                "role": "user",
                "content": "Task:\nSummarize the latest release brief with draft logic.",
            },
        ]
        assert kwargs["output_schema"] == request.output_schema
        assert kwargs["prompt_tools"] == request.tools
        assert response.run.version_id == "draft-v1"

    def test_execute_playground_run_can_disable_schema_for_tool_runs(self):
        from server.playground_routes import execute_playground_run

        version = _make_version(with_tools=True)
        request = PlaygroundRunCreate(
            prompt_id="test-prompt",
            version_id="v1",
            provider="openai",
            model="gpt-4o",
            temperature=0,
            disable_output_schema=True,
            input_variables={"question": "the latest release brief"},
        )

        call_mock = AsyncMock(return_value={
            "content": "",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "lookup_dataset",
                    "arguments": {"dataset_name": "release_briefs"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            "schema_enforced": False,
        })

        with (
            patch("server.playground_routes._load_prompt_version", return_value=version),
            patch("server.playground_routes.call_llm_messages", call_mock),
            patch("server.playground_routes.insert_run", MagicMock()),
        ):
            response = asyncio.run(execute_playground_run(request))

        kwargs = call_mock.await_args.kwargs
        assert kwargs["output_schema"] is None
        assert kwargs["prompt_tools"] == version.tools
        assert response.run.output_schema is None

    def test_execute_playground_run_rejects_anthropic(self):
        from server.playground_routes import execute_playground_run

        version = _make_version()
        request = PlaygroundRunCreate(
            prompt_id="test-prompt",
            version_id="v1",
            provider="anthropic",
            model="claude-3-sonnet-20240229",
        )

        with patch("server.playground_routes._load_prompt_version", return_value=version):
            with pytest.raises(HTTPException, match="Only OpenAI is supported"):
                asyncio.run(execute_playground_run(request))

    def test_execute_playground_run_rejects_empty_message_payloads(self):
        from server.playground_routes import execute_playground_run

        version = PromptVersion(
            prompt_id="test-prompt",
            version_id="v1",
            name="Empty prompt",
            components=[
                PromptComponent(
                    type=PromptComponentType.role,
                    content="You are a careful analyst.",
                    enabled=False,
                )
            ],
            created_at=utc_timestamp(),
        )
        request = PlaygroundRunCreate(
            prompt_id="test-prompt",
            version_id="v1",
            provider="openai",
            model="gpt-4o",
        )

        with patch("server.playground_routes._load_prompt_version", return_value=version):
            with pytest.raises(HTTPException, match="at least one enabled prompt component"):
                asyncio.run(execute_playground_run(request))


class TestPlaygroundAnalysis:
    def test_project_embeddings_2d_returns_center_for_single_point(self):
        from server.playground_routes import _project_embeddings_2d

        points, average_similarity = _project_embeddings_2d([[0.2, 0.4, 0.8]])

        assert points == [(0.5, 0.5)]
        assert average_similarity == [1.0]

    def test_analyze_playground_outputs_returns_grouped_points(self):
        from server.playground_routes import (
            PlaygroundAnalysisRequest,
            analyze_playground_outputs,
        )

        request = PlaygroundAnalysisRequest(
            items=[
                {
                    "id": "a-1",
                    "group_id": "a",
                    "label": "run 1",
                    "output": "alpha",
                },
                {
                    "id": "b-1",
                    "group_id": "b",
                    "label": "run 2",
                    "output": "beta",
                },
            ]
        )

        with patch(
            "server.playground_routes.embed_openai_texts",
            AsyncMock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ):
            response = asyncio.run(analyze_playground_outputs(request))

        assert len(response.points) == 2
        assert {point.group_id for point in response.points} == {"a", "b"}
        assert all(0 <= point.x <= 1 for point in response.points)
        assert all(0 <= point.y <= 1 for point in response.points)

    def test_analyze_playground_outputs_rejects_unsupported_embedding_models(self):
        from server.playground_routes import (
            PlaygroundAnalysisRequest,
            analyze_playground_outputs,
        )

        request = PlaygroundAnalysisRequest(
            items=[
                {
                    "id": "a-1",
                    "group_id": "a",
                    "label": "run 1",
                    "output": "alpha",
                },
                {
                    "id": "b-1",
                    "group_id": "b",
                    "label": "run 2",
                    "output": "beta",
                },
            ],
            embedding_model="text-embedding-3-large",
        )

        with pytest.raises(HTTPException, match="Unsupported embedding model"):
            asyncio.run(analyze_playground_outputs(request))

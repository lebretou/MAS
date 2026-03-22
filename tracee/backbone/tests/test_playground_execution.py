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

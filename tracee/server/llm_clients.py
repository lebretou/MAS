"""shared llm client helpers for playground and guided start."""

import json
import logging
import os
from typing import TypedDict

from fastapi import HTTPException

from backbone.models.playground_run import PlaygroundToolCall
from backbone.models.prompt_artifact import PromptTool

logger = logging.getLogger(__name__)

_openai_client = None


class LlmMessage(TypedDict):
    role: str
    content: str


def get_openai_client():
    """Get or create the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set",
            )
        _openai_client = openai.AsyncOpenAI(api_key=api_key)
    return _openai_client


def build_openai_response_format(schema: dict, *, strict: bool = True) -> dict:
    """Build OpenAI json_schema response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("title", "output"),
            "schema": schema,
            "strict": strict,
        },
    }


def supports_openai_json_schema(model: str) -> bool:
    """Return whether the model supports json_schema output."""
    return model.startswith("gpt-4o") or model.startswith("gpt-4.1")


def build_openai_tool(tool: PromptTool) -> dict:
    """Build an OpenAI tool definition from an authored tool."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema(),
        },
    }

async def call_openai_messages(
    *,
    messages: list[LlmMessage],
    model: str,
    temperature: float,
    max_tokens: int | None,
    output_schema: dict | None = None,
    prompt_tools: list[PromptTool] | None = None,
    json_schema_strict: bool = True,
) -> dict:
    """Call OpenAI chat completions and normalize the response."""
    client = get_openai_client()

    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if prompt_tools:
            params["tools"] = [build_openai_tool(tool) for tool in prompt_tools]
        if output_schema:
            if not supports_openai_json_schema(model):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Structured output requires an OpenAI model with json_schema "
                        f"support. Unsupported model: {model}"
                    ),
                )
            params["response_format"] = build_openai_response_format(output_schema, strict=json_schema_strict)
        response = await client.chat.completions.create(**params)

        message = response.choices[0].message
        content = message.content or ""
        tool_calls: list[PlaygroundToolCall] = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                parsed_arguments = json.loads(arguments) if arguments else {}
                tool_calls.append(
                    PlaygroundToolCall(
                        call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=parsed_arguments,
                    )
                )
        if not content and tool_calls:
            content = json.dumps([call.model_dump() for call in tool_calls], indent=2)
        usage = response.usage

        return {
            "content": content,
            "tool_calls": tool_calls or None,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "schema_enforced": bool(output_schema) and not tool_calls,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("OpenAI API call failed (model=%s)", model)
        raise HTTPException(
            status_code=500,
            detail="OpenAI API error. See server logs for details.",
        )


async def call_llm_messages(
    *,
    messages: list[LlmMessage],
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int | None,
    output_schema: dict | None = None,
    prompt_tools: list[PromptTool] | None = None,
    json_schema_strict: bool = True,
) -> dict:
    """Call the provider-specific messages API."""
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return await call_openai_messages(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            output_schema=output_schema,
            prompt_tools=prompt_tools,
            json_schema_strict=json_schema_strict,
        )
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported provider: {provider}. Supported: openai",
    )

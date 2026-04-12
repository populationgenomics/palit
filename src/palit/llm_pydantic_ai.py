#!/usr/bin/env python3
"""Pydantic AI backend for LLM batch processing.

Supports AWS Bedrock (``bedrock/*`` model strings) and OpenAI-compatible APIs
(everything else, e.g. llama.cpp via ``OPENAI_BASE_URL``).
"""

import asyncio
import logging
import os
from typing import Any

import jsonschema
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.output import NativeOutput, StructuredDict, ToolOutput
from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider
from pydantic_ai.providers.openai import OpenAIProvider

from palit.llm import PromptResult

logger = logging.getLogger(__name__)


class PydanticAIProcessor:
    """LLM backend using Pydantic AI for Bedrock and OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        region: str | None = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model: Model

        if model.startswith("bedrock/"):
            model_id = model.removeprefix("bedrock/")
            self._model = BedrockConverseModel(
                model_id,
                provider=BedrockProvider(
                    region_name=region,
                    aws_read_timeout=1800,
                    aws_connect_timeout=60,
                ),
                profile=BedrockModelProfile(
                    bedrock_supports_tool_choice=False,
                    bedrock_send_back_thinking_parts=True,
                ),
            )
            self._is_bedrock = True
        else:
            # OpenAI-compatible (e.g. llama.cpp server).
            # Pass base_url explicitly so the provider auto-sets a placeholder
            # API key when OPENAI_API_KEY is not configured.
            base_url = os.environ.get("OPENAI_BASE_URL")
            self._model = OpenAIChatModel(
                model,
                provider=OpenAIProvider(base_url=base_url),
            )
            self._is_bedrock = False

    async def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]:
        if not prompts:
            return []

        tasks = [self._process_single(prompt, schema) for prompt in prompts]
        return list(await asyncio.gather(*tasks))

    async def _process_single(self, prompt: str, schema: dict[str, Any]) -> PromptResult | None:
        try:
            output_schema = StructuredDict(schema)

            if self._is_bedrock:
                output_type: Any = ToolOutput(output_schema)
                instructions: str | None = (
                    "Always return your response by calling the final_result tool."
                )
                model_settings: BedrockModelSettings | OpenAIChatModelSettings = (
                    BedrockModelSettings(
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        bedrock_additional_model_requests_fields={
                            "thinking": {"type": "adaptive"},
                            "output_config": {"effort": "medium"},
                        },
                    )
                )
            else:
                output_type = NativeOutput(output_schema, strict=True)
                instructions = None
                model_settings = OpenAIChatModelSettings(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            agent: Agent[None, dict[str, Any]] = Agent(
                self._model,
                output_type=output_type,
                retries=0,
                instructions=instructions,
                model_settings=model_settings,
            )

            result = await agent.run(prompt)
            usage = result.usage()
            logger.info(
                "Token usage: input=%d output=%d total=%d",
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
            )

            parsed_json: dict[str, Any] = result.output
            raw_response = result.all_messages_json().decode()

            jsonschema.validate(parsed_json, schema)

            return PromptResult(raw_response=raw_response, parsed_json=parsed_json)
        except Exception:
            logger.exception("Failed to process prompt")
            return None

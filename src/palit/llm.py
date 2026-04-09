#!/usr/bin/env python3
"""LLM utilities for batch processing with multiple backends.

Provides a common ``LLMProcessor`` protocol with implementations:

* **HarmonyProcessor** — vLLM offline batch inference (``openai/gpt-oss-*``)
* **PydanticAIProcessor** — Pydantic AI for Bedrock and OpenAI-compatible APIs

Use :func:`create_llm_processor` to obtain the right backend for a model string.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Result from processing a prompt."""

    raw_response: str
    parsed_json: dict[str, Any]


class LLMProcessor(Protocol):
    async def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]: ...


def create_llm_processor(
    model: str,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    reasoning_effort: str = "high",
) -> LLMProcessor:
    """Create an LLM processor for the given model string.

    Dispatch logic:
    - ``openai/gpt-oss-*`` → HarmonyProcessor (vLLM offline batch)
    - ``bedrock/*`` → PydanticAIProcessor (AWS Bedrock)
    - Everything else → PydanticAIProcessor (OpenAI-compatible, reads OPENAI_BASE_URL)
    """
    if model.startswith("openai/gpt-oss-"):
        # vLLM/Harmony are optional ML dependencies (Linux GPU only)
        from palit.llm_harmony import HarmonyProcessor

        return HarmonyProcessor(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            reasoning_effort=reasoning_effort,
        )

    from palit.llm_pydantic_ai import PydanticAIProcessor

    return PydanticAIProcessor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

#!/usr/bin/env python3
"""LLM utilities for batch processing with Harmony and vLLM."""

import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any

import jsonschema
from openai_harmony import HarmonyEncoding
from vllm import LLM, SamplingParams
from vllm.entrypoints import harmony_utils
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams

logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Result from processing a prompt."""

    raw_response: str
    parsed_json: dict[str, Any]


class HarmonyBatchProcessor:
    """Harmony-based batch processor using vLLM for structured outputs."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        reasoning_effort: str = "high",
    ):
        """Initialize Harmony batch processor."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.reasoning_effort = reasoning_effort
        self.llm: LLM | None = None
        self.encoding = harmony_utils.get_encoding()

    def initialize_model(self) -> None:
        """Initialize the vLLM model."""
        logger.info(f"Loading model {self.model} with vLLM...")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Max tokens: {self.max_tokens}")
        logger.info(f"  Tensor parallel size: {self.tensor_parallel_size}")

        self.llm = LLM(
            model=self.model,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def get_encoding(self) -> HarmonyEncoding:
        """Get the Harmony encoding for token counting."""
        return self.encoding

    def extract_final_json(
        self, output: RequestOutput, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract JSON from the final channel of Harmony-formatted output using vLLM utilities"""
        # Get the generated tokens
        generated_tokens = output.outputs[0].token_ids

        try:
            # Use vLLM's harmony utilities to parse the output
            _reasoning_content, final_content, _is_tool_call = harmony_utils.parse_chat_output(
                generated_tokens
            )

            # The JSON should be in the final_content
            if final_content:
                try:
                    parsed_json: dict[str, Any] = json.loads(final_content)
                    # Validate against schema
                    jsonschema.validate(parsed_json, schema)
                    return parsed_json
                except (json.JSONDecodeError, jsonschema.ValidationError):
                    # Invalid JSON or doesn't match schema
                    logger.debug(f"Invalid JSON in final content: {final_content[:100]}")
                    return None

            # No final content found
            logger.debug("No final content found, only reasoning")
            return None

        except Exception:
            # If Harmony parsing fails
            logger.exception("Harmony parsing failed")
            return None

    def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]:
        """
        Process a batch of prompts and return results in order.

        Args:
            prompts: List of prompt strings
            schema: JSON schema for structured output

        Returns:
            List of PromptResult objects or None for failed prompts, in same order as input
        """
        if not prompts:
            return []

        if self.llm is None:
            self.initialize_model()

        logger.info(f"Processing batch of {len(prompts)} prompts...")

        # Convert prompts to Harmony format token IDs
        token_prompts = []
        for prompt in prompts:
            # Create system message with configured reasoning effort and user message
            system_msg = harmony_utils.get_system_message(reasoning_effort=self.reasoning_effort)
            user_msg = harmony_utils.get_user_message(prompt)
            tokens = harmony_utils.render_for_completion([system_msg, user_msg])
            token_prompt = TokensPrompt(prompt_token_ids=tokens)
            token_prompts.append(token_prompt)

        # Create sampling params with structured output
        guided_decoding = GuidedDecodingParams(json=schema)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            guided_decoding=guided_decoding,
            stop_token_ids=harmony_utils.get_stop_tokens_for_assistant_actions(),
        )

        start_time = time.time()

        # Process all prompts in batch using token prompts
        assert self.llm is not None, "Model not initialized"
        outputs = self.llm.generate(prompts=token_prompts, sampling_params=sampling_params)

        end_time = time.time()
        batch_time = end_time - start_time

        # Process results in order
        results: list[PromptResult | None] = []
        failed_count = 0
        token_lengths: list[int] = []

        for i, output in enumerate(outputs):
            if output.outputs and len(output.outputs) > 0:
                response_text = output.outputs[0].text
                if response_text:
                    token_lengths.append(len(output.outputs[0].token_ids))
                    # Try to extract and validate JSON using Harmony
                    json_data = self.extract_final_json(output, schema)
                    if json_data is not None:
                        results.append(PromptResult(response_text, json_data))
                    else:
                        logger.warning(f"Invalid/unparseable JSON for prompt {i}")
                        failed_count += 1
                        results.append(None)
                else:
                    logger.warning(f"Empty response for prompt {i}")
                    failed_count += 1
                    results.append(None)
            else:
                logger.warning(f"No output generated for prompt {i}")
                failed_count += 1
                results.append(None)

        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} prompts")

        logger.info(f"Batch completed in {batch_time:.1f}s")
        logger.info(f"  Items per second: {len(prompts) / batch_time:.1f}")
        if token_lengths:
            logger.info(
                "  Response token stats: min=%d mean=%.1f max=%d",
                min(token_lengths),
                statistics.fmean(token_lengths),
                max(token_lengths),
            )

        return results

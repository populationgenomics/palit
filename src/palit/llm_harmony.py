#!/usr/bin/env python3
"""Harmony/vLLM backend for LLM batch processing.

Requires the optional ``ml`` dependencies (vLLM, openai-harmony) which are
only available on Linux GPU machines.
"""

import json
import logging
import statistics
import time
from typing import Any

import jsonschema
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.parser import harmony_utils
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import StructuredOutputsParams

from palit.llm import PromptResult

logger = logging.getLogger(__name__)


class HarmonyProcessor:
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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.reasoning_effort = reasoning_effort
        self.llm: LLM | None = None

    def _initialize_model(self) -> None:
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

    def _extract_final_json(
        self, output: RequestOutput, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract JSON from the final channel of Harmony-formatted output."""
        generated_tokens = output.outputs[0].token_ids

        try:
            _reasoning_content, final_content, _is_tool_call = harmony_utils.parse_chat_output(
                generated_tokens
            )

            if final_content:
                try:
                    parsed_json: dict[str, Any] = json.loads(final_content)
                    jsonschema.validate(parsed_json, schema)
                    return parsed_json
                except (json.JSONDecodeError, jsonschema.ValidationError):
                    logger.debug(f"Invalid JSON in final content: {final_content[:100]}")
                    return None

            logger.debug("No final content found, only reasoning")
            return None

        except Exception:
            logger.exception("Harmony parsing failed")
            return None

    async def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]:
        """Process a batch of prompts via vLLM offline batch inference.

        vLLM handles GPU-level parallelism internally, so the async wrapper
        simply runs the synchronous ``LLM.generate()`` call directly.
        """
        if not prompts:
            return []

        if self.llm is None:
            self._initialize_model()

        logger.info(f"Processing batch of {len(prompts)} prompts...")

        # Convert prompts to Harmony format token IDs
        token_prompts = []
        for prompt in prompts:
            system_msg = harmony_utils.get_system_message(reasoning_effort=self.reasoning_effort)
            user_msg = harmony_utils.get_user_message(prompt)
            tokens = harmony_utils.render_for_completion([system_msg, user_msg])
            token_prompt = TokensPrompt(prompt_token_ids=tokens)
            token_prompts.append(token_prompt)

        # Create sampling params with structured output
        structured_outputs = StructuredOutputsParams(json=schema)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            structured_outputs=structured_outputs,
            stop_token_ids=harmony_utils.get_stop_tokens_for_assistant_actions(),
        )

        start_time = time.time()

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
                    json_data = self._extract_final_json(output, schema)
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

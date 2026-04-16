#!/usr/bin/env python3
"""Bedrock batch inference backend for LLM processing.

Submits prompts as a batch job via S3, polls for completion, and collects
results. Provides a 50% cost discount over real-time Bedrock API calls.

Falls back to real-time ``PydanticAIProcessor`` when batch size < 100
(Bedrock's minimum).
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import boto3
import jsonschema

from palit.llm import PromptResult
from palit.llm_pydantic_ai import PydanticAIProcessor

logger = logging.getLogger(__name__)

MIN_BATCH_SIZE = 100
POLL_INTERVAL_SECONDS = 300


@dataclass
class _Sidecar:
    """Persisted state for an in-flight batch job."""

    job_arn: str
    record_ids: list[str]


class BedrockBatchProcessor:
    """Bedrock batch inference via S3.

    Uses the InvokeModel JSONL format with native JSON schema structured output
    and adaptive thinking.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
        *,
        s3_bucket: str,
        s3_prefix: str,
        role_arn: str,
        region: str,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        self._role_arn = role_arn

        session = boto3.Session(region_name=region)
        self._s3 = session.client("s3")
        self._bedrock = session.client("bedrock")

        # Real-time fallback for small batches
        self._realtime = PydanticAIProcessor(
            model=f"bedrock/{model_id}",
            temperature=temperature,
            max_tokens=max_tokens,
            region=region,
        )

    async def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]:
        if not prompts:
            return []

        if len(prompts) < MIN_BATCH_SIZE:
            logger.info(
                "Batch too small (%d < %d), using real-time API",
                len(prompts),
                MIN_BATCH_SIZE,
            )
            return await self._realtime.process_batch(prompts, schema)

        record_ids = [f"r-{i}" for i in range(len(prompts))]

        # Check for in-flight or completed job
        sidecar = self._read_sidecar()
        if sidecar and sidecar.record_ids == record_ids:
            logger.info("Found in-flight batch job: %s", sidecar.job_arn)
        else:
            if sidecar:
                logger.warning(
                    "Stale sidecar (record count mismatch: %d vs %d), resubmitting",
                    len(sidecar.record_ids),
                    len(record_ids),
                )
            sidecar = self._submit(prompts, schema, record_ids)

        # Poll until done
        status = await self._poll(sidecar.job_arn)

        if status != "Completed":
            self._delete_sidecar()
            return [None] * len(prompts)

        # Collect results (scoped to this job's output subdirectory)
        job_id = sidecar.job_arn.rsplit("/", 1)[-1]
        results = self._collect(record_ids, schema, job_id)
        self._delete_sidecar()
        return results

    def _build_model_input(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": "medium",
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                },
            },
        }

    def _submit(
        self,
        prompts: list[str],
        schema: dict[str, Any],
        record_ids: list[str],
    ) -> _Sidecar:
        lines = []
        for record_id, prompt in zip(record_ids, prompts, strict=True):
            record = {
                "recordId": record_id,
                "modelInput": self._build_model_input(prompt, schema),
            }
            lines.append(json.dumps(record))
        input_jsonl = "\n".join(lines) + "\n"

        input_key = f"{self._s3_prefix}/input.jsonl"
        output_uri = f"s3://{self._s3_bucket}/{self._s3_prefix}/output/"

        logger.info(
            "Uploading %d records to s3://%s/%s",
            len(prompts),
            self._s3_bucket,
            input_key,
        )
        self._s3.put_object(Bucket=self._s3_bucket, Key=input_key, Body=input_jsonl.encode())

        sanitized = re.sub(r"[^a-zA-Z0-9\-\+\.]", "-", self._s3_prefix)
        job_name = f"palit-{sanitized}-{int(time.time())}"
        logger.info("Creating batch inference job: %s", job_name)
        response = self._bedrock.create_model_invocation_job(
            jobName=job_name,
            modelId=self.model_id,
            roleArn=self._role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3InputFormat": "JSONL",
                    "s3Uri": f"s3://{self._s3_bucket}/{input_key}",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_uri,
                }
            },
        )

        sidecar = _Sidecar(
            job_arn=response["jobArn"],
            record_ids=record_ids,
        )
        self._write_sidecar(sidecar)
        logger.info("Job ARN: %s", sidecar.job_arn)
        return sidecar

    async def _poll(self, job_arn: str) -> str:
        while True:
            response = self._bedrock.get_model_invocation_job(jobIdentifier=job_arn)
            status: str = response["status"]
            logger.info("Batch job status: %s", status)

            if status in ("Completed", "Failed", "Stopped"):
                if status != "Completed":
                    logger.error("Batch job message: %s", response.get("message", "N/A"))
                return status

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def _collect(
        self, record_ids: list[str], schema: dict[str, Any], job_id: str
    ) -> list[PromptResult | None]:
        output_prefix = f"{self._s3_prefix}/output/{job_id}/"
        list_response = self._s3.list_objects_v2(Bucket=self._s3_bucket, Prefix=output_prefix)

        results_by_id: dict[str, PromptResult | None] = {}
        total_input_tokens = 0
        total_output_tokens = 0

        for obj in list_response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl.out"):
                continue

            logger.info("Downloading results from s3://%s/%s", self._s3_bucket, key)
            body = self._s3.get_object(Bucket=self._s3_bucket, Key=key)["Body"].read().decode()

            for line in body.strip().split("\n"):
                record = json.loads(line)
                record_id = record.get("recordId", "")

                if "error" in record:
                    error = record["error"]
                    logger.warning(
                        "Record %s failed: %s (code %s)",
                        record_id,
                        error.get("errorMessage", "unknown"),
                        error.get("errorCode", "?"),
                    )
                    results_by_id[record_id] = None
                    continue

                model_output = record.get("modelOutput", {})
                usage = model_output.get("usage", {})
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)

                content = model_output.get("content", [])
                # Adaptive thinking can produce multiple thinking+text block pairs
                # (undocumented behavior when thinking is skipped). Later blocks tend
                # to be more complete, so take the last text block.
                text_blocks = [b for b in content if b.get("type") == "text"]
                if len(text_blocks) > 1:
                    logger.info(
                        "Record %s: %d text blocks in response, using last",
                        record_id,
                        len(text_blocks),
                    )
                text_block = text_blocks[-1] if text_blocks else None
                if text_block is None:
                    logger.warning("Record %s: no text block in response", record_id)
                    results_by_id[record_id] = None
                    continue

                try:
                    parsed_json = json.loads(text_block["text"])
                    jsonschema.validate(parsed_json, schema)
                except (json.JSONDecodeError, jsonschema.ValidationError):
                    logger.exception("Record %s: invalid JSON output", record_id)
                    results_by_id[record_id] = None
                    continue

                raw_response = json.dumps(record)
                results_by_id[record_id] = PromptResult(
                    raw_response=raw_response,
                    parsed_json=parsed_json,
                )

        logger.info(
            "Batch complete: %d records, input=%d output=%d tokens, %d failures",
            len(results_by_id),
            total_input_tokens,
            total_output_tokens,
            sum(1 for v in results_by_id.values() if v is None),
        )

        return [results_by_id.get(rid) for rid in record_ids]

    # -- Sidecar persistence on S3 --

    def _sidecar_key(self) -> str:
        return f"{self._s3_prefix}/job.json"

    def _write_sidecar(self, sidecar: _Sidecar) -> None:
        data = {
            "jobArn": sidecar.job_arn,
            "recordIds": sidecar.record_ids,
            "submittedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._s3.put_object(
            Bucket=self._s3_bucket,
            Key=self._sidecar_key(),
            Body=json.dumps(data).encode(),
        )

    def _read_sidecar(self) -> _Sidecar | None:
        try:
            response = self._s3.get_object(Bucket=self._s3_bucket, Key=self._sidecar_key())
            data = json.loads(response["Body"].read().decode())
            return _Sidecar(
                job_arn=data["jobArn"],
                record_ids=data["recordIds"],
            )
        except self._s3.exceptions.NoSuchKey:
            return None

    def _delete_sidecar(self) -> None:
        self._s3.delete_object(Bucket=self._s3_bucket, Key=self._sidecar_key())

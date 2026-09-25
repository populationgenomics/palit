"""Lookup tools the model calls during evidence extraction.

* ``lookup_variants``: HGVS normalisation and gnomAD v4.1 counts from the
  variant-lookup service (https://github.com/populationgenomics/variant-lookup).
* ``lookup_genes``: HGNC resolution from the local HGNC data (approved symbol,
  location, locus group), which the inheritance-mode rules depend on.

Both take many items per call, so one extraction needs a single lookup round.
Neither is ``strict``: the extraction schema plus a strict tool exceeds the API's
compiled-grammar limit, so inputs are validated here instead.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx2
import jsonschema
import tenacity
from anthropic.types import ToolParam
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from palit.hgnc import HgncResolver

logger = logging.getLogger(__name__)

LOOKUP_VARIANTS_TOOL: ToolParam = {
    "name": "lookup_variants",
    "description": (
        "Normalises variants (Mutalyzer; protein changes are back-translated to candidate DNA "
        "changes) and looks up gnomAD v4.1 joint exome+genome counts. Takes many variants in one "
        "call: call it once with every variant from the paper. For each input returns either an "
        "error (notation could not be parsed or mapped) or one or more normalised candidates, each "
        "with gnomAD AC, AN, homozygote, heterozygote and hemizygote counts and FAF95 popmax, or a "
        "marker that the candidate is absent from gnomAD."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "gene_symbol": {
                            "type": "string",
                            "description": "Gene symbol of the gene the variant is in.",
                        },
                        "variant": {
                            "type": "string",
                            "description": (
                                "One notation exactly as in the gene's `variants` array "
                                "(e.g. NM_000123.4:c.1234A>T)."
                            ),
                        },
                        "genome_build": {"type": "string", "enum": ["GRCh38", "GRCh37", "unknown"]},
                    },
                    "required": ["gene_symbol", "variant", "genome_build"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["variants"],
        "additionalProperties": False,
    },
}

LOOKUP_GENES_TOOL: ToolParam = {
    "name": "lookup_genes",
    "description": (
        "Resolves gene symbols (approved, previous, or alias) against HGNC. Takes many symbols in "
        "one call: call it once with every gene symbol from the paper. For each input returns the "
        "approved symbol, HGNC ID, name, cytogenetic location (pseudoautosomal genes list both an X "
        "and a Y band, e.g. 'Xp22.33 and Yp11.2'), chromosome, and locus group, or 'not found'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
        "required": ["symbols"],
        "additionalProperties": False,
    },
}

TOOLS: list[ToolParam] = [LOOKUP_VARIANTS_TOOL, LOOKUP_GENES_TOOL]

_VALIDATORS = {
    tool["name"]: jsonschema.Draft202012Validator(tool["input_schema"]) for tool in TOOLS
}


# ---------------------------------------------------------------------------
# Variant-lookup service client
# ---------------------------------------------------------------------------

# Per-request timeout. Mutalyzer's first-touch on a cold reference-sequence
# cache can take well over a minute (NCBI E-fetch + back-translate fan-out).
# Steady-state requests finish in ~2-3s; we trade a few wasted seconds on
# truly stuck calls for not dropping legitimate cold-cache traffic.
_REQUEST_TIMEOUT_S = 180.0

# The service enforces 8 in-flight requests cluster-wide via nginx limit_conn;
# staying at parity avoids burning retries on 429s in the common case.
MAX_IN_FLIGHT = 8


class VariantLookupSettings(BaseSettings):
    """Loaded from environment + .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(..., alias="VARIANT_LOOKUP_BASE_URL")
    api_key: str = Field(..., alias="VARIANT_LOOKUP_API_KEY")


def _is_5xx_or_transport(exc: BaseException) -> bool:
    if isinstance(exc, httpx2.TransportError):
        return True
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class VariantLookupClient:
    """Async wrapper around ``POST /v1/variant``.

    Concurrency: ``max_in_flight`` semaphore. 429s honour ``Retry-After``
    (handled in-band); 5xx and transport errors retry via tenacity.
    """

    def __init__(self, settings: VariantLookupSettings, max_in_flight: int = MAX_IN_FLIGHT) -> None:
        self._endpoint = f"{settings.base_url.rstrip('/')}/v1/variant"
        self._headers = {"Authorization": f"Bearer {settings.api_key}"}
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._http = httpx2.AsyncClient(timeout=_REQUEST_TIMEOUT_S, follow_redirects=True)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def lookup_one(self, body: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            return await self._post_with_retries(body)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(7),
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        retry=tenacity.retry_if_exception(_is_5xx_or_transport),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        # Inner 429 loop: nginx limit_conn returns 429 + Retry-After when the
        # cluster cap is exceeded; honour the header rather than backing off
        # blindly. Bounded so a misbehaving service doesn't spin forever.
        for _ in range(10):
            response = await self._http.post(self._endpoint, json=body, headers=self._headers)
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            retry_after = float(response.headers.get("Retry-After", "1.0"))
            await asyncio.sleep(retry_after)
        response.raise_for_status()
        raise RuntimeError("unreachable: 429 loop fell through without raising")


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolOutcome:
    """The ``tool_result`` content for one ``tool_use`` block."""

    content: str  # JSON text shown to the model
    is_error: bool


def summarise_variant_response(item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """The model-facing result for one variant, keeping every normalised candidate."""
    error = response.get("error")
    if error is not None:
        return {
            **item,
            "status": "error",
            "error_code": error.get("code", ""),
            "error": error.get("message", ""),
        }
    candidates = [
        {
            "hgvs_c": entry.get("hgvs_c"),
            "hgvs_p": entry.get("hgvs_p"),
            "vcf": entry["pseudo_vcf"],
            "gnomad_v4_1": entry["frequency"]
            if entry.get("frequency") is not None
            else "absent from gnomAD v4.1",
        }
        for entry in response.get("normalized") or []
    ]
    if not candidates:
        return {
            **item,
            "status": "error",
            "error_code": "NO_NORMALIZED_VARIANTS",
            "error": "service returned an empty normalized[] list",
        }
    return {**item, "status": "ok", "candidates": candidates}


@dataclass(frozen=True)
class FrequencyRow:
    """One ``variant_frequencies`` row derived from a ``lookup_variants`` result."""

    variant_id: str  # pseudo-VCF on success; the original text on failure
    normalization: dict[str, Any]
    gnomad: dict[str, Any]


def frequency_row(result: dict[str, Any]) -> FrequencyRow:
    """Collapse a summarised lookup result to one row.

    When a protein change back-translates to several candidates, the one with the
    highest gnomAD AC is kept (ties: lexicographically first pseudo-VCF), since it
    is the candidate that could contradict pathogenicity.
    """
    if result["status"] == "error":
        return FrequencyRow(
            variant_id=result["variant"],
            normalization={
                "original_text": result["variant"],
                "error_code": result["error_code"],
                "error_message": result["error"],
                "upstream": None,
            },
            gnomad={"normalization_error": True},
        )

    def ac(candidate: dict[str, Any]) -> int:
        freq = candidate["gnomad_v4_1"]
        return int(freq["ac"]) if isinstance(freq, dict) else -1

    candidates: list[dict[str, Any]] = result["candidates"]
    best = min(candidates, key=lambda c: (-ac(c), c["vcf"]))
    freq = best["gnomad_v4_1"]
    return FrequencyRow(
        variant_id=best["vcf"],
        normalization={
            "hgvs_c": best["hgvs_c"],
            "hgvs_p": best["hgvs_p"],
            "original_text": result["variant"],
            "total_normalizations": len(candidates),
            "selected_for_max_ac": ac(best) if ac(best) >= 0 else None,
        },
        gnomad=freq if isinstance(freq, dict) else {"variant_not_found": True},
    )


class LookupRunner:
    """Executes ``lookup_variants`` and ``lookup_genes`` calls."""

    def __init__(self, variant_client: VariantLookupClient, hgnc_resolver: HgncResolver) -> None:
        self._variants = variant_client
        self._hgnc = hgnc_resolver

    async def run(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        validator = _VALIDATORS.get(name)
        if validator is None:
            return ToolOutcome(json.dumps({"error": f"unknown tool {name}"}), is_error=True)
        errors = [e.message for e in validator.iter_errors(tool_input)]
        if errors:
            return ToolOutcome(
                json.dumps({"error": "invalid input", "details": errors[:5]}), is_error=True
            )
        if name == LOOKUP_VARIANTS_TOOL["name"]:
            results = await asyncio.gather(
                *(self.lookup_variant(item) for item in tool_input["variants"])
            )
            return ToolOutcome(json.dumps({"results": list(results)}), is_error=False)
        return ToolOutcome(
            json.dumps({"results": [self.lookup_gene(symbol) for symbol in tool_input["symbols"]]}),
            is_error=False,
        )

    async def lookup_variant(self, item: dict[str, Any]) -> dict[str, Any]:
        """Look up one ``{gene_symbol, variant, genome_build}`` item."""
        body: dict[str, Any] = {"gene": item["gene_symbol"], "variant": item["variant"]}
        if item["genome_build"] != "unknown":
            body["genome_build"] = item["genome_build"]
        try:
            response = await self._variants.lookup_one(body)
        except httpx2.HTTPError as e:
            logger.warning(
                "Variant lookup failed for %s %s: %s", item["gene_symbol"], item["variant"], e
            )
            response = {
                "error": {
                    "code": "LOOKUP_FAILED",
                    "message": f"lookup service error: {type(e).__name__}",
                }
            }
        return summarise_variant_response(item, response)

    def lookup_gene(self, symbol: str) -> dict[str, Any]:
        entry = self._hgnc.resolve(symbol)
        if entry is None:
            return {"symbol": symbol, "status": "not found"}
        return {
            "symbol": symbol,
            "status": "ok",
            "approved_symbol": entry.symbol,
            "hgnc_id": f"HGNC:{entry.hgnc_id}",
            "name": entry.name,
            "location": entry.location,
            "chromosome": entry.chromosome,
            "locus_group": entry.locus_group,
        }

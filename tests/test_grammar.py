"""Every production structured-output configuration must compile on the API.

The binding limit is an undocumented cap on compiled grammar size, which counts
the output schema plus every declared tool (strict or not). Measured headroom for
the extraction schema is small, so each configuration a stage sends is checked
here with a 16-token request (a fraction of a cent each).

Needs Claude credentials, so it is opt-in: ``uv run pytest -m api``.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import pytest
from anthropic.types import ToolParam

from palit.llm import MODEL, AnthropicSettings, json_output_config, make_client
from palit.lookup_tools import TOOLS as EXTRACTION_TOOLS
from palit.mondo_tools import TOOLS as MONDO_TOOLS
from palit.scan_mechanisms import MechanismScanResult

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _schema(name: str) -> dict[str, Any]:
    return json.loads((PROMPTS / name).read_text())  # type: ignore[no-any-return]


# (stage, output schema, tools declared alongside it)
CONFIGURATIONS: list[tuple[str, dict[str, Any], list[ToolParam]]] = [
    ("relevance", _schema("relevance_assessment_schema.json"), []),
    ("relevance_panelapp", _schema("relevance_panelapp_check_schema.json"), []),
    ("extraction", _schema("evidence_extraction_schema.json"), EXTRACTION_TOOLS),
    ("assess_genes", _schema("aggregate_assessment_schema.json"), []),
    ("map_mondo", _schema("map_mondo_schema.json"), MONDO_TOOLS),
    ("match_panels", _schema("panel_matching_schema.json"), []),
    ("tournament", _schema("tournament_selection_schema.json"), []),
    ("scan_mechanisms", MechanismScanResult.model_json_schema(), []),
]


async def _compile(schema: dict[str, Any], tools: list[ToolParam]) -> None:
    client = make_client(AnthropicSettings())
    await client.messages.create(
        model=MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with any valid output."}],
        tools=tools,
        output_config=json_output_config(schema, "low"),
    )


@pytest.mark.api
@pytest.mark.parametrize(
    ("stage", "schema", "tools"), CONFIGURATIONS, ids=[c[0] for c in CONFIGURATIONS]
)
def test_configuration_compiles(stage: str, schema: dict[str, Any], tools: list[ToolParam]) -> None:
    try:
        asyncio.run(_compile(schema, tools))
    except anthropic.BadRequestError as e:
        pytest.fail(f"{stage}: {e.message}")

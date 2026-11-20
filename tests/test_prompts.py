"""The lumping and splitting criteria are one text, shared by the stages that apply them."""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from palit.extract_evidence import render_system_prompt

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
ASSOCIATION_STAGES = ["aggregate_assessment_prompt.j2", "expansion_triage_prompt.j2"]


def _render(template: str, **variables: object) -> str:
    env = Environment(loader=FileSystemLoader(PROMPTS), autoescape=False, undefined=StrictUndefined)
    return env.get_template(template).render(**variables)


def _criteria(paper_level: bool) -> str:
    return _render("partials/lump_split_criteria.j2", paper_level=paper_level).strip()


def test_extraction_applies_the_criteria_to_one_paper() -> None:
    prompt = render_system_prompt(PROMPTS / "evidence_extraction_prompt.j2", None)
    criteria = _criteria(paper_level=True)
    assert criteria in prompt
    assert "the paper presenting it as a distinct disease" in criteria
    assert "stays a separate disease entity" in criteria
    assert "association" not in criteria


@pytest.mark.parametrize("copies_gencc_terms", [True, False])
def test_association_stages_apply_the_criteria_across_papers(copies_gencc_terms: bool) -> None:
    rules = _render("partials/lump_split.j2", copies_gencc_terms=copies_gencc_terms)
    criteria = _criteria(paper_level=False)
    assert rules.startswith(criteria)
    assert "the papers presenting it as a distinct disease" in criteria
    assert "stays a separate association" in criteria


@pytest.mark.parametrize("template", ASSOCIATION_STAGES)
def test_association_stages_include_the_shared_rules(template: str) -> None:
    assert '{% include "partials/lump_split.j2" %}' in (PROMPTS / template).read_text()


def test_grouping_is_decided_by_clingen_then_panelapp_then_the_criteria() -> None:
    rules = _render("partials/lump_split.j2", copies_gencc_terms=True)
    order = [
        rules.index("**(1) ClinGen's curations of this gene:**"),
        rules.index("**(2) PanelApp Australia's existing listing, outside ClinGen's coverage:**"),
        rules.index("**(3) Criteria 1"),
        rules.index("**STEP 3 "),
    ]
    assert order == sorted(order)
    assert "does not mention a phenotype is not such a criterion" in rules

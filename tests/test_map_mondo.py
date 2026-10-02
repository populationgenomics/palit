"""Tests for the MONDO tools and the map-mondo stage, on a small OBO fixture."""

import asyncio
import json
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import pronto
import pytest
from anthropic.types import Message

from palit.gencc import GenccIndex
from palit.llm import LlmRequest, LlmResult, ResultStatus
from palit.map_mondo import (
    LAST_ROUND,
    SCHEMA_PATH,
    STAGE,
    Association,
    Conversation,
    MappingRunner,
    association_text,
    count_unmapped,
    parse_mapping,
    select_associations,
)
from palit.mondo_tools import MondoIndex, MondoToolRunner, tokens

ROOT = Path(__file__).resolve().parents[1]
SCHEMA: dict[str, Any] = json.loads((ROOT / SCHEMA_PATH).read_text())

OBO = """format-version: 1.2
ontology: mondo

[Term]
id: BFO:0000016
name: disposition

[Term]
id: MONDO:0000001
name: disease
is_a: BFO:0000016 ! disposition

[Term]
id: MONDO:0042489
name: disease susceptibility
is_a: BFO:0000016 ! disposition

[Term]
id: MONDO:0021125
name: disease characteristic

[Term]
id: MONDO:0000010
name: craniosynostosis characteristic
is_a: MONDO:0021125 ! disease characteristic

[Term]
id: MONDO:0005583
name: non-human animal disease
is_a: MONDO:0000001 ! disease

[Term]
id: MONDO:1000001
name: craniosynostosis, non-human animal
is_a: MONDO:0005583 ! non-human animal disease

[Term]
id: MONDO:0000100
name: craniosynostosis
def: "Premature fusion of one or more cranial sutures." []
synonym: "premature closure of cranial sutures" EXACT []
synonym: "CSO" RELATED []
xref: Orphanet:1531
xref: OMIMPS:123100
xref: UMLS:C0010278
is_a: MONDO:0000001 ! disease

[Term]
id: MONDO:0000101
name: craniosynostosis 7
def: "A craniosynostosis caused by heterozygous mutation in the SMAD6 gene." []
synonym: "craniosynostosis type 7" EXACT []
xref: OMIM:617439
is_a: MONDO:0000100 ! craniosynostosis

[Term]
id: MONDO:0000102
name: nonsyndromic craniosynostosis
is_a: MONDO:0000100 ! craniosynostosis

[Term]
id: MONDO:0000103
name: sagittal craniosynostosis
is_a: MONDO:0000102 ! nonsyndromic craniosynostosis

[Term]
id: MONDO:0000104
name: scaphocephaly variant
is_a: MONDO:0000103 ! sagittal craniosynostosis

[Term]
id: MONDO:0000110
name: synostosis syndrome
synonym: "craniosynostosis" BROAD []
is_a: MONDO:0000001 ! disease

[Term]
id: MONDO:0000200
name: hypertension
is_a: MONDO:0000001 ! disease

[Term]
id: MONDO:0000201
name: renovascular hypertension
is_a: MONDO:0000200 ! hypertension

[Term]
id: MONDO:0000300
name: obsolete craniosynostosis old
is_obsolete: true
replaced_by: MONDO:0000100
"""


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> MondoIndex:
    path = tmp_path_factory.mktemp("mondo") / "mondo.obo"
    path.write_text(OBO)
    return MondoIndex.from_ontology(pronto.Ontology(str(path)))


def _ids(index: MondoIndex, query: str, limit: int = 10) -> list[str]:
    return [hit.term.id for hit in index.search(query, limit)]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tokens_drop_stop_words_and_punctuation() -> None:
    assert tokens("SMAD6-related Craniosynostosis, type 7") == ["smad6", "craniosynostosis", "7"]


def test_search_ranks_label_then_exact_synonym_then_other_synonym_then_overlap(
    index: MondoIndex,
) -> None:
    ranked = _ids(index, "craniosynostosis")
    # The label match, then the BROAD synonym of another term, then partial overlaps,
    # the shortest unmatched remainder first.
    assert ranked[:2] == ["MONDO:0000100", "MONDO:0000110"]
    assert ranked[2:] == ["MONDO:0000101", "MONDO:0000102", "MONDO:0000103"]
    assert _ids(index, "premature closure of cranial sutures")[0] == "MONDO:0000100"
    # "type" is a stop word, so the label "craniosynostosis 7" matches exactly.
    assert _ids(index, "Craniosynostosis type 7")[0] == "MONDO:0000101"


def test_search_by_word_overlap_prefers_more_specific_names(index: MondoIndex) -> None:
    assert _ids(index, "SMAD6-related renovascular hypertension")[:2] == [
        "MONDO:0000201",
        "MONDO:0000200",
    ]


def test_search_excludes_obsolete_non_human_and_non_disease_terms(index: MondoIndex) -> None:
    found = set(_ids(index, "obsolete craniosynostosis old non-human animal characteristic"))
    assert found
    assert not found & {"MONDO:0000300", "MONDO:1000001", "MONDO:0000010"}


def test_search_by_xref_and_bounded_count(index: MondoIndex) -> None:
    assert _ids(index, "OMIM:617439") == ["MONDO:0000101"]
    assert _ids(index, "ORPHA:1531") == ["MONDO:0000100"]
    assert len(_ids(index, "craniosynostosis", limit=2)) == 2
    assert _ids(index, "of the") == []


def test_lookup_lists_xrefs_ancestors_to_the_root_and_bounded_descendants(
    index: MondoIndex,
) -> None:
    result = index.lookup_result("MONDO:0000100")
    assert result["status"] == "ok"
    assert result["xrefs"] == {"OMIMPS": ["OMIMPS:123100"], "Orphanet": ["Orphanet:1531"]}
    assert result["ancestors"] == [{"id": "MONDO:0000001", "label": "disease", "distance": 1}]
    assert [(d["id"], d["depth"]) for d in result["descendants"]] == [
        ("MONDO:0000102", 1),  # a grouping, listed before the single disease
        ("MONDO:0000101", 1),
        ("MONDO:0000103", 2),
        ("MONDO:0000104", 3),
    ]
    assert result["descendant_counts_by_depth"] == [2, 1, 1]

    deep = index.lookup_result("MONDO:0000104", max_ancestors=2)
    assert [(a["id"], a["distance"]) for a in deep["ancestors"]] == [
        ("MONDO:0000103", 1),
        ("MONDO:0000102", 2),
    ]
    assert deep["ancestors_not_listed"] == 2  # craniosynostosis and the disease root; no BFO

    bounded = index.lookup_result("MONDO:0000100", levels=2, max_descendants=1)
    assert [d["id"] for d in bounded["descendants"]] == ["MONDO:0000102"]
    assert bounded["descendant_counts_by_depth"] == [2, 1]


def test_lookup_of_obsolete_unknown_and_ineligible_terms(index: MondoIndex) -> None:
    assert index.lookup_result("MONDO:0000300") == {
        "id": "MONDO:0000300",
        "status": "obsolete",
        "label": "obsolete craniosynostosis old",
        "replaced_by": [{"id": "MONDO:0000100", "label": "craniosynostosis"}],
        "consider": [],
    }
    assert index.lookup_result("MONDO:9999999") == {"id": "MONDO:9999999", "status": "not found"}
    assert "note" in index.lookup_result("MONDO:1000001")


def test_tool_runner_validates_input(index: MondoIndex) -> None:
    runner = MondoToolRunner(index)
    ok = runner.run("search_mondo", {"queries": ["hypertension"]})
    assert not ok.is_error
    assert json.loads(ok.content)["results"][0]["candidates"][0]["id"] == "MONDO:0000200"
    assert runner.run("search_mondo", {"queries": ["a"] * 6}).is_error
    assert runner.run("lookup_mondo", {"id": "MONDO:0000100"}).is_error
    assert runner.run("other_tool", {}).is_error


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def _message(stop_reason: str, content: list[dict[str, Any]]) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "service_tier": "standard",
            },
        }
    )


def _answer(mondo_id: str, label: str = "renovascular hypertension") -> Message:
    answer = {"mondo_id": mondo_id, "label": label, "match": "broader", "rationale": "r"}
    return _message("end_turn", [{"type": "text", "text": json.dumps(answer)}])


@pytest.mark.parametrize(
    ("mondo_id", "problem"),
    [
        ("MONDO:9999999", "not in the loaded MONDO release"),
        ("MONDO:0000300", "obsolete"),
        ("MONDO:1000001", "not a human disease term"),
    ],
)
def test_parse_mapping_rejects_unknown_obsolete_and_ineligible_terms(
    index: MondoIndex, mondo_id: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_mapping(_answer(mondo_id), jsonschema.Draft202012Validator(SCHEMA), index)


def test_parse_mapping_takes_the_ontology_label(index: MondoIndex) -> None:
    mapping = parse_mapping(
        _answer("MONDO:0000201", "Renovascular Hypertension"),
        jsonschema.Draft202012Validator(SCHEMA),
        index,
    )
    assert (mapping.mondo_id, mapping.label, mapping.match) == (
        "MONDO:0000201",
        "renovascular hypertension",
        "broader",
    )


# ---------------------------------------------------------------------------
# Selection and storage
# ---------------------------------------------------------------------------


def _assessment(name: str | None) -> str:
    return json.dumps(
        {
            "proposed_disease_name": name,
            "description": "renal artery stenosis with hypertension",
            "inheritance_mode": "Monoallelic",
            "summary": "Two families.",
        }
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Gene 6772 with a GenCC-anchored association (1), an unmapped one (2) and a refused one (3)."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            """
            INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping,
                panelapp_context_json, unassessed_reports_json, quality_concerns_json)
            VALUES (6772, '{}', '{}', '{}', '[]', '[]')
            """
        )
        conn.execute(
            """
            INSERT INTO associations (id, hgnc_id, position, assessment_json, mondo_id,
                mondo_label, mondo_match)
            VALUES (1, 6772, 0, ?, 'MONDO:0000101', 'craniosynostosis 7', 'panelapp_gencc')
            """,
            (_assessment(None),),
        )
        conn.executemany(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (?, 6772, ?, ?)",
            [
                (2, 1, _assessment("SMAD6-related renovascular hypertension")),
                (3, 2, _assessment("SMAD6-related other disease")),
            ],
        )
        conn.execute(
            """
            INSERT INTO llm_requests (custom_id, stage, subject, round, model, status)
            VALUES ('map_mondo-1-x', ?, '3', 1, 'claude-opus-5-5', 'refused')
            """,
            (STAGE,),
        )
    return path


def test_select_associations_takes_only_unmapped_unrefused_rows(db_path: Path) -> None:
    assert select_associations(db_path, None) == [
        Association(
            id=2,
            hgnc_id=6772,
            proposed_disease_name="SMAD6-related renovascular hypertension",
            description="renal artery stenosis with hypertension",
            inheritance_mode="Monoallelic",
            summary="Two families.",
        )
    ]


def test_rows_of_a_deleted_aggregation_are_neither_selected_nor_counted(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM gene_aggregations WHERE hgnc_id = 6772")
        (left_behind,) = conn.execute("SELECT COUNT(*) FROM associations").fetchone()
    assert left_behind == 3
    assert select_associations(db_path, None) == []
    assert count_unmapped(db_path) == 0


def test_deleting_an_aggregation_cascades_where_foreign_keys_are_on(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM gene_aggregations WHERE hgnc_id = 6772")
        (left_behind,) = conn.execute("SELECT COUNT(*) FROM associations").fetchone()
    assert left_behind == 0


def test_association_text_lists_the_gene_and_its_gencc_rows(db_path: Path) -> None:
    (association,) = select_associations(db_path, None)
    text = association_text(association, "SMAD6", GenccIndex({}))
    assert "GENE: SMAD6 (HGNC:6772)" in text
    assert "Proposed disease name: SMAD6-related renovascular hypertension" in text
    assert "no GenCC submissions" in text


class ScriptedTransport:
    """Answers each request from a script keyed by (association id, round)."""

    def __init__(self, script: Callable[[int, int, LlmRequest], Message]) -> None:
        self._script = script
        self.requests: list[tuple[int, LlmRequest]] = []

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for i, request in enumerate(requests):
            self.requests.append((round_no, request))
            results.append(
                LlmResult(
                    custom_id=f"{stage}-{round_no}-{request.subject}-{i}-{len(self.requests)}",
                    batch_id=None,
                    stage=stage,
                    subject=request.subject,
                    round=round_no,
                    model="claude-opus-5-5",
                    status=ResultStatus.SUCCEEDED,
                    message=self._script(int(request.subject), round_no, request),
                    error_type=None,
                )
            )
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def _search_then_answer(mondo_id: str) -> Callable[[int, int, LlmRequest], Message]:
    def script(association_id: int, round_no: int, request: LlmRequest) -> Message:
        if round_no == 1:
            return _message(
                "tool_use",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_mondo",
                        "input": {"queries": ["renovascular hypertension"]},
                    }
                ],
            )
        return _answer(mondo_id)

    return script


def _conversation(association_id: int) -> Conversation:
    return Conversation(association_id, 1, [{"role": "user", "content": "map this"}])


def _row(db_path: Path, association_id: int) -> tuple[Any, ...]:
    with sqlite3.connect(db_path) as conn:
        return tuple(
            conn.execute(
                "SELECT mondo_id, mondo_label, mondo_match, mondo_raw IS NOT NULL "
                "FROM associations WHERE id = ?",
                (association_id,),
            ).fetchone()
        )


def _subjects(db_path: Path) -> list[tuple[str, int, str]]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT subject, round, status FROM llm_requests WHERE stage = ? AND subject = '2' "
            "ORDER BY round",
            (STAGE,),
        ).fetchall()


def test_runner_stores_a_valid_mapping_after_a_tool_round(db_path: Path, index: MondoIndex) -> None:
    transport = ScriptedTransport(_search_then_answer("MONDO:0000201"))
    runner = MappingRunner(
        transport=transport, db_path=db_path, system="s", schema=SCHEMA, index=index
    )
    outcome = asyncio.run(runner.advance([_conversation(2)]))

    assert (outcome.stored, outcome.failed) == (1, 0)
    assert _row(db_path, 2) == ("MONDO:0000201", "renovascular hypertension", "broader", 1)
    assert _subjects(db_path) == [("2", 1, "succeeded"), ("2", 2, "succeeded")]
    # The second round replays the first answer and carries the tool result.
    second = transport.requests[1][1].params["messages"]
    tool_result = second[2]["content"][0]
    assert tool_result["type"] == "tool_result" and not tool_result["is_error"]
    assert json.loads(tool_result["content"])["results"][0]["candidates"][0]["id"] == (
        "MONDO:0000201"
    )


def test_runner_records_but_does_not_store_an_obsolete_answer(
    db_path: Path, index: MondoIndex
) -> None:
    runner = MappingRunner(
        transport=ScriptedTransport(_search_then_answer("MONDO:0000300")),
        db_path=db_path,
        system="s",
        schema=SCHEMA,
        index=index,
    )
    outcome = asyncio.run(runner.advance([_conversation(2)]))

    assert (outcome.stored, outcome.failed) == (0, 1)
    assert _row(db_path, 2) == (None, None, None, 0)
    assert len(_subjects(db_path)) == 2
    assert [a.id for a in select_associations(db_path, None)] == [2]


def test_runner_stops_tool_calls_at_the_last_round(db_path: Path, index: MondoIndex) -> None:
    def always_search(association_id: int, round_no: int, request: LlmRequest) -> Message:
        return _search_then_answer("unused")(association_id, 1, request)

    transport = ScriptedTransport(always_search)
    runner = MappingRunner(
        transport=transport, db_path=db_path, system="s", schema=SCHEMA, index=index
    )
    outcome = asyncio.run(runner.advance([_conversation(2)]))

    assert (outcome.stored, outcome.failed) == (0, 1)
    assert [round_no for round_no, _ in transport.requests] == list(range(1, LAST_ROUND + 1))
    last_user_turn = transport.requests[-1][1].params["messages"][-1]["content"]
    assert last_user_turn[-1]["type"] == "text"

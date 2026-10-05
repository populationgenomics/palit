"""Tests for the tournament's per-gene prompt and its rounds."""

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from anthropic.types import Message

from palit.gencc import NO_PAA_ASSOCIATIONS, PAA_ASSOCIATIONS_CAVEAT, GenccIndex
from palit.hgnc import HgncResolver
from palit.llm import FALLBACK_MODEL, MODEL, LlmRequest, LlmResult, ResultStatus
from palit.papers import Paper, PubmedMetadata
from palit.tournament import TournamentEntry, run_tournaments, tournament_prompt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "prompts/tournament_selection_prompt.txt").read_text()


def _paper(doi: str, title: str) -> Paper:
    return Paper(
        doi=doi,
        pmid=None,
        title=title,
        abstract="Three families.",
        authors="",
        journal="",
        source="pubmed",
        source_date="2024-01-15",
        source_metadata=PubmedMetadata(),
        source_type="expansion",
        source_details="1",
    )


def test_entry_carries_the_genes_gencc_rows(
    hgnc_resolver: HgncResolver, gencc_index: GenccIndex
) -> None:
    entry = TournamentEntry.for_gene(1, [], hgnc_resolver, gencc_index)
    assert (entry.key, entry.gene_symbol) == ("1", "GENEA")
    assert entry.gencc_context == (
        "- disease A (MONDO:0000001) | Autosomal recessive | Strong\n"
        "- disease B (MONDO:0000002) | Autosomal dominant | Limited"
    )


def test_entry_without_gencc_rows_says_so(
    hgnc_resolver: HgncResolver, gencc_index: GenccIndex
) -> None:
    entry = TournamentEntry.for_gene(4, [], hgnc_resolver, gencc_index)
    assert entry.gencc_context == NO_PAA_ASSOCIATIONS


def test_prompt_shows_gene_curation_and_papers(
    hgnc_resolver: HgncResolver, gencc_index: GenccIndex
) -> None:
    papers = [_paper("10.1/a", "GENEA variants in disease B")]
    entry = TournamentEntry.for_gene(1, papers, hgnc_resolver, gencc_index)
    prompt = tournament_prompt(TEMPLATE, entry, papers, max_papers=5)
    assert "GENE: GENEA\n" in prompt
    assert (
        "(GenCC associations: disease | mode of inheritance | class):\n"
        f"{entry.gencc_context}\n\n{PAA_ASSOCIATIONS_CAVEAT} "
    ) in prompt
    assert "<paper id=0><date>2024-01-15</date><title>GENEA variants in disease B</title>" in prompt
    assert "UP TO 5 papers" in prompt


class RefusingTransport:
    """MODEL refuses every prompt; FALLBACK_MODEL keeps the first paper of each."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str, str]] = []  # (round, subject, model)

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for request in requests:
            model = request.params["model"]
            self.sent.append((round_no, request.subject, model))
            refused = model == MODEL
            message = Message.model_validate(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [] if refused else [{"type": "text", "text": '{"papers": [0]}'}],
                    "stop_reason": "refusal" if refused else "end_turn",
                    "stop_sequence": None,
                    "stop_details": (
                        {"type": "refusal", "category": "bio", "explanation": None}
                        if refused
                        else None
                    ),
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
            results.append(
                LlmResult(
                    custom_id=f"{stage}-{len(self.sent)}",
                    batch_id=None,
                    stage=stage,
                    subject=request.subject,
                    round=round_no,
                    model=model,
                    status=ResultStatus.REFUSED if refused else ResultStatus.SUCCEEDED,
                    message=message,
                    error_type=None,
                    error_message=None,
                )
            )
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def test_a_refused_prompt_is_resent_on_the_fallback_model_within_its_round(
    tmp_path: Path, hgnc_resolver: HgncResolver, gencc_index: GenccIndex
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    papers = [_paper(f"10.1/{i}", f"paper {i}") for i in range(4)]
    entry = TournamentEntry.for_gene(1, papers, hgnc_resolver, gencc_index)
    transport = RefusingTransport()
    outcomes = asyncio.run(
        run_tournaments(
            [entry],
            transport=transport,
            db_path=db_path,
            stage="expand_literature",
            prompt_template=TEMPLATE,
            schema=json.loads((ROOT / "prompts/tournament_selection_schema.json").read_text()),
            max_papers=1,
            papers_per_round=2,
            max_retries=2,
        )
    )
    # Round 1 has two prompts, round 2 one; each is refused once and then answered.
    assert transport.sent == [
        (1, "1:1:0", MODEL),
        (1, "1:1:1", MODEL),
        (1, "1:1:0", FALLBACK_MODEL),
        (1, "1:1:1", FALLBACK_MODEL),
        (2, "1:2:0", MODEL),
        (2, "1:2:0", FALLBACK_MODEL),
    ]
    assert [p.doi for p in outcomes["1"].selected_papers] == ["10.1/0"]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM llm_requests").fetchone() == (6,)

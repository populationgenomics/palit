"""Which papers and genes make up a run's corpus.

An initial paper (from the run's literature window) enters the corpus only when
its relevance result says it is relevant. A paper the relevance assessment
rejected, or has not assessed, is not extracted even when it has a PDF: the
ledger can carry a download status over from an earlier run. Expansion papers
(tournament picks, PanelApp-cited and referenced papers) are never assessed for
relevance; they are in the corpus once downloaded.

The run's genes are the genes with recent evidence from a relevant initial
paper. Only these genes get a tournament, PanelApp publication seeding,
referenced-paper discovery and an aggregation. An extraction from any paper
still counts as evidence for a gene of the run.
"""

import sqlite3

# SQL conditions on a `papers` row aliased `p`. `IS` compares NULL as a value, so
# each condition is 0 or 1, never NULL, and can also be summed or negated.
RELEVANT_PAPER = "json_extract(p.relevance_assessment_json, '$.relevant') IS 1"
CORPUS_PAPER = f"(p.source_type IS 'expansion' OR {RELEVANT_PAPER})"

# The HGNC IDs of the run's genes, for use as a subquery: `hgnc_id IN ({RUN_GENES})`.
RUN_GENES = f"""
    SELECT gm.hgnc_id FROM gene_mentions gm
    JOIN papers p ON p.doi = gm.paper_doi
    WHERE gm.source = 'recent_evidence' AND {RELEVANT_PAPER}
"""


def run_genes(conn: sqlite3.Connection) -> list[int]:
    """The run's genes, in HGNC ID order."""
    rows = conn.execute(f"SELECT DISTINCT hgnc_id FROM ({RUN_GENES}) ORDER BY hgnc_id").fetchall()
    return [row[0] for row in rows]

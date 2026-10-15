"""Which papers and genes make up a run's corpus.

The run's genes are the genes with recent evidence from a relevant initial paper.
Only these genes get a tournament, PanelApp citation seeding, referenced-paper
discovery and an aggregation. An extraction from any paper still counts as
evidence for a gene of the run.

A paper is in the corpus, and extracted once downloaded, when it is one of:

- an initial paper (from the run's literature window, or copied from the
  baseline in a panel-scoped run) whose relevance result says it is relevant;
- an expansion paper (tournament picks, referenced papers, and PanelApp-cited
  papers seeding added, copied from the baseline or fetched), which is never
  assessed for relevance;
- a paper PanelApp cites for one of the run's genes (table
  ``panelapp_citations``, filled by seeding), whatever relevance said about it.

An initial paper outside these is not extracted even when it has a PDF: the
ledger can carry a download status over from an earlier run. The run's genes come
from relevant papers only, however a paper entered the corpus: the extraction of
a cited paper the screen rejected adds no genes to the run, while that of a cited
paper the screen passed does, as for any relevant paper. Such a paper can be
extracted only after seeding, for example when reduce-literature had left it
unscheduled and seeding scheduled it again, so the run's genes can grow after
seeding, and seeding is then run again for the new genes.
"""

import sqlite3

# SQL conditions on a `papers` row aliased `p`. `IS` compares NULL as a value, so
# each condition is 0 or 1, never NULL, and can also be summed or negated.
RELEVANT_PAPER = "json_extract(p.relevance_assessment_json, '$.relevant') IS 1"

# The HGNC IDs of the run's genes, for use as a subquery: `hgnc_id IN ({RUN_GENES})`.
RUN_GENES = f"""
    SELECT gm.hgnc_id FROM gene_mentions gm
    JOIN papers p ON p.doi = gm.paper_doi
    WHERE gm.source = 'recent_evidence' AND {RELEVANT_PAPER}
"""

# A paper PanelApp cites for one of the run's genes.
CITED_PAPER = f"""p.doi IN (
    SELECT pc.paper_doi FROM panelapp_citations pc WHERE pc.hgnc_id IN ({RUN_GENES})
)"""

# A paper the run takes in beyond its relevance screen: an expansion paper, or a
# paper PanelApp cites for one of the run's genes.
EXPANSION_OR_CITED_PAPER = f"(p.source_type IS 'expansion' OR {CITED_PAPER})"

CORPUS_PAPER = f"({EXPANSION_OR_CITED_PAPER} OR {RELEVANT_PAPER})"


def run_genes(conn: sqlite3.Connection) -> list[int]:
    """The run's genes, in HGNC ID order."""
    rows = conn.execute(f"SELECT DISTINCT hgnc_id FROM ({RUN_GENES}) ORDER BY hgnc_id").fetchall()
    return [row[0] for row in rows]

# PanelApp Australia Literature Assessment

LLM-based literature assessment system for rare disease gene curation. It screens papers for relevance and extracts evidence against PanelApp Australia diagnostic criteria. It then groups each gene's evidence into gene-disease-MoI associations and rates each one. The report shows every association next to what PanelApp Australia already curates, with the panels it matches.

## Setup

### Installation

```bash
# Installation (all pipeline stages; LLM calls go to the Claude API)
uv sync

# With ML dependencies (only for the PubMed baseline screening classifier)
uv sync --extra ml

# The report's PDF viewer page (Node.js 24+; rebuild after viewer/ changes)
npm --prefix viewer ci && npm --prefix viewer run build
```

Install the [NCBI EDirect tools](https://www.ncbi.nlm.nih.gov/books/NBK565821/):

```bash
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
```

Enable the local pre-commit hook (runs the same lint set as CI):

```bash
uv run pre-commit install
```

### Database Setup

The system uses multiple databases:

- **Main workflow** (`data/db.sqlite`): Per-run database, created from `schema.sql`, seeded from the ledger by `palit ingest-pubmed`
- **Ingestion ledger** (`data/pubmed_ingestion_ledger.sqlite`): Created from `ledger_schema.sql` by `palit ledger init`. The single canonical dedup/disposition memory across all runs (see "PubMed ingestion ledger" below)
- **Screening workflow** (`data/pubmed_baseline_screening.sqlite`): Created from `schema.sql` by `palit screen-pubmed`
- **Classifier training** (`data/screening_classifier/training.sqlite`): Created from `src/palit/screening_classifier/training.sql` (only needed for training)

Both main and screening workflows use the same schema for consistency, allowing the same tools (e.g., `assess-relevance`) to work on both databases.

Stages that read PanelApp Australia's GenCC submissions or the MONDO ontology download `gencc_submissions.tsv` and `mondo.obo` into the run database's directory (e.g. `data/`) when the files are missing or older than 7 days.

### Claude API

LLM stages call the Claude API through the Anthropic SDK, authenticated with an `ant` OAuth profile instead of an API key. Log in once per profile (`brew install anthropics/tap/ant` first if needed):

```bash
ant auth login --profile <profile>
```

Palit reads the profile name from `PALIT_ANTHROPIC_PROFILE`, set either in the environment or in `.env` (see `.env.example`). The variable is required: an LLM stage stops with a validation error naming it before sending any request. Palit deliberately ignores `ANTHROPIC_PROFILE` and `ANTHROPIC_API_KEY`, which Claude Code sessions may export for their own workspace. The SDK refreshes the access token itself. The refresh token eventually expires, so when a previously working profile starts failing authentication, run `ant auth login --profile <profile>` again. Run one palit process per profile: the SDK serialises token refreshes within a process only.

Requests go to Claude Opus 5.5 (`MODEL` in `src/palit/llm.py`). When Opus 5.5 refuses a paper, gene or association in a stage, the stage's next attempt sends that subject to Claude Sonnet 5.5 (`FALLBACK_MODEL`) with the same prompt, schema, effort and token limit. The multi-round stages (`extract-evidence`, `map-mondo`) restart the subject's conversation from round 1 on Sonnet 5.5, since thinking blocks cannot be replayed to another model, and the tournament resends a refused prompt on Sonnet 5.5 within its round. A subject is refused for good only when Sonnet 5.5 refuses it too; this is what "refused" means in the stage summaries and the report, and stages skip such subjects. The fallback attempt runs in the same invocation and counts toward `--max-retries`. With `--limit`, a stage runs one attempt, so the fallback waits for the next invocation. `--retry-refused` on `assess-relevance` and `extract-evidence` sends the subjects refused for good in earlier invocations through both models once more. The report marks the relevance assessments, extractions and aggregations that came from Sonnet 5.5. The fallback is palit's own because the Batches API rejects the server-side `fallbacks` parameter.

`assess-relevance` settles a paper refused for good as not relevant, so that `ledger writeback` marks it settled and later runs do not ingest it again. Its `relevance_assessment_json` keeps the usual two-level shape and records the refusal:

```json
{"relevant": false, "screen": null, "panelapp_check": null,
 "refused": {"level": "screen", "model": "claude-sonnet-5-5", "category": "bio"}}
```

`level` is `screen` or `panelapp_check`. For a refusal at the PanelApp check, `screen` holds the scope screen the paper passed, and the paper's genes stay in `gene_mentions` like those of any paper the screen passed. Every invocation first settles the papers refused for good that still have no result, since a run database's recorded refusals may lack one. With `--retry-refused`, it reopens the settled papers instead, and any that both models refuse again are settled again. The report shows these papers as refused by both models: they are kept out of the screen misses, the PanelApp-check rejections and the low-confidence review list, and the panel-publication sensitivity leaves them out.

`uv run palit llm costs --db-path data/db.sqlite` shows requests, refusals, tokens, and USD cost per stage, model and service tier for a run database. Every stage prints the same summary for itself when it finishes, listing each refused request; `uv run palit llm refusals --db-path data/db.sqlite` lists all refusals with their safety-classifier category and the subject's outcome: recovered (answered after the refusal), refused for good (Sonnet 5.5 refused it last), or not answered yet. `uv run pytest -m api` checks that every stage's structured-output configuration still compiles on the API for both models (it needs the profile above).

### External Services

#### Variant Frequency Lookup

Evidence extraction (`palit extract-evidence`) requires a running [variant-lookup](https://github.com/populationgenomics/variant-lookup) service. Copy `.env.example` to `.env` and set both:

```
VARIANT_LOOKUP_BASE_URL=https://<host>:<port>
VARIANT_LOOKUP_API_KEY=<bearer-token>
```

`.env` is gitignored. The command exits immediately on startup if either variable is missing.

## Complete Workflow

```bash
# Configuration. PANEL_DATE selects the PanelApp snapshot the run is assessed
# against. Date it on the day the run is set up, after the literature window has
# closed, so that a gene a curator added during the window is already on its
# panel when the run assesses it.
PANEL_DATE=2025-10-20
START_DATE=2025-10-01
END_DATE=2025-10-15
LEDGER=data/pubmed_ingestion_ledger.sqlite

# 0. One-time: create the ledger — the dedup/disposition memory shared across all
#    runs (see "PubMed ingestion ledger" below). Seed it from existing run DBs with
#    `palit ledger seed` if you have prior corpora.
uv run palit ledger init --ledger $LEDGER

# 1. Ingest papers for the window through the ledger. The ledger replaces the old
#    buffer window + --previous-db: late-indexed stragglers arrive via the FTP
#    update-file sync (constant cost, unbounded horizon), and papers already settled
#    (assessed not relevant, or downloaded) are never reconsidered while
#    relevant-not-downloaded papers are re-emitted for a download retry. Preprints
#    first so their metadata (version) survives for automatic PDF download;
#    ingest-pubmed backfills PMIDs into preprint rows without overwriting them.
uv run palit ingest-preprints --ledger $LEDGER $START_DATE $END_DATE
uv run palit ingest-pubmed --ledger $LEDGER $START_DATE $END_DATE

# 2. Assess relevance of papers in two levels, sent as Message Batches. A scope
#    screen of every title and abstract lists the genes of papers with human
#    genetic evidence; a PanelApp check then compares those genes with their
#    entries on the target panels (at PANEL_DATE) and with PanelApp Australia's
#    GenCC rows, which rate each association separately, and keeps a paper only
#    if a gene is new, has a new disease or inheritance mode, or the association
#    is still amber or red. Safe to interrupt and re-run: it re-attaches to
#    batches still in flight. Papers Opus 5.5 refuses go to Sonnet 5.5 (see
#    "Claude API" above); papers both refuse are settled as not relevant, and
#    --retry-refused sends them once more.
uv run palit assess-relevance --panel-date $PANEL_DATE

# 2a. (Optional) Screen the PubMed baseline with the retrospective prompt. The
#     baseline is a comprehensive repository, so the screen decides alone.
uv run palit assess-relevance --db-path data/pubmed_baseline_screening.sqlite --prompt-path prompts/retrospective_screening_prompt.txt --screen-only

# 3. Download full-text papers (automated PMC + preprints, manual fallback)
uv run palit download-papers attempt-pmc
uv run palit download-papers download-preprints
uv run palit download-papers open-browser
# ... manually download PDFs to data/papers/ ...
uv run palit download-papers register

# 4. Extract evidence from the PDFs: Claude reads each PDF, looks up its genes
#    (HGNC) and variants (gnomAD v4.1, via the variant-lookup service; requires
#    VARIANT_LOOKUP_* env vars, see Setup) in one round, and cites verbatim
#    quotes. Two batch rounds; safe to interrupt and re-run. Papers Opus 5.5
#    refuses are extracted by Sonnet 5.5; re-runs skip papers both refused. Add
#    --retry-refused to send each of them once more, since the safety
#    classifier does not refuse the same paper every time.
#    A PDF too long for the context window, often an article with all its
#    supplements in one file, is sent with only the leading pages that fit,
#    and the model is told so. Quotes are still checked against the full PDF,
#    which the report shows.
#    Each disease entity carries three family counts. The reported count is
#    every family the paper reports. The qualifying count (`family_count`) is
#    the families whose genotype passes the qualifying variant gate. The
#    independent count drops families that share the variant's ancestral
#    origin; criteria A to C use it. For registries and case series without
#    family structure, the qualifying and independent counts are derived from
#    the patients who carry a qualifying genotype.
#    An initial paper is extracted only when assess-relevance found it relevant,
#    even if a PDF is on file; expansion papers are always extracted. The genes
#    with evidence from a relevant paper are this run's genes: steps 5, 6 and 9
#    work on them alone.
uv run palit extract-evidence

# 5. Discover papers referenced in evidence (citation-based expansion)
uv run palit discover-citations discover

# 5a. Optionally add papers manually that weren't found automatically
uv run palit discover-citations add --gene GENE_SYMBOL PMID1 PMID2 ...

# 6. Expand literature beyond citations. Tournament selection over the screened
#    baseline, bounded by --cutoff-date to the literature preceding the window
#    and favouring associations that PanelApp Australia's GenCC rows rate below
#    Strong or do not list, then unconditional seeding of the publications
#    PanelApp already cites for each gene in the --panel-date snapshot (see
#    "PanelApp publication seeding" below).
uv run palit expand-literature --cutoff-date $START_DATE --panel-date $PANEL_DATE

# 7. Download expansion papers (same workflow as step 3)
uv run palit download-papers attempt-pmc
uv run palit download-papers download-preprints
uv run palit download-papers open-browser --expansion-only
# ... manually download PDFs to data/papers/ ...
uv run palit download-papers register

# 8. Extract evidence from expansion papers (same as step 4)
uv run palit extract-evidence

# 9. Aggregate each gene's evidence across papers into gene-disease-MoI
#    associations, one request per gene. The model groups the papers' disease
#    entities into associations, anchored on what PanelApp Australia already
#    curates for the gene: its GenCC rows, and its entries and reviews on every
#    target panel that holds it. Each association records its relation to
#    PanelApp Australia (existing, new disease or new MoI) and its rating from
#    the evidence in this run's corpus. A gene is skipped when PanelApp already
#    cites all of its papers. Safe to interrupt and re-run: it re-attaches to
#    batches still in flight and skips genes that already have an aggregation.
uv run palit assess-genes --panel-date $PANEL_DATE

# 10. Map each association that reuses no PanelApp Australia GenCC row onto a
#     MONDO term. Claude searches the local MONDO release with tools and
#     answers with a term and a match type: exact when the term names the
#     disease, broader when it is the most specific term that includes the
#     disease. The report marks broader terms. Requests go out immediately, not
#     as batches, because a mapping takes several tool rounds. Re-runs map only
#     the associations still without a term.
uv run palit map-mondo

# 11. Match each association to diagnostic panels by its disease, MoI and
#     summary
uv run palit match-panels --panel-date $PANEL_DATE

# 12. Generate the report package: index.html, the PDF viewer page, and each
#     cited paper's PDF (symlinked) with its quote highlights (citations/*.json).
#     Each gene shows one block per association, with its relation to PanelApp
#     Australia and its rating in this corpus. `aws s3 sync` uploads the
#     symlink targets.
uv run palit generate-report --report-id report_mendeliome --panel-date $PANEL_DATE

# 13. Fold this run's dispositions back into the ledger so future runs skip the
#     papers settled here and resume any relevant-not-downloaded ones. Covers
#     expansion/discovered-citation papers too (keyed by DOI). Separate from, and
#     run alongside, the baseline-screening update below.
uv run palit ledger writeback --db-path data/db.sqlite --run-id report_mendeliome --ledger $LEDGER
```

## PanelApp Publication Seeding

Tournament selection keeps a minimal, non-redundant evidence set, so it routinely
discards papers a PanelApp curator has already cited — a single-family report loses
to a comprehensive cohort by design. Others were never candidates at all, because the
screened baseline does not reach back far enough to contain them. Either way the
report's "new evidence" framing ends up resting on papers we never read.

`expand-literature` therefore finishes with an unconditional seeding pass: for every
gene this run assesses, the publications PanelApp cites for it — the gene entity's
list plus those cited in individual reviews, the same union `assess-genes` compares
evidence against — are fetched from PubMed (batched efetch) or CrossRef and stored as
expansion papers with `source_details = 'panelapp:{hgnc_id}'`. They then flow through
the normal download and extraction steps.

Seeding runs after `--force-all`'s expansion wipe, so seeded papers survive it, and
before the tournament's early return, so a rerun still picks up publications added to
a panel since. It is idempotent: papers already held are counted and skipped.

Publications that never resolve to a DOI, and those whose PDF cannot be obtained, are
logged and simply stay out of the corpus — seeding widens coverage but does not
guarantee it.

## PubMed Ingestion Ledger

PubMed keeps indexing records weeks-to-months after their create date (CRDT), so a
fixed fetch window silently loses thousands of late-indexed papers every month. The
ingestion ledger (`data/pubmed_ingestion_ledger.sqlite`) is a single canonical
database that makes ingestion robust to this lag. It records, per DOI, the refreshed
bibliographic metadata plus the terminal disposition each run wrote back, and is the
dedup/disposition memory that replaces the old per-run buffer window + `--previous-db`
set-difference.

Two sources feed it, complementary by recency:

- **FTP update files** (`https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/`) — applied
  incrementally by file number. A late-indexed straggler arrives in whatever file
  first adds it, so old-date papers are caught at constant per-run cost without
  re-fetching old days. Revised records (e.g. a late-attached abstract) reappear in
  later files and refresh the row.
- **Thin live efetch** over the current window — the freshest view of the newest
  papers, where the FTP files can briefly lag.

Each run partitions previously-seen DOIs into **settled** (assessed not relevant,
including papers both models refused to assess, or downloaded — never reconsidered)
and **actionable** (not downloaded, and either never assessed or relevant —
re-emitted into the run). A downloaded paper is settled whatever its relevance: the
expansion papers a run writes back are never assessed for relevance, and they must
not return as new papers. A CRDT month is finalised and dropped from the actionable
set after a 6-month **closure horizon**, which bounds the work set.

```bash
LEDGER=data/pubmed_ingestion_ledger.sqlite

# Create the ledger once.
uv run palit ledger init --ledger $LEDGER

# Seed it from existing run databases (cross-era: DOI- and PMID-keyed corpora).
uv run palit ledger seed --ledger $LEDGER --db data/db_2026_april.sqlite --db data/db_2026_may.sqlite ...

# Apply any new FTP update files to the ledger (also done inside ingest-pubmed).
uv run palit ledger sync --ledger $LEDGER

# Per run: ingest-pubmed seeds the run DB from the ledger's actionable set;
# at run end, fold dispositions back.
uv run palit ledger writeback --db-path data/db.sqlite --run-id <run> --ledger $LEDGER
```

## Relevance Screening Classifier

### Training Workflow

```bash
# 1. Install ML dependencies and setup W&B
uv sync --extra ml
wandb login

# 2. Create training database
sqlite3 data/screening_classifier/training.sqlite < src/palit/screening_classifier/training.sql

# 3. Extract positive PMIDs from main workflow database
uv run palit screening-classifier extract-pmids

# 4. Prepare training data (fetches negatives from PubMed, assigns train/val/test splits)
uv run palit screening-classifier prepare-data

# 5. Train classifier
uv run palit screening-classifier train

# 6. Evaluate classifier
uv run palit screening-classifier evaluate
```

Model outputs saved to `outputs/best_model/` (HuggingFace format + optimal threshold).

### Screening PubMed Baseline

Once trained, use the classifier to screen PubMed baseline XML files:

```bash
# Download PubMed baseline (all XML files + checksums, ~47GB compressed)
mkdir -p data/pubmed_baseline
cd data/pubmed_baseline

for kind in baseline updatefiles
do
        curl -s https://ftp.ncbi.nlm.nih.gov/pubmed/$kind/ | \
                grep -oP '(?<=href=")[^"]*\.(xml\.gz|md5)' | \
                parallel --bar -j 8 "if [ ! -f {} ]; then curl -s -O \"https://ftp.ncbi.nlm.nih.gov/pubmed/$kind/{}\"; else echo \"{} exists, skipping.\"; fi"
done

cd ../..

# Screen baseline files with trained classifier
uv run palit screen-pubmed \
  --checkpoint outputs/best_model \
  --baseline-dir data/pubmed_baseline \
  --output-db data/pubmed_baseline_screening.sqlite
```

Relevant papers are stored in `pubmed_baseline_screening.sqlite`. Processing progress is tracked in `data/screening_progress.json` for resumability.

### Retrospective Assessment of Baseline Screening

For historical baseline screening (2000-2025), use the **retrospective screening prompt** which evaluates papers based on the evidence they provide rather than novelty:

```bash
# Retrospective mode: evaluates historical evidence value, not novelty
uv run palit assess-relevance \
  --db-path data/pubmed_baseline_screening.sqlite \
  --prompt-path prompts/retrospective_screening_prompt.txt \
  --screen-only
```

**Key difference from standard relevance assessment:**

- **Standard prompt** (`relevance_assessment_prompt.txt`): the scope screen of the monthly run, followed by the PanelApp check that asks whether the evidence is new relative to the target panels - optimized for recent literature
- **Retrospective prompt** (`retrospective_screening_prompt.txt`): Asks "Does this provide SUBSTANTIAL evidence for gene-disease relationships?" - optimized for historical baseline screening

The retrospective prompt evaluates papers in their historical context, accepting important early descriptions of gene-disease associations even if those genes are now well-established. This ensures comprehensive coverage across 25 years of literature for downstream tournament selection and analysis.

### Updating the Baseline

After each fortnightly processing run completes, feed the papers its scope screen passed back into the baseline screening DB so it grows as a comprehensive repository. This includes papers the PanelApp check found already curated, which the monthly report leaves out:

```bash
FORTNIGHTLY_DB=data/db_2026_february_h1.sqlite

sqlite3 data/pubmed_baseline_screening.sqlite <<'SQL'
ATTACH '$FORTNIGHTLY_DB' AS source;

CREATE TEMP TABLE relevant_dois AS
SELECT doi FROM source.papers
WHERE relevance_assessment_json IS NOT NULL
  AND json_extract(relevance_assessment_json, '$.screen.relevant') = 1;

INSERT OR IGNORE INTO papers
  (doi, pmid, title, abstract, authors, journal, source_date,
   source, source_metadata, source_type, source_details, download_status,
   relevance_assessment_raw, relevance_assessment_json)
SELECT doi, pmid, title, abstract, authors, journal, source_date,
       source, source_metadata, source_type, source_details, 'scheduled',
       relevance_assessment_raw, relevance_assessment_json
FROM source.papers WHERE doi IN relevant_dois;

INSERT OR IGNORE INTO gene_mentions
  (hgnc_id, paper_gene_symbol, paper_doi, source)
SELECT hgnc_id, paper_gene_symbol, paper_doi, source
FROM source.gene_mentions
WHERE source = 'relevance_assessment'
  AND paper_doi IN relevant_dois;

DROP TABLE relevant_dois;
DETACH source;
SQL
```

This step is tracked as `UPDATE_BASELINE` in the pipeline tracker and also syncs the updated baseline to the cluster.

### Panel-Specific Curation

For curating literature for a specific panel (e.g., Arthrogryposis):

```bash
# Configuration. As above, PANEL_DATE is the day the run is set up.
PANEL_DATE=2025-10-20
PANEL_ID=47  # Arthrogryposis panel ID
PANEL_NAME=arthrogryposis

# 1. Copy pre-filtered baseline DB papers to new database. Only the paper
#    metadata is copied: step 2 assesses relevance afresh and schedules the
#    relevant papers for download.
sqlite3 data/$PANEL_NAME.sqlite < schema.sql
sqlite3 data/$PANEL_NAME.sqlite "ATTACH 'data/pubmed_baseline_screening.sqlite' AS source; INSERT INTO papers (doi, pmid, title, abstract, authors, journal, source, source_date, source_metadata, source_type, source_details) SELECT doi, pmid, title, abstract, authors, journal, source, source_date, source_metadata, 'initial', source_details FROM source.papers"

# 2. Assess relevance scoped to the panel: the screen gets the panel description,
#    and the PanelApp check compares against this panel only
uv run palit assess-relevance \
  --db-path data/$PANEL_NAME.sqlite \
  --panel-date $PANEL_DATE \
  --scope-panel-id $PANEL_ID \
  --prompt-path prompts/panel_relevance_assessment_prompt.txt

# 3. (Optional) Reduce literature for well-researched genes
# Genes with hundreds of papers make the aggregation prompt long and slow. For panels with well-researched genes (e.g., POLG
# with 200+ papers), use tournament selection to keep only the most informative.
# It counts and selects among the papers step 2 assessed relevant:
uv run palit reduce-literature --db-path data/$PANEL_NAME.sqlite

# 4. Download full-text papers (now reduced set if step 3 was run)
uv run palit download-papers attempt-pmc --db-path data/$PANEL_NAME.sqlite
uv run palit download-papers download-preprints --db-path data/$PANEL_NAME.sqlite
uv run palit download-papers open-browser --db-path data/$PANEL_NAME.sqlite
# ... manually download PDFs ...
uv run palit download-papers register --db-path data/$PANEL_NAME.sqlite

# 5. Extract evidence and assess genes (panel-scoped). Each association's
#    summary explains how it relates to the panel's scope.
uv run palit extract-evidence --db-path data/$PANEL_NAME.sqlite --panel-date $PANEL_DATE --scope-panel-id $PANEL_ID
uv run palit assess-genes --db-path data/$PANEL_NAME.sqlite --panel-date $PANEL_DATE --target-panel-ids $PANEL_ID --scope-panel-id $PANEL_ID

# 6. Map associations without a GenCC row onto MONDO terms, as in the main
#    workflow. Scoped runs skip match-panels.
uv run palit map-mondo --db-path data/$PANEL_NAME.sqlite

# 7. Generate report package with panel-scoped novelty detection
uv run palit generate-report \
  --report-id panel_$PANEL_NAME \
  --db-path data/$PANEL_NAME.sqlite \
  --panel-date $PANEL_DATE \
  --target-panel-ids $PANEL_ID
```

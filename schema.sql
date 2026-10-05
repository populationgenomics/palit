-- Gene-Panel Centric Literature Assessment Database Schema
-- Creates fresh database with complete schema for gene-panel workflow

-- Enable Write-Ahead Logging (WAL) mode: readers (e.g. report generation or ad-hoc
-- queries) don't block a running stage's writes. This setting persists in the
-- database file.
PRAGMA journal_mode=WAL;

-- Core papers table
CREATE TABLE papers (
    doi TEXT PRIMARY KEY,
    pmid INTEGER,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    journal TEXT,

    -- Where the paper came from ('pubmed', 'biorxiv', 'medrxiv', etc.)
    source TEXT NOT NULL,

    -- Source-specific date (e.g., entrez_date for PubMed, posted_date for preprints)
    source_date DATE,

    -- Source-specific metadata as JSON (e.g., {"pmid": 12345, "pmcid": "PMC...", "version": "1"})
    source_metadata JSON,

    -- Where the paper came from ('initial' for primary search, 'expansion' for supplementary literature)
    source_type TEXT,
    source_details TEXT,

    -- Download status tracking
    download_status TEXT CHECK(download_status IN ('scheduled', 'downloaded', 'manual_required')),

    -- Relevance assessment of title and abstract, in two levels: a scope screen,
    -- then a check of the screen's genes against PanelApp (see assess_relevance.py)
    relevance_screen_raw JSON,  -- The Claude message of the scope screen
    relevance_screen_json JSON,  -- The parsed screen, kept until the PanelApp check is done
    relevance_assessment_raw JSON,  -- {"screen": message or null, "panelapp_check": message or null}
    relevance_assessment_json JSON,  -- Final: {"relevant", "screen", "panelapp_check"}, plus
                                     -- "refused" {"level", "model", "category"} when both models
                                     -- refused a level (then relevant is false; screen is null
                                     -- for a refused screen)
    evidence_extraction_raw TEXT,  -- The final Claude message of the extraction conversation
    evidence_extraction_json JSON
);

-- Normalized gene-paper relationships (automatically maintained from evidence extraction)
-- Tracks which papers mention which genes with patient/disease evidence
-- Query relevance_assessment_json.panelapp_check.associations or evidence_extraction_json.disease_entities for actual disease associations
CREATE TABLE gene_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hgnc_id INTEGER NOT NULL,
    paper_gene_symbol TEXT NOT NULL,     -- Symbol as the paper writes it, uppercased (may be a previous or alias symbol); diagnostic only, hgnc_id is the key
    paper_doi TEXT NOT NULL,
    source TEXT CHECK(source IN ('recent_evidence', 'expansion_evidence', 'relevance_assessment')) NOT NULL,

    FOREIGN KEY (paper_doi) REFERENCES papers(doi),
    UNIQUE(paper_doi, hgnc_id, source)
);

CREATE INDEX idx_gene_mentions_hgnc_id ON gene_mentions(hgnc_id);
CREATE INDEX idx_gene_mentions_paper ON gene_mentions(paper_gene_symbol);
CREATE INDEX idx_gene_mentions_paper_doi ON gene_mentions(paper_doi);
CREATE INDEX idx_gene_mentions_source_gene ON gene_mentions(source, hgnc_id);

-- Track completed tournament selection runs per gene (used for resumability)
CREATE TABLE tournament_results (
    hgnc_id INTEGER PRIMARY KEY,
    selected_dois_json JSON,
    tournament_raw_responses_json JSON  -- Includes LLM reasoning content
);

CREATE INDEX idx_papers_pmid ON papers(pmid) WHERE pmid IS NOT NULL;

-- Indexes for source tracking
CREATE INDEX idx_papers_source ON papers(source);
CREATE INDEX idx_papers_source_type ON papers(source_type);
CREATE INDEX idx_papers_source_details ON papers(source_details);
CREATE INDEX idx_papers_download_status ON papers(download_status);

-- Partial indices for efficiently finding unprocessed papers
-- assess_relevance.py: find papers needing relevance assessment
CREATE INDEX idx_papers_relevance_status ON papers(relevance_assessment_json)
    WHERE relevance_assessment_json IS NULL;

-- extract_evidence.py: find papers needing evidence extraction
CREATE INDEX idx_papers_evidence_status ON papers(evidence_extraction_json)
    WHERE evidence_extraction_json IS NULL;

-- Multiple files: find papers with completed evidence extraction
CREATE INDEX idx_papers_has_evidence ON papers(evidence_extraction_json)
    WHERE evidence_extraction_json IS NOT NULL;

-- One aggregation call per gene (assess-genes): what the model saw and its gene-level output.
CREATE TABLE gene_aggregations (
    hgnc_id INTEGER PRIMARY KEY,
    assessment_raw TEXT NOT NULL,          -- Message JSON as returned by the API
    paper_id_mapping JSON NOT NULL,        -- {AuthorYear: DOI} mapping used during assessment
    filtered_papers_json JSON,             -- [{doi, reason}] removed by the preprint gate; NULL when none
    -- PanelApp Australia's curation shown to the model:
    --   {"gencc_rows": [PanelApp Australia GenCC rows],
    --    "disputes": [Disputed/Refuted GenCC submissions, any submitter],
    --    "panel_entries": [entries on the target panels, or on all panels when all_panels],
    --    "all_panels": <bool, true for an Incidentalome-only gene>}
    panelapp_context_json JSON NOT NULL,
    -- Reviews on every target panel that held the gene at assess time, as fetched via
    -- PanelApp's evaluations endpoint: [{"panel_id": <int>, "evaluations": [<raw evaluation
    -- dicts>]}, ...] in target-panel order. NULL when the gene was on no target panel.
    existing_panel_reviews_json JSON,
    unassessed_reports_json JSON NOT NULL, -- [{phenotype, inheritance_mode, dois, reason}]
    quality_concerns_json JSON NOT NULL    -- [{concern, dois, citations: [{doi, quote}]}]
);

-- One row per gene-disease-MoI association of one gene in this run. Storing a gene's
-- aggregation deletes its old rows explicitly, in the same transaction. Deleting a
-- gene_aggregations row by hand cascades only where PRAGMA foreign_keys is on; the
-- stages and the report read associations through gene_aggregations, so rows left
-- behind are ignored until the gene is stored again.
CREATE TABLE associations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- per-run identity
    hgnc_id INTEGER NOT NULL REFERENCES gene_aggregations(hgnc_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,             -- order in the model output
    assessment_json JSON NOT NULL,         -- the association object, with dois and criteria as a list
    mondo_id TEXT,                         -- NULL until map-mondo has run for this row
    mondo_label TEXT,
    -- panelapp_gencc: reuses a PanelApp Australia GenCC row's disease (set by assess-genes);
    -- exact / broader: chosen by map-mondo
    mondo_match TEXT CHECK(mondo_match IN ('panelapp_gencc', 'exact', 'broader')),
    mondo_raw TEXT,                        -- map-mondo's final message; NULL for panelapp_gencc
    matched_panels_json JSON,              -- [{"panel_id", "rationale"}]
    matched_panels_raw TEXT,
    UNIQUE(hgnc_id, position),
    CHECK((mondo_id IS NULL) = (mondo_match IS NULL))
);

CREATE INDEX idx_associations_hgnc ON associations(hgnc_id);
CREATE INDEX idx_associations_unmapped ON associations(id) WHERE mondo_id IS NULL;
CREATE INDEX idx_associations_unmatched ON associations(id) WHERE matched_panels_json IS NULL;

-- Variant frequency information from gnomAD v4.1, one row per extracted variant.
-- Filled by extract-evidence from its lookup_variants tool results (variant-lookup
-- service). The JSON shapes mirror the service's response; see README.md
-- § "External services" and ARCHITECTURE.md in the variant-lookup repo.
CREATE TABLE variant_frequencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id TEXT NOT NULL,  -- Pseudo-VCF (chr-pos-ref-alt) on success; original variant text on normalization failure
    hgnc_id INTEGER NOT NULL,  -- For report generation and indexing
    paper_doi TEXT NOT NULL,
    quote TEXT NOT NULL,  -- The extraction's quote for this variant; see citation_locations
    -- Success: {hgvs_c, hgvs_p, original_text, total_normalizations, selected_for_max_ac}.
    -- Service-side failure (no normalized variant returned):
    --   {original_text, error_code, error_message, upstream}.
    normalization JSON NOT NULL,
    -- Success (variant found in gnomAD), flat per the service's Frequency model:
    --   {ac, an, homozygote_count, heterozygote_count, hemizygote_count,
    --    faf95_popmax, faf95_popmax_population}.
    -- Pseudo-VCF resolved but not in gnomAD: {variant_not_found: true}.
    -- Pre-gnomAD failure (no lookup happened):  {normalization_error: true}.
    gnomad JSON NOT NULL,

    FOREIGN KEY (paper_doi) REFERENCES papers(doi),
    UNIQUE(variant_id, paper_doi, hgnc_id)
);

CREATE INDEX idx_variant_frequencies_variant_id ON variant_frequencies(variant_id);
CREATE INDEX idx_variant_frequencies_hgnc_id ON variant_frequencies(hgnc_id);
CREATE INDEX idx_variant_frequencies_paper_doi ON variant_frequencies(paper_doi);

-- Claude API bookkeeping (see src/palit/llm.py).
-- One row per submitted Message Batch. A batch is collected once every request
-- in it has been recorded; stages re-attach to uncollected batches on start.
CREATE TABLE llm_batches (
    batch_id TEXT PRIMARY KEY,            -- msgbatch_...
    stage TEXT NOT NULL,                  -- e.g. 'relevance', 'extraction', 'assess_genes'
    round INTEGER NOT NULL,               -- 1 for single-shot stages
    model TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    ended_at TEXT,                        -- processing_status reached 'ended'
    collected_at TEXT                     -- every request recorded
);

CREATE INDEX idx_llm_batches_uncollected ON llm_batches(stage) WHERE collected_at IS NULL;

-- One row per Messages request, batched or immediate. Also the per-stage usage
-- record: the Console usage report cannot split palit's traffic by stage.
CREATE TABLE llm_requests (
    custom_id TEXT PRIMARY KEY,           -- '<stage>-<round>-<random>'
    batch_id TEXT REFERENCES llm_batches(batch_id),  -- NULL for immediate requests
    stage TEXT NOT NULL,
    subject TEXT NOT NULL,                -- DOI, HGNC ID or association id as text
    round INTEGER NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'succeeded', 'refused', 'errored', 'expired', 'canceled')),
    stop_reason TEXT,
    refusal_category TEXT,                -- stop_details.category of a refusal, e.g. 'bio'
    error_type TEXT,                      -- e.g. 'invalid_request_error', 'overloaded_error'
    service_tier TEXT,                    -- 'batch' or 'standard'
    input_tokens INTEGER,                 -- uncached input
    cache_write_5m_tokens INTEGER,
    cache_write_1h_tokens INTEGER,
    cache_read_tokens INTEGER,
    output_tokens INTEGER,                -- includes thinking
    -- Why the stage rejected the answer and will ask again, e.g. 'schema violation
    -- at $.gene_evaluations[0]: ...'; NULL for an answer the stage used
    rejection TEXT,
    completed_at TEXT
);

CREATE INDEX idx_llm_requests_batch ON llm_requests(batch_id);
CREATE INDEX idx_llm_requests_stage_subject ON llm_requests(stage, subject);

-- Input messages of a multi-round conversation, stored byte-for-byte: Opus 5.5
-- rejects a replayed thinking block whose preceding history was edited.
CREATE TABLE llm_conversations (
    stage TEXT NOT NULL,
    subject TEXT NOT NULL,
    round INTEGER NOT NULL,               -- the round these messages are the input to
    messages_json TEXT NOT NULL,
    PRIMARY KEY (stage, subject, round)
);

-- Files API uploads, so each PDF is uploaded once per content version.
CREATE TABLE uploaded_files (
    doi TEXT PRIMARY KEY REFERENCES papers(doi),
    file_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,                 -- of the upload: the local PDF, or its leading pages
    uploaded_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

-- How many leading pages of a long PDF fit its paper's round-1 extraction
-- request, from token counts, once per PDF version and version of the
-- counting rules. The upload holds only those pages when they are fewer than
-- the PDF's.
CREATE TABLE pdf_page_limits (
    doi TEXT PRIMARY KEY REFERENCES papers(doi),
    sha256 TEXT NOT NULL,                 -- of the local PDF that was counted
    rules_version INTEGER NOT NULL,       -- extract_evidence.PAGE_LIMIT_RULES_VERSION
    total_pages INTEGER NOT NULL,
    pages INTEGER NOT NULL,               -- leading pages that fit; 0 if not even the first
    input_tokens INTEGER NOT NULL,        -- of the round-1 request with those pages,
                                          -- a page too large to count estimated
    counted_at TEXT NOT NULL
);

-- Where each extraction quote sits in its paper's PDF, resolved by anchorite at
-- extraction time. The report's PDF viewer draws these boxes.
CREATE TABLE citation_locations (
    paper_doi TEXT NOT NULL REFERENCES papers(doi),
    quote TEXT NOT NULL,
    -- [{"page": 1-based, "top", "left", "bottom", "right"}], 0-1000 page
    -- coordinates, one box per visual line; [] when the quote can't be placed
    bboxes_json JSON NOT NULL,
    PRIMARY KEY (paper_doi, quote)
);

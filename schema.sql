-- Gene-Panel Centric Literature Assessment Database Schema
-- Creates fresh database with complete schema for gene-panel workflow

-- Enable Write-Ahead Logging (WAL) mode for concurrent access
-- WAL allows multiple processes to read/write simultaneously without lock contention.
-- Enables parallel GPU processing with assess_relevance.py using --shard-index.
-- This setting persists in the database file.
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

    -- Main assessment fields (arrays of 3 results for majority voting)
    relevance_assessment_raw JSON,  -- Array of 3 raw text responses, including LLM reasoning content
    relevance_assessment_json JSON,  -- Array of 3 parsed objects
    evidence_extraction_raw TEXT,  -- Includes LLM reasoning content
    evidence_extraction_json JSON,

    -- PDF citation linking
    bbox_mapping JSON
);

-- Normalized gene-paper relationships (automatically maintained from evidence extraction)
-- Tracks which papers mention which genes with patient/disease evidence
-- Query relevance_assessment_json.associations or evidence_extraction_json.disease_entities for actual disease associations
CREATE TABLE gene_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hgnc_id INTEGER NOT NULL,
    paper_gene_symbol TEXT NOT NULL,     -- Original symbol mentioned in paper (may be alias)
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

-- Single assessment per gene with matched panels
CREATE TABLE gene_assessments (
    hgnc_id INTEGER PRIMARY KEY,
    assessment_raw TEXT,   -- Raw LLM response including reasoning
    assessment_json JSON,  -- Contains full aggregate assessment
    paper_id_mapping JSON NOT NULL,  -- {AuthorYear: DOI} mapping used during assessment
    filtered_papers_json JSON,  -- [{doi, reason}] papers excluded from assessment (e.g. preprint family gate)
    matched_panels_raw TEXT,  -- Raw LLM response for panel matching including reasoning
    matched_panels_json JSON,    -- Array: [{"panel_id": 137, "rationale": "..."}, ...]
    -- Reviews on the single target panel returned by find_gene_panel at assess time,
    -- as fetched via PanelApp's evaluations endpoint. Shape:
    --   {"panel_id": <int>, "evaluations": [<raw evaluation dicts>]}
    -- NULL when the gene was not on any target panel at assess time.
    existing_panel_reviews_json JSON
);

-- match_panels.py: find genes needing panel matching
CREATE INDEX idx_gene_assessments_unmatched ON gene_assessments(matched_panels_json)
    WHERE matched_panels_json IS NULL;

-- Variant frequency information from gnomAD.
-- Populated by `palit fetch-variant-frequencies` from the variant-lookup
-- service. The JSON shapes mirror the service's response — see
-- README.md § "External services" and ARCHITECTURE.md in the
-- variant-lookup repo for the full contract.
CREATE TABLE variant_frequencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id TEXT NOT NULL,  -- Pseudo-VCF (chr-pos-ref-alt) on success; original variant text on normalization failure
    hgnc_id INTEGER NOT NULL,  -- For report generation and indexing
    paper_doi TEXT NOT NULL,
    box_id INTEGER NOT NULL,  -- For PDF citation linking
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
    UNIQUE(variant_id, paper_doi, box_id)  -- Allow same variant in paper if different box_id
);

CREATE INDEX idx_variant_frequencies_variant_id ON variant_frequencies(variant_id);
CREATE INDEX idx_variant_frequencies_hgnc_id ON variant_frequencies(hgnc_id);
CREATE INDEX idx_variant_frequencies_paper_doi ON variant_frequencies(paper_doi);

-- Fixed gene-disease associations ("entities") the pipeline assesses against.
-- Seeded from an external curation source by `palit seed-entities` and read-only
-- thereafter: no pipeline stage creates, edits or deletes rows here.
CREATE TABLE gene_disease_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hgnc_id INTEGER NOT NULL,
    mondo_id TEXT NOT NULL,
    disease_title TEXT NOT NULL,

    -- Same inheritance vocabulary as evidence extraction, plus the combined
    -- value understood by panelapp_integration.decompose_moi.
    moi TEXT NOT NULL CHECK(moi IN ('Monoallelic', 'Biallelic', 'Monoallelic_and_biallelic', 'X-linked', 'Mitochondrial', 'Other')),

    -- Source inheritance labels behind `moi`, in source row order ("; "-joined).
    -- Distinct source labels that map to the same enum for one (gene, MONDO) —
    -- e.g. "X-linked" and "X-linked recessive" — are deduplicated into a single
    -- entity carrying both titles, and must agree on their classification.
    gencc_moi_titles TEXT NOT NULL,

    -- Report display only; never fed to a prompt, so assessments stay independent
    -- of the source's own verdict.
    gencc_classification TEXT NOT NULL,

    source TEXT NOT NULL,  -- Provenance, e.g. 'gencc:PanelApp Australia'

    -- An entity is a (gene, disease, inheritance mode) triple: the same gene and
    -- MONDO term curated as both dominant and recessive is two entities, each
    -- rated on its own evidence.
    UNIQUE(hgnc_id, mondo_id, moi)
);

CREATE INDEX idx_gene_disease_entities_hgnc ON gene_disease_entities(hgnc_id);

-- Which papers carry evidence for which entity
CREATE TABLE entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,
    paper_doi TEXT NOT NULL,

    FOREIGN KEY (entity_id) REFERENCES gene_disease_entities(id),
    FOREIGN KEY (paper_doi) REFERENCES papers(doi),
    UNIQUE(entity_id, paper_doi)
);

CREATE INDEX idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX idx_entity_mentions_paper ON entity_mentions(paper_doi);

-- Single aggregate assessment per gene-disease entity
CREATE TABLE gene_disease_assessments (
    entity_id INTEGER PRIMARY KEY,
    hgnc_id INTEGER NOT NULL,  -- For report generation and indexing
    assessment_raw TEXT NOT NULL,   -- Raw LLM response including reasoning
    assessment_json JSON NOT NULL,  -- Contains full aggregate assessment
    paper_id_mapping JSON NOT NULL,  -- {AuthorYear: DOI} mapping used during assessment
    filtered_papers_json JSON,  -- [{doi, reason}] papers excluded from assessment (e.g. preprint family gate)

    FOREIGN KEY (entity_id) REFERENCES gene_disease_entities(id)
);

CREATE INDEX idx_gene_disease_assessments_hgnc ON gene_disease_assessments(hgnc_id);

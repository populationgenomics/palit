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
    matched_panels_raw TEXT,  -- Raw LLM response for panel matching including reasoning
    matched_panels_json JSON    -- Array: [{"panel_id": 137, "rationale": "..."}, ...]
);

-- match_panels.py: find genes needing panel matching
CREATE INDEX idx_gene_assessments_unmatched ON gene_assessments(matched_panels_json)
    WHERE matched_panels_json IS NULL;

-- Variant frequency information from gnomAD
CREATE TABLE variant_frequencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing primary key
    variant_id TEXT NOT NULL,  -- gnomAD pseudo-VCF style (chr-pos-ref-alt)
    hgnc_id INTEGER NOT NULL,  -- For report generation and indexing
    paper_doi TEXT NOT NULL,  -- Paper where variant was mentioned
    box_id INTEGER NOT NULL,  -- For PDF citation linking
    normalization JSON NOT NULL,  -- HGVS c/p from variant normalizer
    gnomad JSON NOT NULL,  -- Raw gnomAD API response or error message

    FOREIGN KEY (paper_doi) REFERENCES papers(doi),
    UNIQUE(variant_id, paper_doi, box_id)  -- Allow same variant in paper if different box_id
);

CREATE INDEX idx_variant_frequencies_variant_id ON variant_frequencies(variant_id);
CREATE INDEX idx_variant_frequencies_hgnc_id ON variant_frequencies(hgnc_id);
CREATE INDEX idx_variant_frequencies_paper_doi ON variant_frequencies(paper_doi);

# PanelApp Australia Literature Assessment

LLM-based literature assessment system for rare disease gene curation. Automatically screens papers for relevance, extracts evidence against PanelApp Australia diagnostic criteria, and generates comprehensive gene-centric reports with panel recommendations.

## Setup

### Installation

```bash
# Basic installation (data ingestion, reporting, variant analysis)
uv sync

# With ML dependencies (required for LLM inference and screening classifier)
uv sync --extra ml

# With macOS-specific Docling acceleration (optional)
uv sync --extra docling-macos
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

- **Main workflow** (`data/db.sqlite`): Created from `schema.sql` by `palit ingest-preprints` or `palit ingest-pubmed`
- **Screening workflow** (`data/pubmed_baseline_screening.sqlite`): Created from `schema.sql` by `palit screen-pubmed`
- **Classifier training** (`data/screening_classifier/training.sqlite`): Created from `src/palit/screening_classifier/training.sql` (only needed for training)

Both main and screening workflows use the same schema for consistency, allowing the same tools (e.g., `assess-relevance`) to work on both databases.

## Complete Workflow

```bash
# Configuration
PANEL_DATE=2025-10-01
END_DATE=2025-10-15

# 1. Setup: Create database and ingest papers (creates DB from schema.sql if needed)
#    --previous-db widens the date range into the previous run's window and skips
#    papers already ingested (buffer window for API flakiness resilience).
#    Preprints first: ensures preprint metadata (version) is preserved for automatic
#    PDF download. PubMed backfills PMIDs into preprint rows without overwriting.
uv run palit ingest-preprints --previous-db data/db_prev.sqlite $BUFFER_START $END_DATE
uv run palit ingest-pubmed --previous-db data/db_prev.sqlite $BUFFER_START $END_DATE

# 2. Assess relevance of papers
uv run palit assess-relevance --panel-date $PANEL_DATE

# 2a. (Optional) Parallel assessment across multiple GPUs
for i in 0 1; do sbatch -p GPU-H100 --gpus=1 -t 24:00:00 -J "assess-relevance-shard-$i" -o "assess_relevance_shard_$i.log" --wrap="uv run palit assess-relevance --panel-date $PANEL_DATE --db-path data/pubmed_baseline_screening.sqlite --prompt-path prompts/retrospective_screening_prompt.txt --shard-index $i --num-shards 2"; done

# 3. Download full-text papers (automated PMC + preprints, manual fallback)
uv run palit download-papers attempt-pmc
uv run palit download-papers download-preprints
uv run palit download-papers open-browser
# ... manually download PDFs to data/papers/ ...
uv run palit docling convert
uv run palit download-papers register

# 4. Extract evidence from full-text papers
uv run palit extract-evidence --panel-date $PANEL_DATE

# 5. Discover papers referenced in evidence (citation-based expansion)
uv run palit discover-citations discover

# 5a. Optionally add papers manually that weren't found automatically
uv run palit discover-citations add --gene GENE_SYMBOL PMID1 PMID2 ...

# 6. Expand literature beyond citations
uv run palit expand-literature --cutoff-date $PANEL_DATE

# 7. Download expansion papers (same workflow as step 3)
uv run palit download-papers attempt-pmc
uv run palit download-papers download-preprints
uv run palit download-papers open-browser --expansion-only
# ... manually download PDFs to data/papers/ ...
uv run palit docling convert
uv run palit download-papers register

# 8. Extract evidence from expansion papers
uv run palit extract-evidence --panel-date $PANEL_DATE

# 9. Aggregate evidence across papers per gene (panel-agnostic)
uv run palit assess-genes --panel-date $PANEL_DATE

# 10. Match genes to appropriate panels based on phenotype descriptions
uv run palit match-panels --panel-date $PANEL_DATE

# 11. Look up variant frequencies from gnomAD
uv run palit fetch-variant-frequencies

# 12. Create annotated PDFs with highlighted citations
uv run palit annotate-pdfs

# 13. Generate assessment report package with panel recommendations
uv run palit generate-report --report-id report_mendeliome --panel-date $PANEL_DATE
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
  --panel-date $PANEL_DATE \
  --prompt-path prompts/retrospective_screening_prompt.txt
```

**Key difference from standard relevance assessment:**

- **Standard prompt** (`relevance_assessment_prompt.txt`): Asks "Is this NEW evidence for diagnostic panels?" - optimized for recent literature
- **Retrospective prompt** (`retrospective_screening_prompt.txt`): Asks "Does this provide SUBSTANTIAL evidence for gene-disease relationships?" - optimized for historical baseline screening

The retrospective prompt evaluates papers in their historical context, accepting important early descriptions of gene-disease associations even if those genes are now well-established. This ensures comprehensive coverage across 25 years of literature for downstream tournament selection and analysis.

### Updating the Baseline

After each fortnightly processing run completes, feed majority-relevant papers back into the baseline screening DB so it grows as a comprehensive repository:

```bash
FORTNIGHTLY_DB=data/db_2026_february_h1.sqlite

sqlite3 data/pubmed_baseline_screening.sqlite <<'SQL'
ATTACH '$FORTNIGHTLY_DB' AS source;

CREATE TEMP TABLE relevant_dois AS
SELECT doi FROM source.papers
WHERE relevance_assessment_json IS NOT NULL
  AND (json_extract(relevance_assessment_json, '$[0].relevant')
     + json_extract(relevance_assessment_json, '$[1].relevant')
     + json_extract(relevance_assessment_json, '$[2].relevant')) >= 2;

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
# Configuration
PANEL_DATE=2025-10-01
PANEL_ID=47  # Arthrogryposis panel ID
PANEL_NAME=arthrogryposis

# 1. Copy pre-filtered baseline DB papers to new database
sqlite3 data/$PANEL_NAME.sqlite < schema.sql
sqlite3 data/$PANEL_NAME.sqlite "ATTACH 'data/pubmed_baseline_screening.sqlite' AS source; INSERT INTO papers (pmid, title, abstract, authors, journal, entrez_date, source_type, source_details) SELECT pmid, title, abstract, authors, journal, entrez_date, 'initial', source_details FROM source.papers"

# 2. Assess relevance scoped to the panel
uv run palit assess-relevance \
  --db-path data/$PANEL_NAME.sqlite \
  --panel-date $PANEL_DATE \
  --scope-panel-id $PANEL_ID \
  --prompt-path prompts/panel_relevance_assessment_prompt.txt

# 3. (Optional) Reduce literature for well-researched genes
# The aggregation step has a practical limit of ~30-40 papers per gene due to
# context window constraints. For panels with well-researched genes (e.g., POLG
# with 200+ papers), use tournament selection to keep only the most informative:
uv run palit reduce-literature --db-path data/$PANEL_NAME.sqlite

# 4. Download full-text papers (now reduced set if step 3 was run)
uv run palit download-papers attempt-pmc --db-path data/$PANEL_NAME.sqlite
uv run palit download-papers download-preprints --db-path data/$PANEL_NAME.sqlite
uv run palit download-papers open-browser --db-path data/$PANEL_NAME.sqlite
# ... manually download PDFs ...
uv run palit docling convert --db-path data/$PANEL_NAME.sqlite
uv run palit download-papers register --db-path data/$PANEL_NAME.sqlite

# 5. Extract evidence and assess genes (panel-scoped)
uv run palit extract-evidence --db-path data/$PANEL_NAME.sqlite --panel-date $PANEL_DATE --scope-panel-id $PANEL_ID
uv run palit assess-genes --db-path data/$PANEL_NAME.sqlite --panel-date $PANEL_DATE --target-panel-ids $PANEL_ID --scope-panel-id $PANEL_ID

# 6. Generate report package with panel-scoped novelty detection
uv run palit annotate-pdfs --db-path data/$PANEL_NAME.sqlite --output-dir data/annotated_$PANEL_NAME
uv run palit generate-report \
  --report-id panel_$PANEL_NAME \
  --db-path data/$PANEL_NAME.sqlite \
  --panel-date $PANEL_DATE \
  --target-panel-ids $PANEL_ID \
  --annotated-dir data/annotated_$PANEL_NAME
```

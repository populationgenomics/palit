-- Minimal schema for relevance screening classifier training
-- Stores PubMed articles (PMID, title, abstract) and labels for training data

CREATE TABLE papers (
    pmid INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    is_relevant BOOLEAN NOT NULL,  -- True for positive examples, False for negatives
    split TEXT CHECK(split IN ('train', 'val', 'test'))  -- Assigned during data prep
);

CREATE INDEX idx_papers_pmid ON papers(pmid);
CREATE INDEX idx_papers_label ON papers(is_relevant);
CREATE INDEX idx_papers_split ON papers(split);

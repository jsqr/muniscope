-- Reset tables for municipal code analysis

-- Run this separately as 'postgres' superuser:
-- CREATE EXTENSION IF NOT EXISTS vector;
-- And if needed:
-- GRANT CREATE ON SCHEMA public TO <username>;

DROP TABLE IF EXISTS associations CASCADE;
DROP TYPE IF EXISTS assoc_type CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_chunks;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS segments;
DROP TABLE IF EXISTS codes;

-- Track associations among provisions

CREATE TYPE assoc_type as ENUM ('definition', 'interpretation', 'limitation', 'penalty');

CREATE TABLE associations (
       assoc_id SERIAL PRIMARY KEY,
       code_id INTEGER,
       assoc_type assoc_type,
       left_id INTEGER, -- ref to primary provision (e.g., an obligation or prohibition)
       right_id INTEGER, -- ref to associated provision (e.g., list of applicable defs)
       CONSTRAINT unique_association UNIQUE (code_id, assoc_type, left_id, right_id)
);

-- Segments may be longer than the optimal chunk size for an embedding
-- (say, 500 or 1000 words), so we break them up into chunks for vector search
CREATE TABLE chunks (
       chunk_id SERIAL PRIMARY KEY,
       segment_id INTEGER, -- foreign key to citable segment
       -- code_id INTEGER, -- foreign key to code
       chunk_idx INTEGER, -- index of chunk within segment
       content TEXT,
       enhanced_content TEXT, -- Additional context (e.g., headings)
       embedding VECTOR(1536), -- FIXME: see above (1536 for text-embedding-3-small)
       CONSTRAINT unique_chunk UNIQUE (segment_id, chunk_idx)
);

-- 'Segments' here are the smallest citable divisions within the organizational
-- scheme of a code. We keep track of all the headings
CREATE TABLE segments (
       segment_id SERIAL PRIMARY KEY,
       code_id INTEGER, -- foreign key to code
       segment_level INTEGER, -- 1-5, with 1 being the highest level
       H1_enumeration TEXT, H1_text TEXT,
       H2_enumeration TEXT, H2_text TEXT,
       H3_enumeration TEXT, H3_text TEXT,
       H4_enumeration TEXT, H4_text TEXT,
       H5_enumeration TEXT, H5_text TEXT, -- lower levels may be unused in some codes
       content TEXT,
       search_vector tsvector, -- for full-text search
       CONSTRAINT unique_segment UNIQUE (code_id, H1_enumeration, H2_enumeration, H3_enumeration, H4_enumeration, H5_enumeration)
);

-- A law, code, or ordinance
CREATE TABLE codes (
       code_id SERIAL PRIMARY KEY,
       title TEXT, -- e.g., 'Municipal Code of Chicago'
       jurisdiction TEXT UNIQUE, -- e.g., 'Chicago' (there may be multiple codes per jurisdiction)
       H1_name TEXT,    H1_pattern TEXT, -- e.g., 'TITLE', '^TITLE\s+[IVXLCDM]+$'
       H2_name TEXT,    H2_pattern TEXT,
       H3_name TEXT,    H3_pattern TEXT,
       H4_name TEXT,    H4_pattern TEXT,
       H5_name TEXT,    H5_pattern TEXT
);

-- Create a materialized view to optimize querying chunks with associated metadata
CREATE MATERIALIZED VIEW mv_chunks AS
SELECT 
    chunks.chunk_id,
    chunks.segment_id,
    chunks.chunk_idx,
    chunks.content,
    chunks.enhanced_content,
    chunks.embedding,
    segments.H1_enumeration,       segments.H1_text,
    segments.H2_enumeration,       segments.H2_text,
    segments.H3_enumeration,       segments.H3_text,
    segments.H4_enumeration,       segments.H4_text,
    segments.H5_enumeration,       segments.H5_text,
    codes.code_id,
    codes.jurisdiction,
    codes.H1_name,
    codes.H2_name,
    codes.H3_name,
    codes.H4_name,
    codes.H5_name
FROM 
    chunks
JOIN 
    segments ON chunks.segment_id = segments.segment_id
JOIN 
    codes ON segments.code_id = codes.code_id;

REFRESH MATERIALIZED VIEW mv_chunks;

-------------------------------------------------------------------
-- Full-text search using the search_vector column in segments   --
-- and the enhanced_content column in chunks                     --
-------------------------------------------------------------------

CREATE OR REPLACE FUNCTION segments_search_vector_update() RETURNS trigger AS $$
BEGIN
  NEW.search_vector :=
    to_tsvector('english', COALESCE(NEW.H1_text, '')) ||
    to_tsvector('english', COALESCE(NEW.H2_text, '')) ||
    to_tsvector('english', COALESCE(NEW.H3_text, '')) ||
    to_tsvector('english', COALESCE(NEW.H4_text, '')) ||
    to_tsvector('english', COALESCE(NEW.H5_text, '')) ||
    to_tsvector('english', COALESCE(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Update search_vector when relevant data in segments changes
CREATE TRIGGER segments_search_vector_update
BEFORE INSERT OR UPDATE ON segments
FOR EACH ROW EXECUTE FUNCTION segments_search_vector_update();

-- Create a generalized inverted index
CREATE INDEX idx_segments_search_vector ON segments USING GIN (search_vector);

-- Do the same for the 'enhanced_content' field in chunks and the mv_chunks view
CREATE INDEX idx_chunks_search_vector ON chunks USING GIN (to_tsvector('english', enhanced_content));
CREATE INDEX idx_mv_chunks_enhanced_content ON mv_chunks USING GIN (to_tsvector('english', enhanced_content));

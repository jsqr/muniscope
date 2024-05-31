-- Reset tables for municipal code analysis

-- Run this separately as 'postgres' superuser:
-- CREATE EXTENSION IF NOT EXISTS vector;
-- And if needed:
-- GRANT CREATE ON SCHEMA public TO <username>;

DROP TABLE IF EXISTS associations CASCADE;
DROP TYPE IF EXISTS assoc_type CASCADE;
DROP TABLE IF EXISTS segments;
DROP TABLE IF EXISTS provisions;
DROP TABLE IF EXISTS codes;

-- Track associations among provisions

CREATE TYPE assoc_type as ENUM ('definition', 'interpretation', 'limitation', 'penalty');

CREATE TABLE associations (
       assoc_id SERIAL PRIMARY KEY,
       code_id INTEGER,
       assoc_type assoc_type,
       left_id INTEGER, -- ref to primary provision (e.g., an obligation or prohibition)
       right_id INTEGER -- ref to associated provision (e.g., list of applicable defs)
);

-- Provisions may be longer than the optimal chunk size for an embedding
-- (say, 500 tokens), so we break them up into segments for vector search
CREATE TABLE segments (
       segment_id SERIAL PRIMARY KEY,
       provision_id INTEGER, -- foreign key to citable provision
       begin_idx INTEGER, -- index of start of segment within provision text
       end_idx INTEGER, -- index of end of segment
       content VARCHAR(500), -- FIXME: replace with dynamic sql to allow tuning
       embedding VECTOR(1536) -- FIXME: see above (1536 for text-embedding-3-small)
);

-- 'Provisions' here are the smallest citable divisions within the organizational
-- scheme of a code. We keep track of all the headings
CREATE TABLE provisions (
       provision_id SERIAL PRIMARY KEY,
       code_id INTEGER, -- foreign key to code
       H1 TEXT,
       H2 TEXT,
       H3 TEXT,
       H4 TEXT,
       H5 TEXT, -- lower levels may be unused in some codes
       content TEXT,
       search_vector tsvector -- for full-text search
);

-- A law, code, or ordinance
CREATE TABLE codes (
       code_id SERIAL PRIMARY KEY,
       title TEXT, -- e.g., 'Municipal Code of Chicago'
       jurisdiction TEXT, -- e.g., 'Chicago' (there may be multiple codes per jurisdiction)
       H1_name TEXT,    H1_pattern TEXT, -- e.g., 'TITLE', '^TITLE\s+[IVXLCDM]+$'
       H2_name TEXT,    H2_pattern TEXT,
       H3_name TEXT,    H3_pattern TEXT,
       H4_name TEXT,    H4_pattern TEXT,
       H5_name TEXT,    H5_pattern TEXT
);

-------------------------------------------------------------------
-- Full-text search using the search_vector column in provisions --
-------------------------------------------------------------------

CREATE OR REPLACE FUNCTION provisions_search_vector_update() RETURNS trigger AS $$
BEGIN
  NEW.search_vector :=
    to_tsvector('english', COALESCE(NEW.H1, '')) ||
    to_tsvector('english', COALESCE(NEW.H2, '')) ||
    to_tsvector('english', COALESCE(NEW.H3, '')) ||
    to_tsvector('english', COALESCE(NEW.H4, '')) ||
    to_tsvector('english', COALESCE(NEW.H5, '')) ||
    to_tsvector('english', COALESCE(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Update search_vector when relevant data in provisions changes
CREATE TRIGGER provisions_search_vector_update
BEFORE INSERT OR UPDATE ON provisions
FOR EACH ROW EXECUTE FUNCTION provisions_search_vector_update();

-- Create a generalized inverted index
CREATE INDEX idx_provisions_search_vector ON provisions USING GIN (search_vector);

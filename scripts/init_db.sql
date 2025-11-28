-- Enable the pgvector extension to handle vector data types
-- This must be run by a superuser on the database first.
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the table if it already exists to ensure a clean setup
DROP TABLE IF EXISTS job_listings;

-- Create the main table for storing job listings
CREATE TABLE job_listings (
    id SERIAL PRIMARY KEY,                          -- Unique identifier for each listing
    job_title VARCHAR(255) NOT NULL,                -- The title of the job
    company VARCHAR(255),                           -- The name of the company
    location VARCHAR(255),                          -- Job location (e.g., "Remote", "New York, NY")
    description TEXT NOT NULL,                      -- The full job description text
    apply_url TEXT,                                 -- URL to the application page
    description_hash VARCHAR(64) UNIQUE NOT NULL,   -- SHA-256 hash for exact deduplication
    source_url TEXT NOT NULL,                       -- URL where the job listing was found
    metadata JSONB,                                 -- Flexible field for extra data (e.g., extraction method)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()   -- Timestamp when the record was created
);

-- Create an index on the hash for fast duplicate lookups
CREATE INDEX idx_description_hash ON job_listings(description_hash);

-- Add comments for documentation
COMMENT ON TABLE job_listings IS 'Stores processed job listings from various sources.';
COMMENT ON COLUMN job_listings.description_hash IS 'SHA-256 hash of the normalized job description for deduplication.';
COMMENT ON COLUMN job_listings.metadata IS 'Stores metadata like extraction method, confidence scores, etc.';

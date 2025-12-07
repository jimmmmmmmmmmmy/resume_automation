-- Migration 003: Add pgvector embeddings for semantic search
-- Phase 3: Resume-Job Matching & Optimization Pipeline

-- =============================================================================
-- Step 1: Enable pgvector extension
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Step 2: Add embedding column to job_listings
-- =============================================================================
ALTER TABLE job_listings
ADD COLUMN IF NOT EXISTS description_embedding vector(384);

-- =============================================================================
-- Step 3: Add summary embedding to resumes
-- =============================================================================
ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS summary_embedding vector(384);

-- =============================================================================
-- Step 4: Create table for section-level embeddings (experience, projects)
-- This allows efficient similarity searches via pgvector
-- =============================================================================
CREATE TABLE IF NOT EXISTS resume_section_embeddings (
    id SERIAL PRIMARY KEY,
    resume_id INT NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    section_type VARCHAR(50) NOT NULL,  -- 'experience', 'project', 'skills'
    section_index INT NOT NULL DEFAULT 0,  -- Index within section (0 for skills)
    content_hash VARCHAR(32) NOT NULL,  -- MD5 hash for cache invalidation
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(resume_id, section_type, section_index)
);

-- =============================================================================
-- Step 5: Create HNSW indexes for fast similarity search
-- HNSW parameters: m=16 (connections), ef_construction=64 (build quality)
-- Using vector_cosine_ops since our embeddings are normalized
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_job_embedding_hnsw
ON job_listings USING hnsw (description_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_resume_summary_embedding_hnsw
ON resumes USING hnsw (summary_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_section_embedding_hnsw
ON resume_section_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Step 6: Index for fast lookups by resume
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_section_by_resume
ON resume_section_embeddings(resume_id, section_type);

-- =============================================================================
-- Step 7: Create table for caching LLM optimization responses
-- =============================================================================
CREATE TABLE IF NOT EXISTS llm_response_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(64) NOT NULL UNIQUE,  -- MD5(section_content + job_desc)
    response_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for cache cleanup (expire old entries)
CREATE INDEX IF NOT EXISTS idx_cache_created ON llm_response_cache(created_at);

-- =============================================================================
-- Verification
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Migration 003 complete!';
    RAISE NOTICE 'Added: description_embedding to job_listings';
    RAISE NOTICE 'Added: summary_embedding to resumes';
    RAISE NOTICE 'Created: resume_section_embeddings table';
    RAISE NOTICE 'Created: llm_response_cache table';
    RAISE NOTICE 'Created: HNSW indexes for similarity search';
END $$;

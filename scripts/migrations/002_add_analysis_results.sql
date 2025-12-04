-- Migration 002: Add analysis_results table
-- Phase 2: Resume-job comparison and analysis storage

CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    resume_id INTEGER REFERENCES resumes(id) ON DELETE CASCADE,
    job_listing_id INTEGER REFERENCES job_listings(id) ON DELETE CASCADE,
    match_score DECIMAL(5,2),
    matching_skills JSONB,
    missing_skills JSONB,
    keyword_suggestions JSONB,
    improvement_suggestions JSONB,
    analysis_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Prevent duplicate analyses for the same resume-job pair
    UNIQUE(resume_id, job_listing_id)
);

-- Index for finding all analyses for a specific resume
CREATE INDEX IF NOT EXISTS idx_analysis_resume ON analysis_results(resume_id);

-- Index for finding all analyses for a specific job
CREATE INDEX IF NOT EXISTS idx_analysis_job ON analysis_results(job_listing_id);

-- Index for finding high-match analyses
CREATE INDEX IF NOT EXISTS idx_analysis_score ON analysis_results(match_score DESC);

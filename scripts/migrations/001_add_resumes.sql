-- Migration 001: Add resumes table
-- Phase 2: Resume parsing and storage

CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    location VARCHAR(255),
    summary TEXT,
    skills JSONB,
    experience JSONB,
    education JSONB,
    projects JSONB,
    raw_text TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for deduplication lookups
CREATE INDEX IF NOT EXISTS idx_resume_file_hash ON resumes(file_hash);

-- GIN index for skills JSONB queries (e.g., finding resumes with specific skills)
CREATE INDEX IF NOT EXISTS idx_resume_skills ON resumes USING GIN(skills);

-- Index for email lookups
CREATE INDEX IF NOT EXISTS idx_resume_email ON resumes(email);

-- Trigger to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_resume_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_resume_updated_at ON resumes;
CREATE TRIGGER trigger_resume_updated_at
    BEFORE UPDATE ON resumes
    FOR EACH ROW
    EXECUTE FUNCTION update_resume_updated_at();

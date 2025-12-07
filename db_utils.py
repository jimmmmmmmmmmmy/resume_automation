"""
Database Utility Module

This module provides a centralized interface for all database operations,
including connection management, deduplication checks, and data persistence.
"""
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from dotenv import load_dotenv

# Import shared data structures
from models import JobListing, Resume, AnalysisResult
import json

# Load environment variables
load_dotenv()

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("Warning: psycopg2 not available. Database functions will not work.")


def reload_env():
    """Reload environment variables from .env file."""
    load_dotenv(override=True)


def _get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables."""
    # Always reload to get latest values
    load_dotenv(override=True)
    return {
        'dbname': os.getenv('DB_NAME', 'resume_optimizer'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.

    Ensures connections are properly closed even if errors occur.

    Yields:
        A psycopg2 connection object.

    Raises:
        RuntimeError: If psycopg2 is not available.
        psycopg2.OperationalError: If connection fails.
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 is not installed. Please run: pip install psycopg2-binary")

    conn = None
    try:
        conn = psycopg2.connect(**_get_db_config())
        yield conn
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to the database. {e}")
        raise
    finally:
        if conn is not None:
            conn.close()


def check_for_duplicate(description_hash: str) -> bool:
    """
    Check if a job listing with the given hash already exists in the database.

    Args:
        description_hash: The SHA-256 hash of the normalized job description.

    Returns:
        True if a duplicate exists, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = "SELECT 1 FROM job_listings WHERE description_hash = %s;"
                cur.execute(sql, (description_hash,))
                return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking for duplicate: {e}")
        return False


def insert_job_listing(job: JobListing, job_hash: str) -> bool:
    """
    Insert a new job listing record into the database.

    Args:
        job: The JobListing object containing the job data.
        job_hash: The SHA-256 hash of the job description.

    Returns:
        True if insertion was successful, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO job_listings
                    (job_title, company, location, apply_url, description,
                     description_hash, source_url, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                # Convert metadata dict to JSON string for PostgreSQL
                import json
                metadata_json = json.dumps(job.metadata) if job.metadata else None

                values = (
                    job.job_title,
                    job.company,
                    job.location,
                    job.apply_url,
                    job.description,
                    job_hash,
                    job.source_url,
                    metadata_json
                )
                cur.execute(sql, values)
                conn.commit()
                return True
    except Exception as e:
        print(f"Database insert failed: {e}")
        return False


def get_all_job_listings(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve all job listings from the database.

    Args:
        limit: Maximum number of records to return.

    Returns:
        A list of dictionaries containing job listing data.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT id, job_title, company, location, apply_url,
                           description, source_url, created_at
                    FROM job_listings
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                cur.execute(sql, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching job listings: {e}")
        return []


def get_job_listing_by_id(listing_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single job listing by its ID.

    Args:
        listing_id: The database ID of the job listing.

    Returns:
        A dictionary containing the job listing data, or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT id, job_title, company, location, apply_url,
                           description, source_url, metadata, created_at
                    FROM job_listings
                    WHERE id = %s;
                """
                cur.execute(sql, (listing_id,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching job listing: {e}")
        return None


def delete_job_listing(listing_id: int) -> bool:
    """
    Delete a job listing by its ID.

    Args:
        listing_id: The database ID of the job listing to delete.

    Returns:
        True if deletion was successful, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = "DELETE FROM job_listings WHERE id = %s;"
                cur.execute(sql, (listing_id,))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error deleting job listing: {e}")
        return False


def test_connection() -> bool:
    """
    Test the database connection.

    Returns:
        True if connection is successful, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return True
    except Exception:
        return False


def check_tables_exist() -> dict:
    """
    Check which required tables exist in the database.

    Returns:
        Dictionary with table names as keys and existence status as values.
    """
    tables = {
        'job_listings': False,
        'resumes': False,
        'analysis_results': False
    }
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('job_listings', 'resumes', 'analysis_results');
                """)
                existing = {row[0] for row in cur.fetchall()}
                for table in tables:
                    tables[table] = table in existing
    except Exception:
        pass
    return tables


def tables_ready() -> bool:
    """
    Check if all required tables exist.

    Returns:
        True if all tables exist, False otherwise.
    """
    tables = check_tables_exist()
    return all(tables.values())


# =============================================================================
# Resume Database Operations (Phase 2)
# =============================================================================

def insert_resume(resume: Resume, file_hash: str) -> Optional[int]:
    """
    Insert a new resume record into the database.

    Args:
        resume: The Resume object containing the parsed data.
        file_hash: SHA-256 hash of the resume file for deduplication.

    Returns:
        The ID of the inserted resume, or None if insertion failed.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO resumes
                    (full_name, email, phone, location, summary, skills,
                     experience, education, projects, raw_text, file_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """
                values = (
                    resume.full_name,
                    resume.email,
                    resume.phone,
                    resume.location,
                    resume.summary,
                    json.dumps(resume.skills),
                    json.dumps(resume.experience),
                    json.dumps(resume.education),
                    json.dumps(resume.projects),
                    resume.raw_text,
                    file_hash,
                    json.dumps(resume.metadata) if resume.metadata else None
                )
                cur.execute(sql, values)
                resume_id = cur.fetchone()[0]
                conn.commit()
                return resume_id
    except Exception as e:
        print(f"Database insert failed: {e}")
        return None


def get_resume_by_id(resume_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a resume by its ID.

    Args:
        resume_id: The database ID of the resume.

    Returns:
        A dictionary containing the resume data, or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT id, full_name, email, phone, location, summary,
                           skills, experience, education, projects, raw_text,
                           file_hash, metadata, created_at, updated_at
                    FROM resumes
                    WHERE id = %s;
                """
                cur.execute(sql, (resume_id,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching resume: {e}")
        return None


def get_resume_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a resume by its file hash.

    Args:
        file_hash: The SHA-256 hash of the resume file.

    Returns:
        A dictionary containing the resume data, or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT id, full_name, email, phone, location, summary,
                           skills, experience, education, projects, raw_text,
                           file_hash, metadata, created_at, updated_at
                    FROM resumes
                    WHERE file_hash = %s;
                """
                cur.execute(sql, (file_hash,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching resume by hash: {e}")
        return None


def get_all_resumes(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve all resumes from the database.

    Args:
        limit: Maximum number of records to return.

    Returns:
        A list of dictionaries containing resume data.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT id, full_name, email, phone, location,
                           skills, created_at, updated_at
                    FROM resumes
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                cur.execute(sql, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching resumes: {e}")
        return []


def delete_resume(resume_id: int) -> bool:
    """
    Delete a resume by its ID.

    Args:
        resume_id: The database ID of the resume to delete.

    Returns:
        True if deletion was successful, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = "DELETE FROM resumes WHERE id = %s;"
                cur.execute(sql, (resume_id,))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error deleting resume: {e}")
        return False


def update_resume(resume_id: int, resume: Resume) -> bool:
    """
    Update an existing resume record.

    Args:
        resume_id: The database ID of the resume to update.
        resume: The Resume object containing the updated data.

    Returns:
        True if update was successful, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    UPDATE resumes
                    SET full_name = %s, email = %s, phone = %s, location = %s,
                        summary = %s, skills = %s, experience = %s, education = %s,
                        raw_text = %s, metadata = %s
                    WHERE id = %s;
                """
                values = (
                    resume.full_name,
                    resume.email,
                    resume.phone,
                    resume.location,
                    resume.summary,
                    json.dumps(resume.skills),
                    json.dumps(resume.experience),
                    json.dumps(resume.education),
                    resume.raw_text,
                    json.dumps(resume.metadata) if resume.metadata else None,
                    resume_id
                )
                cur.execute(sql, values)
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error updating resume: {e}")
        return False


# =============================================================================
# Analysis Results Database Operations (Phase 2)
# =============================================================================

def insert_analysis_result(
    resume_id: int,
    job_id: int,
    match_score: float,
    matching_skills: List[str],
    missing_skills: List[str],
    keyword_suggestions: List[str],
    improvement_suggestions: List[str],
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Insert a new analysis result into the database.

    Args:
        resume_id: The ID of the resume being analyzed.
        job_id: The ID of the job listing compared against.
        match_score: The calculated match score (0-100).
        matching_skills: List of skills that match.
        missing_skills: List of skills missing from the resume.
        keyword_suggestions: List of suggested keywords to add.
        improvement_suggestions: List of improvement suggestions.
        metadata: Additional analysis metadata.

    Returns:
        The ID of the inserted analysis result, or None if insertion failed.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO analysis_results
                    (resume_id, job_listing_id, match_score, matching_skills,
                     missing_skills, keyword_suggestions, improvement_suggestions,
                     analysis_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (resume_id, job_listing_id)
                    DO UPDATE SET
                        match_score = EXCLUDED.match_score,
                        matching_skills = EXCLUDED.matching_skills,
                        missing_skills = EXCLUDED.missing_skills,
                        keyword_suggestions = EXCLUDED.keyword_suggestions,
                        improvement_suggestions = EXCLUDED.improvement_suggestions,
                        analysis_metadata = EXCLUDED.analysis_metadata,
                        created_at = NOW()
                    RETURNING id;
                """
                values = (
                    resume_id,
                    job_id,
                    match_score,
                    json.dumps(matching_skills),
                    json.dumps(missing_skills),
                    json.dumps(keyword_suggestions),
                    json.dumps(improvement_suggestions),
                    json.dumps(metadata) if metadata else None
                )
                cur.execute(sql, values)
                result_id = cur.fetchone()[0]
                conn.commit()
                return result_id
    except Exception as e:
        print(f"Error inserting analysis result: {e}")
        return None


def get_analysis_for_resume(resume_id: int) -> List[Dict[str, Any]]:
    """
    Get all analysis results for a specific resume.

    Args:
        resume_id: The ID of the resume.

    Returns:
        A list of analysis result dictionaries.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT ar.*, jl.job_title, jl.company
                    FROM analysis_results ar
                    JOIN job_listings jl ON ar.job_listing_id = jl.id
                    WHERE ar.resume_id = %s
                    ORDER BY ar.created_at DESC;
                """
                cur.execute(sql, (resume_id,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching analysis results: {e}")
        return []


def get_analysis_for_job(job_id: int) -> List[Dict[str, Any]]:
    """
    Get all analysis results for a specific job listing.

    Args:
        job_id: The ID of the job listing.

    Returns:
        A list of analysis result dictionaries.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT ar.*, r.full_name, r.email
                    FROM analysis_results ar
                    JOIN resumes r ON ar.resume_id = r.id
                    WHERE ar.job_listing_id = %s
                    ORDER BY ar.match_score DESC;
                """
                cur.execute(sql, (job_id,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching analysis results: {e}")
        return []


def get_analysis_by_ids(resume_id: int, job_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific analysis result by resume and job IDs.

    Args:
        resume_id: The ID of the resume.
        job_id: The ID of the job listing.

    Returns:
        The analysis result dictionary, or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT *
                    FROM analysis_results
                    WHERE resume_id = %s AND job_listing_id = %s;
                """
                cur.execute(sql, (resume_id, job_id))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching analysis result: {e}")
        return None


# =============================================================================
# Phase 3: Embedding and Optimization Functions
# =============================================================================

def update_resume_sections(resume_id: int, updated_data: Dict[str, Any]) -> bool:
    """
    Update resume sections (experience, projects, summary) in database.

    Args:
        resume_id: ID of resume to update.
        updated_data: Dict containing updated fields.

    Returns:
        True if update successful.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    UPDATE resumes
                    SET
                        summary = COALESCE(%s, summary),
                        experience = COALESCE(%s, experience),
                        projects = COALESCE(%s, projects),
                        updated_at = NOW()
                    WHERE id = %s;
                """
                values = (
                    updated_data.get('summary'),
                    json.dumps(updated_data['experience']) if 'experience' in updated_data else None,
                    json.dumps(updated_data['projects']) if 'projects' in updated_data else None,
                    resume_id
                )
                cur.execute(sql, values)
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error updating resume: {e}")
        return False


def store_job_embedding(job_id: int, embedding_list: List[float]) -> bool:
    """
    Store embedding vector for a job listing.

    Args:
        job_id: ID of the job listing.
        embedding_list: List of floats representing the embedding.

    Returns:
        True if successful.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    UPDATE job_listings
                    SET description_embedding = %s
                    WHERE id = %s;
                """
                cur.execute(sql, (embedding_list, job_id))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error storing job embedding: {e}")
        return False


def store_resume_summary_embedding(resume_id: int, embedding_list: List[float]) -> bool:
    """
    Store summary embedding for a resume.

    Args:
        resume_id: ID of the resume.
        embedding_list: List of floats representing the embedding.

    Returns:
        True if successful.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    UPDATE resumes
                    SET summary_embedding = %s
                    WHERE id = %s;
                """
                cur.execute(sql, (embedding_list, resume_id))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error storing resume embedding: {e}")
        return False


def store_resume_embeddings(resume_id: int, resume: dict) -> bool:
    """
    Generate and store embeddings for all resume sections.
    Uses content hash to skip unchanged sections.

    Call this:
    - After initial resume save
    - After any section optimization is applied

    Args:
        resume_id: ID of the resume.
        resume: Resume data dictionary.

    Returns:
        True if successful.
    """
    try:
        # Import here to avoid circular imports
        from embeddings import embed_text, text_hash

        from text_utils import (
            format_experience_text,
            format_project_text,
            format_skills_text
        )

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sections_to_embed = []

                # Prepare experience sections
                for i, exp in enumerate(resume.get('experience', []) or []):
                    if exp:
                        content = format_experience_text(exp)
                        if content.strip():
                            sections_to_embed.append(('experience', i, content))

                # Prepare project sections
                for i, proj in enumerate(resume.get('projects', []) or []):
                    if proj:
                        content = format_project_text(proj)
                        if content.strip():
                            sections_to_embed.append(('project', i, content))

                # Prepare skills (single embedding for all skills)
                skills = resume.get('skills', []) or []
                if skills:
                    content = format_skills_text(skills)
                    if content.strip():
                        sections_to_embed.append(('skills', 0, content))

                # Update summary embedding directly on resumes table
                if resume.get('summary'):
                    summary_embedding = embed_text(resume['summary']).tolist()
                    cur.execute(
                        "UPDATE resumes SET summary_embedding = %s WHERE id = %s",
                        (summary_embedding, resume_id)
                    )

                # Upsert each section embedding
                for section_type, section_index, content in sections_to_embed:
                    content_md5 = text_hash(content)
                    embedding = embed_text(content).tolist()

                    sql = """
                        INSERT INTO resume_section_embeddings
                        (resume_id, section_type, section_index, content_hash, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (resume_id, section_type, section_index)
                        DO UPDATE SET
                            content_hash = EXCLUDED.content_hash,
                            embedding = EXCLUDED.embedding,
                            created_at = NOW()
                        WHERE resume_section_embeddings.content_hash != EXCLUDED.content_hash;
                    """
                    cur.execute(sql, (resume_id, section_type, section_index, content_md5, embedding))

                conn.commit()
                return True
    except Exception as e:
        print(f"Error storing section embeddings: {e}")
        return False


def check_embeddings_table_exists() -> bool:
    """
    Check if the resume_section_embeddings table exists.

    Returns:
        True if the table exists.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'resume_section_embeddings'
                    );
                """)
                return cur.fetchone()[0]
    except Exception:
        return False


def check_pgvector_available() -> bool:
    """
    Check if pgvector extension is available.

    Returns:
        True if pgvector is installed and enabled.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_extension WHERE extname = 'vector'
                    );
                """)
                return cur.fetchone()[0]
    except Exception:
        return False

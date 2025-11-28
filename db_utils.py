"""
Database Utility Module

This module provides a centralized interface for all database operations,
including connection management, deduplication checks, and data persistence.
"""
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from dotenv import load_dotenv

# Import shared data structure
from models import JobListing

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


def _get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables."""
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

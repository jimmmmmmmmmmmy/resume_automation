"""
LLM-Powered Section Optimization Module

This module uses OpenAI API to generate improved versions of weak resume sections,
with response caching to reduce costs and latency.

Key Features:
- Prompts explicitly forbid fabrication of metrics/facts
- DB-backed caching for LLM responses
- Section-specific prompts for experience, projects, and summary
"""

from typing import Dict, Any, Optional
import os

from dotenv import load_dotenv

from embeddings import text_hash
import db_utils

load_dotenv()

# Lazy-loaded OpenAI client
_client = None


def _get_client():
    """
    Lazy initialize OpenAI client.

    Returns:
        OpenAI client instance.

    Raises:
        ValueError: If OPENAI_API_KEY not set.
    """
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client


# System prompt with strict anti-hallucination rules
OPTIMIZER_SYSTEM_PROMPT = """You are an expert resume writer and career coach.
Your task is to rewrite resume sections to better match a target job description
while maintaining truthfulness and the candidate's authentic experience.

STRICT RULES:
- Do NOT invent new metrics, numbers, or achievements that aren't in the original
- Do NOT add skills or technologies the candidate didn't mention
- Do NOT exaggerate or fabricate accomplishments
- ONLY rephrase and reorganize existing information

Guidelines:
- Use strong action verbs (Led, Developed, Implemented, Optimized, Spearheaded)
- If the original has metrics, preserve them. If not, don't add fake ones.
- Mirror keywords from the job description naturally where they fit
- Keep bullet points concise (1-2 lines each)
- Focus on impact and results, not just responsibilities
- Reorder bullets to lead with most relevant content
"""


def _get_cache_key(section_content: str, job_description: str) -> str:
    """
    Generate cache key from section content and job description.

    Args:
        section_content: The section text to be optimized.
        job_description: The target job description.

    Returns:
        64-character cache key (MD5 hash).
    """
    # Truncate job description to prevent overly long keys
    combined = f"{section_content}|||{job_description[:2000]}"
    return text_hash(combined)


def _check_cache(cache_key: str) -> Optional[str]:
    """
    Check if we have a cached response for this key.

    Args:
        cache_key: The cache key to look up.

    Returns:
        Cached response text or None if not found.
    """
    try:
        with db_utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT response_text FROM llm_response_cache WHERE cache_key = %s",
                    (cache_key,)
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception:
        return None


def _store_cache(cache_key: str, response: str) -> None:
    """
    Store response in cache.

    Args:
        cache_key: The cache key.
        response: The response text to cache.
    """
    try:
        with db_utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO llm_response_cache (cache_key, response_text)
                       VALUES (%s, %s)
                       ON CONFLICT (cache_key) DO UPDATE SET
                           response_text = EXCLUDED.response_text,
                           created_at = NOW()""",
                    (cache_key, response)
                )
                conn.commit()
    except Exception as e:
        print(f"Cache store failed: {e}")


def generate_section_rewrite(
    section_type: str,
    section_content: Any,
    job_description: str,
    use_cache: bool = True
) -> str:
    """
    Generate a rewrite for a resume section.

    Args:
        section_type: 'experience', 'project', or 'summary'.
        section_content: The section data (dict for exp/proj, str for summary).
        job_description: The target job description.
        use_cache: Whether to check/store in cache (default True).

    Returns:
        Rewritten section text.
    """
    # Build content string for cache key
    if section_type == 'experience':
        content_str = _format_experience_for_prompt(section_content)
    elif section_type == 'project':
        content_str = _format_project_for_prompt(section_content)
    else:
        content_str = str(section_content)

    # Check cache first
    if use_cache:
        cache_key = _get_cache_key(content_str, job_description)
        cached = _check_cache(cache_key)
        if cached:
            return cached

    # Build prompt based on section type
    if section_type == 'experience':
        prompt = _build_experience_rewrite_prompt(section_content, job_description)
    elif section_type == 'project':
        prompt = _build_project_rewrite_prompt(section_content, job_description)
    elif section_type == 'summary':
        prompt = _build_summary_rewrite_prompt(section_content, job_description)
    else:
        return "Unsupported section type"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()

        # Store in cache
        if use_cache:
            _store_cache(cache_key, result)

        return result

    except Exception as e:
        return f"Error generating suggestion: {str(e)}"


def _format_experience_for_prompt(exp: Dict) -> str:
    """
    Format experience for cache key generation.

    Args:
        exp: Experience dictionary.

    Returns:
        Formatted string for cache key.
    """
    bullets = '\n'.join(exp.get('description', []) or [])
    return f"{exp.get('title', '')}|{exp.get('company', '')}|{bullets}"


def _format_project_for_prompt(proj: Dict) -> str:
    """
    Format project for cache key generation.

    Args:
        proj: Project dictionary.

    Returns:
        Formatted string for cache key.
    """
    bullets = '\n'.join(proj.get('description', []) or [])
    tech_list = proj.get('technologies', []) or []
    tech = ','.join(tech_list) if isinstance(tech_list, list) else str(tech_list)
    return f"{proj.get('name', '')}|{tech}|{bullets}"


def _build_experience_rewrite_prompt(exp: Dict, job_desc: str) -> str:
    """
    Build prompt for rewriting experience section.

    Args:
        exp: Experience dictionary.
        job_desc: Target job description.

    Returns:
        Complete prompt for LLM.
    """
    description_list = exp.get('description', []) or []
    bullets = '\n'.join(f"- {b}" for b in description_list)

    return f"""Rewrite this work experience section to better align with the target job.

CURRENT EXPERIENCE:
Title: {exp.get('title', 'Unknown Title')}
Company: {exp.get('company', 'Unknown Company')}
Dates: {exp.get('dates', '')}

Current Bullet Points:
{bullets}

TARGET JOB DESCRIPTION (key requirements to align with):
{job_desc[:2000]}

Provide an improved version of the bullet points that:
1. Uses keywords from the job description naturally
2. Emphasizes relevant skills and achievements
3. Maintains truthfulness - DO NOT add metrics or facts not in the original
4. Reorders to lead with most relevant content

Return ONLY the rewritten bullet points, one per line starting with "- ".
Do not include the title, company, or any other text."""


def _build_project_rewrite_prompt(proj: Dict, job_desc: str) -> str:
    """
    Build prompt for rewriting project section.

    Args:
        proj: Project dictionary.
        job_desc: Target job description.

    Returns:
        Complete prompt for LLM.
    """
    description_list = proj.get('description', []) or []
    bullets = '\n'.join(f"- {b}" for b in description_list)
    tech_list = proj.get('technologies', []) or []
    tech = ', '.join(tech_list) if tech_list else 'Not specified'

    return f"""Rewrite this project section to better align with the target job.

CURRENT PROJECT:
Name: {proj.get('name', 'Unknown Project')}
Technologies: {tech}

Current Description:
{bullets}

TARGET JOB DESCRIPTION (key requirements to align with):
{job_desc[:2000]}

Provide an improved version of the project description that:
1. Highlights technologies/skills mentioned in the job description
2. Emphasizes relevant outcomes and technical achievements
3. Uses strong action verbs
4. DO NOT add capabilities or results not mentioned in the original

Return ONLY the rewritten bullet points, one per line starting with "- ".
Do not include the project name or technologies list."""


def _build_summary_rewrite_prompt(summary: Any, job_desc: str) -> str:
    """
    Build prompt for rewriting summary section.

    Args:
        summary: Summary string or dict.
        job_desc: Target job description.

    Returns:
        Complete prompt for LLM.
    """
    if isinstance(summary, dict):
        summary_text = summary.get('content', str(summary))
    else:
        summary_text = str(summary)

    return f"""Rewrite this professional summary to better align with the target job.

CURRENT SUMMARY:
{summary_text}

TARGET JOB DESCRIPTION:
{job_desc[:2000]}

Provide an improved summary that:
1. Highlights experience relevant to the job requirements
2. Incorporates key skills mentioned in the job description
3. Is concise (2-4 sentences)
4. Maintains the candidate's authentic background - no fabrication

Return ONLY the rewritten summary paragraph. No bullet points or extra formatting."""


def check_api_available() -> bool:
    """
    Check if OpenAI API is configured and accessible.

    Returns:
        True if API is available, False otherwise.
    """
    try:
        client = _get_client()
        # Quick test - list models
        client.models.list()
        return True
    except Exception:
        return False


def clear_old_cache(days: int = 30) -> int:
    """
    Clear cache entries older than specified days.

    Args:
        days: Number of days after which to expire cache entries.

    Returns:
        Count of deleted entries.
    """
    try:
        with db_utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM llm_response_cache WHERE created_at < NOW() - INTERVAL '%s days'",
                    (days,)
                )
                count = cur.rowcount
                conn.commit()
                return count
    except Exception:
        return 0


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the LLM response cache.

    Returns:
        Dictionary with cache statistics.
    """
    try:
        with db_utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM llm_response_cache")
                total = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM llm_response_cache WHERE created_at > NOW() - INTERVAL '7 days'"
                )
                recent = cur.fetchone()[0]

                return {
                    'total_entries': total,
                    'entries_last_7_days': recent
                }
    except Exception:
        return {'total_entries': 0, 'entries_last_7_days': 0}

"""
Resume-Job Matching Module

This module implements semantic matching between resumes and job listings
using pgvector SQL queries for scalability.

Key Features:
- Uses HNSW index for O(log n) similarity search
- Section-level scoring for granular optimization
- Configurable thresholds with user-friendly interpretations
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np

from embeddings import embed_text, cosine_similarity, list_to_embedding
from text_utils import (
    format_experience_text,
    format_project_text,
    format_skills_text
)
import db_utils


@dataclass
class SectionScore:
    """Score for a single resume section."""
    section_type: str  # 'summary', 'experience', 'project', 'skills'
    section_index: int  # Index within section type (0 for summary/skills)
    score: float
    content: Any  # Original content for display/optimization


@dataclass
class MatchResult:
    """Result of matching a resume against a job."""
    resume_id: int
    resume_name: str
    overall_score: float
    section_scores: List[SectionScore] = field(default_factory=list)
    weak_sections: List[SectionScore] = field(default_factory=list)

    def score_dict(self) -> Dict[str, float]:
        """Convert to dict format for renderer."""
        result = {}
        for s in self.section_scores:
            if s.section_type in ('summary', 'skills'):
                result[s.section_type] = s.score
            else:
                result[f"{s.section_type}_{s.section_index}"] = s.score
        return result


# Scoring configuration - weights for different section types
SECTION_WEIGHTS = {
    'summary': 0.15,
    'experience': 0.40,  # Total weight divided among entries
    'projects': 0.25,    # Total weight divided among entries
    'skills': 0.20
}

# Score interpretation thresholds
# Based on empirical observation: sentence-BERT cosine similarity
# for related content typically falls in 0.4-0.8 range
SCORE_THRESHOLDS = {
    'weak': 0.50,      # Below this: definitely needs work
    'moderate': 0.65,  # Below this: could be improved
    'strong': 0.75,    # Above this: good match
}


def get_score_interpretation(score: float) -> str:
    """
    Return human-readable interpretation of match score.

    Args:
        score: Match score between 0 and 1.

    Returns:
        User-friendly description of match quality.
    """
    if score >= SCORE_THRESHOLDS['strong']:
        return "Strong match - well aligned with job requirements"
    elif score >= SCORE_THRESHOLDS['moderate']:
        return "Good match - minor improvements possible"
    elif score >= SCORE_THRESHOLDS['weak']:
        return "Moderate match - optimization recommended"
    else:
        return "Weak match - significant gaps to address"


def find_top_matching_resumes(
    job_id: int,
    limit: int = 5,
    threshold: float = SCORE_THRESHOLDS['weak']
) -> List[MatchResult]:
    """
    Find the best matching resumes for a job using pgvector similarity search.

    This performs the similarity search IN SQL using the HNSW index,
    not by loading all resumes into Python.

    Args:
        job_id: ID of the job to match against.
        limit: Maximum number of resumes to return.
        threshold: Minimum score threshold for weak section flagging.

    Returns:
        List of MatchResults sorted by overall score descending.
    """
    job = db_utils.get_job_listing_by_id(job_id)
    if not job:
        return []

    # Get job embedding (from DB or compute)
    job_embedding = _get_job_embedding(job)
    if job_embedding is None:
        return []

    job_embedding_list = job_embedding.tolist()

    # Try pgvector search first, fall back to loading all resumes
    candidates = []
    try:
        with db_utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                # First try: resumes with summary embeddings
                sql = """
                    SELECT id, full_name, summary, experience, projects, skills, education,
                           1 - (summary_embedding <=> %s::vector) as summary_similarity
                    FROM resumes
                    WHERE summary_embedding IS NOT NULL
                    ORDER BY summary_embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(sql, (job_embedding_list, job_embedding_list, limit * 2))
                candidates = cur.fetchall()

                # If no embeddings, fall back to all resumes
                if not candidates:
                    candidates = _fallback_get_resumes(limit * 2)
    except Exception as e:
        print(f"Error finding matching resumes: {e}")
        # Fall back to loading all resumes
        candidates = _fallback_get_resumes(limit * 2)

    if not candidates:
        return []

    # Score each candidate in detail
    results = []
    for row in candidates:
        resume = _row_to_resume_dict(row)
        match_result = score_resume_sections(resume, job_embedding, threshold)
        results.append(match_result)

    # Sort by overall score and return top N
    results.sort(key=lambda x: x.overall_score, reverse=True)
    return results[:limit]


def score_resume_sections(
    resume: Dict[str, Any],
    job_embedding: np.ndarray,
    threshold: float = SCORE_THRESHOLDS['weak']
) -> MatchResult:
    """
    Score all sections of a resume against a job embedding.

    Args:
        resume: Resume data dict (must include 'id' and 'full_name').
        job_embedding: Pre-computed job description embedding.
        threshold: Score below which sections are flagged as weak.

    Returns:
        MatchResult with section scores and weak sections identified.
    """
    section_scores = []
    weak_sections = []

    # Score summary
    if resume.get('summary'):
        summary_text = resume['summary']
        if isinstance(summary_text, str) and summary_text.strip():
            score = cosine_similarity(embed_text(summary_text), job_embedding)
            section = SectionScore('summary', 0, score, summary_text)
            section_scores.append(section)
            if score < threshold:
                weak_sections.append(section)

    # Score each experience entry
    for i, exp in enumerate(resume.get('experience', []) or []):
        if not exp:
            continue
        exp_text = format_experience_text(exp)
        if exp_text.strip():
            score = cosine_similarity(embed_text(exp_text), job_embedding)
            section = SectionScore('experience', i, score, exp)
            section_scores.append(section)
            if score < threshold:
                weak_sections.append(section)

    # Score each project
    for i, proj in enumerate(resume.get('projects', []) or []):
        if not proj:
            continue
        proj_text = format_project_text(proj)
        if proj_text.strip():
            score = cosine_similarity(embed_text(proj_text), job_embedding)
            section = SectionScore('project', i, score, proj)
            section_scores.append(section)
            if score < threshold:
                weak_sections.append(section)

    # Score skills
    skills = resume.get('skills', []) or []
    if skills:
        skills_text = format_skills_text(skills)
        if skills_text.strip():
            score = cosine_similarity(embed_text(skills_text), job_embedding)
            section = SectionScore('skills', 0, score, skills)
            section_scores.append(section)
            if score < threshold:
                weak_sections.append(section)

    # Calculate weighted overall score
    overall_score = _calculate_weighted_score(section_scores)

    # Sort weak sections by score ascending (worst first)
    weak_sections.sort(key=lambda x: x.score)

    return MatchResult(
        resume_id=resume.get('id', 0),
        resume_name=resume.get('full_name', 'Unknown'),
        overall_score=overall_score,
        section_scores=section_scores,
        weak_sections=weak_sections
    )


def match_single_resume(
    resume_id: int,
    job_id: int,
    threshold: float = SCORE_THRESHOLDS['weak']
) -> Optional[MatchResult]:
    """
    Score a specific resume against a job.

    Use this when user has already selected which resume to optimize.

    Args:
        resume_id: ID of the resume to score.
        job_id: ID of the job to match against.
        threshold: Score below which sections are flagged as weak.

    Returns:
        MatchResult or None if resume/job not found.
    """
    job = db_utils.get_job_listing_by_id(job_id)
    resume = db_utils.get_resume_by_id(resume_id)

    if not job or not resume:
        return None

    job_embedding = _get_job_embedding(job)
    if job_embedding is None:
        return None

    return score_resume_sections(resume, job_embedding, threshold)


def _get_job_embedding(job: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Get embedding for a job, from DB or by computing.

    Args:
        job: Job dictionary from database.

    Returns:
        Numpy embedding array or None if no description.
    """
    description = job.get('description', '')
    if not description or not description.strip():
        return None

    # Check if embedding already stored
    if job.get('description_embedding'):
        return list_to_embedding(job['description_embedding'])

    # Compute embedding
    return embed_text(description)


def _calculate_weighted_score(section_scores: List[SectionScore]) -> float:
    """
    Calculate weighted overall score from section scores.

    Experience and project weights are divided among their entries.
    If summary is missing, its weight is redistributed to other sections.

    Args:
        section_scores: List of section scores.

    Returns:
        Weighted average score between 0 and 1.
    """
    if not section_scores:
        return 0.0

    # Count entries per category
    exp_count = sum(1 for s in section_scores if s.section_type == 'experience')
    proj_count = sum(1 for s in section_scores if s.section_type == 'project')
    has_summary = any(s.section_type == 'summary' for s in section_scores)
    has_skills = any(s.section_type == 'skills' for s in section_scores)

    # Calculate adjusted weights if summary is missing
    base_weights = SECTION_WEIGHTS.copy()
    if not has_summary:
        # Redistribute summary weight to other sections
        redistrib = base_weights['summary']
        base_weights['summary'] = 0
        if exp_count > 0:
            base_weights['experience'] += redistrib * 0.5
        if proj_count > 0:
            base_weights['projects'] += redistrib * 0.3
        if has_skills:
            base_weights['skills'] += redistrib * 0.2

    total_weight = 0.0
    weighted_sum = 0.0

    for section in section_scores:
        if section.section_type == 'summary':
            weight = base_weights['summary']
        elif section.section_type == 'skills':
            weight = base_weights['skills']
        elif section.section_type == 'experience':
            weight = base_weights['experience'] / max(exp_count, 1)
        elif section.section_type == 'project':
            weight = base_weights['projects'] / max(proj_count, 1)
        else:
            weight = 0.0

        weighted_sum += section.score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _row_to_resume_dict(row) -> Dict[str, Any]:
    """
    Convert database row to resume dictionary.

    Args:
        row: Database row tuple from cursor.

    Returns:
        Resume dictionary with expected fields.
    """
    return {
        'id': row[0],
        'full_name': row[1],
        'summary': row[2],
        'experience': row[3] or [],
        'projects': row[4] or [],
        'skills': row[5] or [],
        'education': row[6] or []
    }


def _fallback_get_resumes(limit: int) -> List:
    """
    Fallback method to get resumes when pgvector query fails.

    Args:
        limit: Maximum number of resumes to return.

    Returns:
        List of resume rows.
    """
    try:
        with db_utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT id, full_name, summary, experience, projects, skills, education,
                           0.0 as summary_similarity
                    FROM resumes
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                cur.execute(sql, (limit,))
                return cur.fetchall()
    except Exception as e:
        print(f"Fallback resume fetch failed: {e}")
        return []

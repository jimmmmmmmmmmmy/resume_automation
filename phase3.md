#### **Phase 3: Resume-Job Matching & Optimization Pipeline**

**1. Overview & Objectives**

*   **1.1. Vision:** To create an intelligent pipeline that takes a saved job listing, finds the best-matching resume from the database, scores each resume section against the job requirements, and provides an interactive "Apply" interface where users can review and optimize underperforming sections with LLM-generated suggestions.

*   **1.2. Key Objectives for Phase 3:**
    *   Implement semantic vectorization for job listings and resume sections
    *   Build a matching algorithm using pgvector SQL queries (not Python iteration)
    *   Create section-level scoring to identify weak areas needing improvement
    *   Integrate LLM API calls to generate targeted rewrite suggestions
    *   Build an "Apply" tab with a PDF-like resume preview and inline editing UI
    *   Ensure embeddings stay fresh after optimizations are applied

**2. System Architecture & Design**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLY TAB UI                                    │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │      Resume PDF Preview         │  │      Optimization Panel          │ │
│  │  ┌───────────────────────────┐  │  │  ┌────────────────────────────┐ │ │
│  │  │ Contact Info              │  │  │  │ Match Score: 78%           │ │ │
│  │  ├───────────────────────────┤  │  │  │ (Above average for role)   │ │ │
│  │  │ Summary [Score: 0.85]     │  │  │  ├────────────────────────────┤ │ │
│  │  ├───────────────────────────┤  │  │  │ Weak Sections:             │ │ │
│  │  │ Experience                │  │  │  │  - Experience #2 (0.62)    │ │ │
│  │  │  └─ Role 1 [0.91]        │  │  │  │  - Projects #1 (0.58)      │ │ │
│  │  │  └─ Role 2 [0.62] ← RED  │  │  │  ├────────────────────────────┤ │ │
│  │  ├───────────────────────────┤  │  │  │ [Optimize Selected]        │ │ │
│  │  │ Projects                  │  │  │  └────────────────────────────┘ │ │
│  │  │  └─ Proj 1 [0.58] ← RED  │  │  │                                  │ │
│  │  │  └─ Proj 2 [0.88]        │  │  │  Suggested Rewrites:            │ │
│  │  ├───────────────────────────┤  │  │  ┌────────────────────────────┐ │ │
│  │  │ Skills [0.82]             │  │  │  │ Option 1: "Led cross..."  │ │ │
│  │  └───────────────────────────┘  │  │  │ [Accept] [Edit] [Reject]  │ │ │
│  └─────────────────────────────────┘  │  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

*   **New Components:**
    *   `text_utils.py` - Shared text formatting functions (prevents duplication)
    *   `embeddings.py` - Vectorization service using Sentence-BERT
    *   `matcher.py` - Resume-job matching using pgvector SQL queries
    *   `optimizer.py` - LLM-powered section rewriting with response caching
    *   `resume_renderer.py` - PDF/HTML resume preview generation

**3. Detailed Task Breakdown**

---

### **Task 1: Shared Text Utilities**

**Objective:** Centralize text formatting to prevent code duplication between db_utils and matcher.

```python
# text_utils.py
"""Shared text formatting utilities for embeddings and display."""

from typing import Dict, List, Any

def format_experience_text(exp: Dict[str, Any]) -> str:
    """
    Format experience entry as text for embedding.

    Used by both db_utils (on save) and matcher (on score).
    """
    parts = [
        exp.get('title', ''),
        exp.get('company', ''),
    ]
    if exp.get('description'):
        parts.extend(exp['description'])
    return ' '.join(filter(None, parts))

def format_project_text(proj: Dict[str, Any]) -> str:
    """Format project entry as text for embedding."""
    parts = [proj.get('name', '')]
    if proj.get('technologies'):
        parts.append(', '.join(proj['technologies']))
    if proj.get('description'):
        parts.extend(proj['description'])
    return ' '.join(filter(None, parts))

def format_skills_text(skills: List[str]) -> str:
    """Format skills list as text for embedding."""
    return ', '.join(skills) if skills else ''
```

**Definition of Done:**
- [ ] `text_utils.py` created with formatting functions
- [ ] All other modules import from text_utils (no duplication)

---

### **Task 2: Embedding Infrastructure**

**Objective:** Set up the vectorization pipeline using Sentence-BERT for semantic similarity comparisons.

#### **Step 2.1: Create the Embeddings Module**

```python
# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import hashlib

# Load model once at module level
# all-MiniLM-L6-v2: 384-dim vectors, ~10-50ms per embedding on CPU
MODEL: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    """Lazy load the model to avoid startup delay."""
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return MODEL

def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding vector for a single text.

    Returns normalized 384-dim vector (dot product = cosine similarity).
    """
    model = _get_model()
    return model.encode(text, normalize_embeddings=True)

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embedding vectors for multiple texts (batched for efficiency)."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two normalized vectors.

    Since vectors are normalized, dot product equals cosine similarity.
    """
    return float(np.dot(vec1, vec2))

def text_hash(text: str) -> str:
    """Generate hash for caching embeddings and LLM responses."""
    return hashlib.md5(text.encode()).hexdigest()
```

#### **Step 2.2: Database Schema Updates**

> **Design Decision:** We use HNSW indexes (not ivfflat) because:
> - HNSW doesn't require pre-existing training data
> - Better speed/accuracy tradeoff for small-medium datasets
> - Simpler tuning (no nprobe parameter needed at query time)

```sql
-- scripts/migrations/003_add_embeddings.sql

-- Enable pgvector extension (run once)
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column to job_listings
ALTER TABLE job_listings
ADD COLUMN IF NOT EXISTS description_embedding vector(384);

-- Add summary embedding to resumes
ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS summary_embedding vector(384);

-- Separate table for section-level embeddings (experience, projects)
-- This allows efficient similarity searches via pgvector
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

-- Create HNSW indexes for fast similarity search
-- HNSW parameters: m=16 (connections), ef_construction=64 (build quality)
CREATE INDEX IF NOT EXISTS idx_job_embedding_hnsw
ON job_listings USING hnsw (description_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_resume_summary_embedding_hnsw
ON resumes USING hnsw (summary_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_section_embedding_hnsw
ON resume_section_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for fast lookups by resume
CREATE INDEX IF NOT EXISTS idx_section_by_resume
ON resume_section_embeddings(resume_id, section_type);

-- Table for caching LLM optimization responses
CREATE TABLE IF NOT EXISTS llm_response_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(64) NOT NULL UNIQUE,  -- MD5(section_content + job_desc)
    response_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Expire old cache entries (optional cleanup)
CREATE INDEX IF NOT EXISTS idx_cache_created ON llm_response_cache(created_at);
```

#### **Step 2.3: Embed on Save**

Update `db_utils.py` to generate and store embeddings when saving jobs/resumes:

```python
# In db_utils.py

from embeddings import embed_text, text_hash
from text_utils import format_experience_text, format_project_text, format_skills_text

def insert_job_listing_with_embedding(job: JobListing, job_hash: str) -> bool:
    """Insert job with its embedding vector."""
    try:
        embedding = embed_text(job.description)
        embedding_list = embedding.tolist()

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO job_listings
                    (job_title, company, location, apply_url, description,
                     description_hash, source_url, metadata, description_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                values = (
                    job.job_title, job.company, job.location, job.apply_url,
                    job.description, job_hash, job.source_url,
                    json.dumps(job.metadata) if job.metadata else None,
                    embedding_list
                )
                cur.execute(sql, values)
                conn.commit()
                return True
    except Exception as e:
        print(f"Database insert failed: {e}")
        return False

def store_resume_embeddings(resume_id: int, resume: dict) -> bool:
    """
    Generate and store embeddings for all resume sections.
    Uses content hash to skip unchanged sections.

    Call this:
    - After initial resume save
    - After any section optimization is applied
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                sections_to_embed = []

                # Prepare experience sections
                for i, exp in enumerate(resume.get('experience', [])):
                    content = format_experience_text(exp)
                    sections_to_embed.append(('experience', i, content))

                # Prepare project sections
                for i, proj in enumerate(resume.get('projects', [])):
                    content = format_project_text(proj)
                    sections_to_embed.append(('project', i, content))

                # Prepare skills (single embedding for all skills)
                if resume.get('skills'):
                    content = format_skills_text(resume['skills'])
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
```

**Definition of Done:**
- [ ] `embeddings.py` module created with embed/similarity functions
- [ ] Migration script creates vector columns and HNSW indexes
- [ ] LLM response cache table created
- [ ] Jobs store embeddings on save
- [ ] Resume sections store embeddings in separate table
- [ ] Content hashing prevents redundant re-embedding
- [ ] Unit tests verify embedding generation and similarity calculations

---

### **Task 3: Resume-Job Matching Algorithm**

**Objective:** Build matching logic using pgvector SQL queries for scalability.

> **Critical Design Decision:** We perform similarity search in PostgreSQL, NOT in Python.
> This leverages the HNSW index and scales to thousands of resumes.

#### **Step 3.1: Create the Matcher Module**

```python
# matcher.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from embeddings import embed_text, cosine_similarity
from text_utils import format_experience_text, format_project_text, format_skills_text
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

# Scoring configuration
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
    """Return human-readable interpretation of match score."""
    if score >= 0.75:
        return "Strong match - well aligned with job requirements"
    elif score >= 0.65:
        return "Good match - minor improvements possible"
    elif score >= 0.50:
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
        job_id: ID of the job to match against
        limit: Maximum number of resumes to return
        threshold: Minimum score threshold for weak section flagging

    Returns:
        List of MatchResults sorted by overall score descending
    """
    job = db_utils.get_job_listing_by_id(job_id)
    if not job:
        return []

    # Get job embedding (from DB or compute)
    if job.get('description_embedding'):
        job_embedding = np.array(job['description_embedding'])
    else:
        job_embedding = embed_text(job['description'])

    job_embedding_list = job_embedding.tolist()

    # Use pgvector to find top matching resumes by summary similarity
    # This is O(log n) with HNSW index, not O(n)
    with db_utils.get_db_connection() as conn:
        with conn.cursor() as cur:
            # Find resumes with closest summary embeddings
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

    if not candidates:
        return []

    # Score each candidate in detail
    results = []
    for row in candidates:
        resume = {
            'id': row[0],
            'full_name': row[1],
            'summary': row[2],
            'experience': row[3] or [],
            'projects': row[4] or [],
            'skills': row[5] or [],
            'education': row[6] or []
        }

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
        resume: Resume data dict (must include 'id' and 'full_name')
        job_embedding: Pre-computed job description embedding
        threshold: Score below which sections are flagged as weak

    Returns:
        MatchResult with section scores and weak sections identified
    """
    section_scores = []
    weak_sections = []

    # Score summary
    if resume.get('summary'):
        score = cosine_similarity(embed_text(resume['summary']), job_embedding)
        section = SectionScore('summary', 0, score, resume['summary'])
        section_scores.append(section)
        if score < threshold:
            weak_sections.append(section)

    # Score each experience entry
    for i, exp in enumerate(resume.get('experience', [])):
        exp_text = format_experience_text(exp)
        score = cosine_similarity(embed_text(exp_text), job_embedding)
        section = SectionScore('experience', i, score, exp)
        section_scores.append(section)
        if score < threshold:
            weak_sections.append(section)

    # Score each project
    for i, proj in enumerate(resume.get('projects', [])):
        proj_text = format_project_text(proj)
        score = cosine_similarity(embed_text(proj_text), job_embedding)
        section = SectionScore('project', i, score, proj)
        section_scores.append(section)
        if score < threshold:
            weak_sections.append(section)

    # Score skills
    if resume.get('skills'):
        skills_text = format_skills_text(resume['skills'])
        score = cosine_similarity(embed_text(skills_text), job_embedding)
        section = SectionScore('skills', 0, score, resume['skills'])
        section_scores.append(section)
        if score < threshold:
            weak_sections.append(section)

    # Calculate weighted overall score
    overall_score = _calculate_weighted_score(section_scores)

    # Sort weak sections by score ascending (worst first)
    weak_sections.sort(key=lambda x: x.score)

    return MatchResult(
        resume_id=resume['id'],
        resume_name=resume.get('full_name', 'Unknown'),
        overall_score=overall_score,
        section_scores=section_scores,
        weak_sections=weak_sections
    )

def _calculate_weighted_score(section_scores: List[SectionScore]) -> float:
    """
    Calculate weighted overall score from section scores.

    Experience and project weights are divided among their entries.
    """
    if not section_scores:
        return 0.0

    # Count entries per category
    exp_count = sum(1 for s in section_scores if s.section_type == 'experience')
    proj_count = sum(1 for s in section_scores if s.section_type == 'project')

    total_weight = 0.0
    weighted_sum = 0.0

    for section in section_scores:
        if section.section_type == 'summary':
            weight = SECTION_WEIGHTS['summary']
        elif section.section_type == 'skills':
            weight = SECTION_WEIGHTS['skills']
        elif section.section_type == 'experience':
            weight = SECTION_WEIGHTS['experience'] / max(exp_count, 1)
        elif section.section_type == 'project':
            weight = SECTION_WEIGHTS['projects'] / max(proj_count, 1)
        else:
            weight = 0.0

        weighted_sum += section.score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0

def match_single_resume(
    resume_id: int,
    job_id: int,
    threshold: float = SCORE_THRESHOLDS['weak']
) -> Optional[MatchResult]:
    """
    Score a specific resume against a job.

    Use this when user has already selected which resume to optimize.
    """
    job = db_utils.get_job_listing_by_id(job_id)
    resume = db_utils.get_resume_by_id(resume_id)

    if not job or not resume:
        return None

    if job.get('description_embedding'):
        job_embedding = np.array(job['description_embedding'])
    else:
        job_embedding = embed_text(job['description'])

    return score_resume_sections(resume, job_embedding, threshold)
```

**Definition of Done:**
- [ ] `matcher.py` uses pgvector SQL queries (not Python iteration)
- [ ] `find_top_matching_resumes()` leverages HNSW index
- [ ] Section-level scoring implemented for all resume sections
- [ ] Score interpretation helper provides user-friendly feedback
- [ ] Weighted scoring correctly distributes weights
- [ ] Weak sections correctly identified based on threshold
- [ ] Unit tests with sample job/resume pairs

---

### **Task 4: LLM-Powered Section Optimization**

**Objective:** Use OpenAI API to generate improved versions of weak resume sections, with response caching.

#### **Step 4.1: Create the Optimizer Module**

```python
# optimizer.py
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from embeddings import text_hash
import db_utils

load_dotenv()

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    """Lazy initialize OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client

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
    """Generate cache key from section content and job description."""
    combined = f"{section_content}|||{job_description[:2000]}"
    return text_hash(combined)

def _check_cache(cache_key: str) -> Optional[str]:
    """Check if we have a cached response for this key."""
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
    """Store response in cache."""
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
        section_type: 'experience', 'project', or 'summary'
        section_content: The section data (dict for exp/proj, str for summary)
        job_description: The target job description
        use_cache: Whether to check/store in cache (default True)

    Returns:
        Rewritten section text
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

    # Build prompt
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
    """Format experience for cache key generation."""
    bullets = '\n'.join(exp.get('description', []))
    return f"{exp.get('title', '')}|{exp.get('company', '')}|{bullets}"

def _format_project_for_prompt(proj: Dict) -> str:
    """Format project for cache key generation."""
    bullets = '\n'.join(proj.get('description', []))
    tech = ','.join(proj.get('technologies', []))
    return f"{proj.get('name', '')}|{tech}|{bullets}"

def _build_experience_rewrite_prompt(exp: Dict, job_desc: str) -> str:
    """Build prompt for rewriting experience section."""
    bullets = '\n'.join(f"- {b}" for b in exp.get('description', []))
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
    """Build prompt for rewriting project section."""
    bullets = '\n'.join(f"- {b}" for b in proj.get('description', []))
    tech = ', '.join(proj.get('technologies', [])) if proj.get('technologies') else 'Not specified'

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

def _build_summary_rewrite_prompt(summary: str, job_desc: str) -> str:
    """Build prompt for rewriting summary section."""
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
    """Check if OpenAI API is configured and accessible."""
    try:
        client = _get_client()
        client.models.list()
        return True
    except Exception:
        return False

def clear_old_cache(days: int = 30) -> int:
    """Clear cache entries older than specified days. Returns count deleted."""
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
```

**Definition of Done:**
- [ ] `optimizer.py` module created with OpenAI v1.0+ client
- [ ] Prompts explicitly forbid fabrication of metrics/facts
- [ ] LLM response caching implemented (DB-backed)
- [ ] Prompts implemented for experience, project, and summary sections
- [ ] Error handling for API failures
- [ ] Cache cleanup utility function
- [ ] API availability check function

---

### **Task 5: Resume Preview Renderer**

**Objective:** Create a PDF-like visual preview of the resume with section highlighting.

```python
# resume_renderer.py
from typing import Dict, List, Any, Optional
from jinja2 import Template
import html

def render_resume_html(
    resume: Dict[str, Any],
    section_scores: Optional[Dict[str, float]] = None,
    threshold: float = 0.50,
    highlighted_section: Optional[str] = None
) -> str:
    """
    Render resume as styled HTML with optional score highlighting.

    Args:
        resume: Resume data dictionary
        section_scores: Optional dict of section -> score for coloring
        threshold: Score below which sections are highlighted red (default 0.50)
        highlighted_section: Currently selected section for editing

    Returns:
        HTML string safe for embedding in Streamlit
    """
    # Escape all user content to prevent XSS
    safe_resume = _escape_resume_content(resume)

    template = Template(RESUME_HTML_TEMPLATE)

    # Prepare section styling based on scores
    section_styles = {}
    if section_scores:
        for section, score in section_scores.items():
            if score < threshold:
                section_styles[section] = 'weak-section'
            elif score >= 0.70:
                section_styles[section] = 'strong-section'
            else:
                section_styles[section] = 'moderate-section'

    return template.render(
        resume=safe_resume,
        section_styles=section_styles,
        highlighted_section=highlighted_section,
        section_scores=section_scores or {}
    )

def _escape_resume_content(resume: Dict[str, Any]) -> Dict[str, Any]:
    """Escape HTML in all string fields to prevent XSS."""
    def escape_value(v):
        if isinstance(v, str):
            return html.escape(v)
        elif isinstance(v, list):
            return [escape_value(item) for item in v]
        elif isinstance(v, dict):
            return {k: escape_value(val) for k, val in v.items()}
        return v

    return escape_value(resume)

def render_resume_plaintext(resume: Dict[str, Any]) -> str:
    """Render resume as plain text for clipboard copying."""
    lines = []

    # Header
    lines.append(resume.get('full_name', 'Your Name').upper())
    contact = ' | '.join(filter(None, [
        resume.get('email'),
        resume.get('phone'),
        resume.get('location')
    ]))
    if contact:
        lines.append(contact)
    lines.append('')

    # Summary
    if resume.get('summary'):
        lines.append('SUMMARY')
        lines.append(resume['summary'])
        lines.append('')

    # Experience
    if resume.get('experience'):
        lines.append('EXPERIENCE')
        for exp in resume['experience']:
            lines.append(f"{exp.get('title', '')} - {exp.get('company', '')}")
            if exp.get('dates'):
                lines.append(exp['dates'])
            for bullet in exp.get('description', []):
                lines.append(f"  - {bullet}")
            lines.append('')

    # Projects
    if resume.get('projects'):
        lines.append('PROJECTS')
        for proj in resume['projects']:
            lines.append(proj.get('name', ''))
            if proj.get('technologies'):
                lines.append(f"Technologies: {', '.join(proj['technologies'])}")
            for bullet in proj.get('description', []):
                lines.append(f"  - {bullet}")
            lines.append('')

    # Skills
    if resume.get('skills'):
        lines.append('SKILLS')
        lines.append(', '.join(resume['skills']))
        lines.append('')

    # Education
    if resume.get('education'):
        lines.append('EDUCATION')
        for edu in resume['education']:
            lines.append(f"{edu.get('degree', '')} - {edu.get('institution', '')}")
            if edu.get('dates'):
                lines.append(edu['dates'])
        lines.append('')

    return '\n'.join(lines)

RESUME_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
    * { box-sizing: border-box; }

    body {
        font-family: 'Georgia', 'Times New Roman', serif;
        max-width: 8.5in;
        margin: 0 auto;
        padding: 0.5in;
        background: white;
        color: #333;
        line-height: 1.4;
        font-size: 11pt;
    }

    .header {
        text-align: center;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    .header h1 {
        margin: 0;
        font-size: 22pt;
        color: #1a1a1a;
    }

    .contact-info {
        font-size: 10pt;
        color: #555;
        margin-top: 5px;
    }

    .section {
        margin-bottom: 15px;
        padding: 8px;
        border-radius: 4px;
        transition: all 0.3s ease;
        position: relative;
    }

    .section-header {
        font-size: 11pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #333;
        margin-bottom: 8px;
        padding-bottom: 3px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #1a1a1a;
    }

    .score-badge {
        font-size: 9pt;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: normal;
        text-transform: none;
        letter-spacing: 0;
    }

    .weak-section {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }

    .weak-section .score-badge {
        background-color: #f44336;
        color: white;
    }

    .moderate-section {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
    }

    .moderate-section .score-badge {
        background-color: #ff9800;
        color: white;
    }

    .strong-section {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }

    .strong-section .score-badge {
        background-color: #4caf50;
        color: white;
    }

    .highlighted {
        box-shadow: 0 0 0 3px #2196f3;
    }

    .entry {
        margin-bottom: 12px;
        padding: 4px;
        border-radius: 3px;
    }

    .entry-header {
        display: flex;
        justify-content: space-between;
        font-weight: bold;
        color: #1a1a1a;
    }

    .entry-subheader {
        font-style: italic;
        color: #555;
        font-size: 10pt;
    }

    ul {
        margin: 5px 0;
        padding-left: 20px;
    }

    li {
        margin-bottom: 3px;
        font-size: 10pt;
        color: #333;
    }

    .skills-list {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }

    .skill-tag {
        background: #e0e0e0;
        color: #333;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 9pt;
    }

    p {
        margin: 5px 0;
        color: #333;
    }
</style>
</head>
<body>
    <div class="header">
        <h1>{{ resume.full_name or 'Your Name' }}</h1>
        <div class="contact-info">
            {{ resume.email or '' }}{% if resume.email and (resume.phone or resume.location) %} | {% endif %}{{ resume.phone or '' }}{% if resume.phone and resume.location %} | {% endif %}{{ resume.location or '' }}
        </div>
    </div>

    {% if resume.summary %}
    <div class="section {{ section_styles.get('summary', '') }}{% if highlighted_section == 'summary' %} highlighted{% endif %}"
         data-section="summary">
        <div class="section-header">
            Summary
            {% if section_scores.get('summary') %}
            <span class="score-badge">{{ "%.0f"|format(section_scores.get('summary', 0) * 100) }}%</span>
            {% endif %}
        </div>
        <p>{{ resume.summary }}</p>
    </div>
    {% endif %}

    {% if resume.experience %}
    <div class="section">
        <div class="section-header">Experience</div>
        {% for exp in resume.experience %}
        <div class="entry {{ section_styles.get('experience_' ~ loop.index0, '') }}"
             data-section="experience_{{ loop.index0 }}">
            <div class="entry-header">
                <span>{{ exp.title or 'Position' }}</span>
                <span>{{ exp.dates or '' }}</span>
            </div>
            <div class="entry-subheader">{{ exp.company or '' }}</div>
            {% if exp.description %}
            <ul>
            {% for bullet in exp.description %}
                <li>{{ bullet }}</li>
            {% endfor %}
            </ul>
            {% endif %}
            {% if section_scores.get('experience_' ~ loop.index0) %}
            <span class="score-badge" style="position: absolute; right: 8px; top: 8px;">{{ "%.0f"|format(section_scores.get('experience_' ~ loop.index0, 0) * 100) }}%</span>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if resume.projects %}
    <div class="section">
        <div class="section-header">Projects</div>
        {% for proj in resume.projects %}
        <div class="entry {{ section_styles.get('project_' ~ loop.index0, '') }}"
             data-section="project_{{ loop.index0 }}">
            <div class="entry-header">
                <span>{{ proj.name or 'Project' }}</span>
            </div>
            {% if proj.technologies %}
            <div class="skills-list" style="margin-bottom: 5px;">
                {% for tech in proj.technologies %}
                <span class="skill-tag">{{ tech }}</span>
                {% endfor %}
            </div>
            {% endif %}
            {% if proj.description %}
            <ul>
            {% for bullet in proj.description %}
                <li>{{ bullet }}</li>
            {% endfor %}
            </ul>
            {% endif %}
            {% if section_scores.get('project_' ~ loop.index0) %}
            <span class="score-badge" style="position: absolute; right: 8px; top: 8px;">{{ "%.0f"|format(section_scores.get('project_' ~ loop.index0, 0) * 100) }}%</span>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if resume.skills %}
    <div class="section {{ section_styles.get('skills', '') }}"
         data-section="skills">
        <div class="section-header">
            Skills
            {% if section_scores.get('skills') %}
            <span class="score-badge">{{ "%.0f"|format(section_scores.get('skills', 0) * 100) }}%</span>
            {% endif %}
        </div>
        <div class="skills-list">
            {% for skill in resume.skills %}
            <span class="skill-tag">{{ skill }}</span>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    {% if resume.education %}
    <div class="section">
        <div class="section-header">Education</div>
        {% for edu in resume.education %}
        <div class="entry">
            <div class="entry-header">
                <span>{{ edu.degree or 'Degree' }}</span>
                <span>{{ edu.dates or '' }}</span>
            </div>
            <div class="entry-subheader">{{ edu.institution or '' }}</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
"""

def render_resume_pdf(resume: Dict[str, Any], output_path: str) -> str:
    """
    Render resume as PDF file.

    Requires WeasyPrint and system dependencies:
    - macOS: brew install cairo pango gdk-pixbuf libffi
    - Ubuntu: apt install libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0

    Returns path to generated PDF.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError(
            "WeasyPrint not installed. Install with: pip install weasyprint\n"
            "Also requires system deps: brew install cairo pango gdk-pixbuf"
        )

    html_content = render_resume_html(resume, section_scores=None)
    HTML(string=html_content).write_pdf(output_path)
    return output_path

def render_resume_pdf_bytes(resume: Dict[str, Any]) -> bytes:
    """Render resume as PDF bytes (for Streamlit download button)."""
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError("WeasyPrint not installed")

    html_content = render_resume_html(resume, section_scores=None)
    return HTML(string=html_content).write_pdf()
```

**Definition of Done:**
- [ ] `resume_renderer.py` module created
- [ ] HTML template renders all resume sections with proper escaping
- [ ] Three-tier color highlighting (red < 0.50, orange 0.50-0.70, green >= 0.70)
- [ ] Plain text export for clipboard
- [ ] PDF export functional (with documented system dependencies)
- [ ] Section selection highlighting for editing

---

### **Task 6: Apply Tab UI Implementation**

**Objective:** Build the Streamlit "Apply" tab with resume preview, optimization controls, and inline editing.

#### **Step 6.1: Add Database Update Function**

```python
# In db_utils.py

def update_resume_sections(resume_id: int, updated_data: Dict[str, Any]) -> bool:
    """
    Update resume sections (experience, projects, summary) in database.

    Args:
        resume_id: ID of resume to update
        updated_data: Dict containing updated fields

    Returns:
        True if update successful
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
```

#### **Step 6.2: Apply Tab Layout**

```python
# In app.py - Add new tab

import matcher
import optimizer
import resume_renderer

# Update tabs
tab1, tab2, tab3, tab4 = st.tabs(["Job Ingestion", "Resume Analysis", "History", "Apply"])

# Initialize Apply tab session state
if 'apply_state' not in st.session_state:
    st.session_state.apply_state = {
        'match_results': None,      # List of MatchResult
        'selected_resume_idx': 0,   # Which resume from results
        'selected_job_id': None,
        'optimizing_section': None,
        'rewrite_result': None,
        'editing_text': None,
        'pdf_bytes': None
    }

with tab4:
    st.markdown("### Apply: Resume Optimization")

    apply_state = st.session_state.apply_state

    # Step 1: Select a job to apply for
    jobs = db_utils.get_all_job_listings(limit=50)
    if not jobs:
        st.info("No saved jobs. Ingest a job listing first.")
        st.stop()

    job_options = {f"{j['job_title']} - {j['company']}": j['id'] for j in jobs}
    selected_job_name = st.selectbox("Select Job to Apply For", options=list(job_options.keys()))
    selected_job_id = job_options[selected_job_name]

    # Threshold configuration with explanation
    col_thresh, col_help = st.columns([2, 3])
    with col_thresh:
        threshold = st.slider(
            "Match Threshold",
            min_value=0.30,
            max_value=0.70,
            value=0.50,
            step=0.05,
        )
    with col_help:
        st.caption(
            "**Score Guide:**\n"
            "- 75%+ = Strong match\n"
            "- 65-74% = Good, minor tweaks\n"
            "- 50-64% = Moderate, optimize recommended\n"
            "- Below 50% = Weak, needs work"
        )

    # Step 2: Find matching resumes
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Find Best Resume Matches", type="primary"):
            with st.spinner("Analyzing resumes..."):
                results = matcher.find_top_matching_resumes(
                    selected_job_id,
                    limit=5,
                    threshold=threshold
                )
                if results:
                    apply_state['match_results'] = results
                    apply_state['selected_resume_idx'] = 0
                    apply_state['selected_job_id'] = selected_job_id
                    apply_state['optimizing_section'] = None
                    apply_state['rewrite_result'] = None
                    apply_state['pdf_bytes'] = None
                else:
                    st.warning("No resumes found. Upload a resume first.")

    with col_btn2:
        if st.button("Clear Results"):
            apply_state['match_results'] = None
            apply_state['optimizing_section'] = None
            apply_state['rewrite_result'] = None
            st.rerun()

    # Display results
    if apply_state['match_results']:
        results = apply_state['match_results']
        job = db_utils.get_job_listing_by_id(apply_state['selected_job_id'])

        if not job:
            st.error("Could not load job data.")
            st.stop()

        # Resume selector if multiple matches
        if len(results) > 1:
            resume_names = [f"{r.resume_name} ({r.overall_score:.0%})" for r in results]
            selected_idx = st.selectbox(
                "Select Resume",
                range(len(results)),
                format_func=lambda i: resume_names[i],
                index=apply_state['selected_resume_idx']
            )
            if selected_idx != apply_state['selected_resume_idx']:
                apply_state['selected_resume_idx'] = selected_idx
                apply_state['optimizing_section'] = None
                apply_state['rewrite_result'] = None
                st.rerun()

        match = results[apply_state['selected_resume_idx']]
        resume = db_utils.get_resume_by_id(match.resume_id)

        if not resume:
            st.error("Could not load resume data.")
            st.stop()

        st.divider()

        # Display layout: Resume preview | Optimization panel
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### Resume Preview")
            st.caption(f"Showing: {resume.get('full_name', 'Resume')} (ID: {match.resume_id})")

            resume_html = resume_renderer.render_resume_html(
                resume,
                section_scores=match.score_dict(),
                threshold=threshold
            )
            st.components.v1.html(resume_html, height=700, scrolling=True)

        with col2:
            st.markdown("#### Optimization Panel")

            # Overall score with interpretation
            score_pct = match.overall_score * 100
            interpretation = matcher.get_score_interpretation(match.overall_score)

            if match.overall_score >= 0.75:
                st.success(f"Match Score: {score_pct:.0f}%")
            elif match.overall_score >= 0.65:
                st.info(f"Match Score: {score_pct:.0f}%")
            elif match.overall_score >= 0.50:
                st.warning(f"Match Score: {score_pct:.0f}%")
            else:
                st.error(f"Match Score: {score_pct:.0f}%")

            st.caption(interpretation)

            # Weak sections list
            st.markdown("**Sections to Improve:**")
            if match.weak_sections:
                for ws in match.weak_sections:
                    section_name = ws.section_type.title()
                    if ws.section_type in ('experience', 'project'):
                        section_name += f" #{ws.section_index + 1}"

                    if st.button(
                        f"Optimize: {section_name} ({ws.score:.0%})",
                        key=f"opt_{ws.section_type}_{ws.section_index}"
                    ):
                        apply_state['optimizing_section'] = ws
                        apply_state['rewrite_result'] = None
                        st.rerun()
            else:
                st.success("All sections score above threshold!")

            # Optimization interface
            if apply_state['optimizing_section']:
                section = apply_state['optimizing_section']
                st.divider()
                st.markdown(f"**Optimizing: {section.section_type.title()}**")

                # Show current content
                with st.expander("Current Content", expanded=False):
                    if section.section_type in ('experience', 'project'):
                        content = section.content
                        st.write(f"**{content.get('title') or content.get('name', 'Unknown')}**")
                        for bullet in content.get('description', []):
                            st.write(f"- {bullet}")
                    else:
                        st.write(section.content)

                # Generate or show rewrite
                if apply_state['rewrite_result'] is None:
                    if st.button("Generate Suggestion", type="primary"):
                        with st.spinner("Generating AI suggestion..."):
                            result = optimizer.generate_section_rewrite(
                                section.section_type,
                                section.content,
                                job['description']
                            )
                            apply_state['rewrite_result'] = result
                            apply_state['editing_text'] = result
                            st.rerun()
                else:
                    st.markdown("**Suggested Rewrite:**")

                    # Editable text area
                    edited_text = st.text_area(
                        "Edit suggestion (or accept as-is):",
                        value=apply_state['editing_text'],
                        height=200,
                        key="edit_textarea"
                    )
                    apply_state['editing_text'] = edited_text

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("Accept", type="primary"):
                            success = _apply_rewrite(
                                section,
                                edited_text,
                                match.resume_id
                            )
                            if success:
                                # Re-generate embeddings for updated section
                                updated_resume = db_utils.get_resume_by_id(match.resume_id)
                                db_utils.store_resume_embeddings(match.resume_id, updated_resume)

                                st.success("Applied and re-indexed!")
                                apply_state['rewrite_result'] = None
                                apply_state['optimizing_section'] = None
                                apply_state['match_results'] = None  # Force rematch
                                st.rerun()
                            else:
                                st.error("Failed to apply changes.")

                    with col_b:
                        if st.button("Regenerate"):
                            apply_state['rewrite_result'] = None
                            st.rerun()

                    with col_c:
                        if st.button("Cancel"):
                            apply_state['optimizing_section'] = None
                            apply_state['rewrite_result'] = None
                            st.rerun()

            # Export section
            st.divider()
            st.markdown("#### Export")

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("Generate PDF"):
                    try:
                        pdf_bytes = resume_renderer.render_resume_pdf_bytes(resume)
                        apply_state['pdf_bytes'] = pdf_bytes
                        st.success("PDF ready!")
                    except ImportError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

                if apply_state.get('pdf_bytes'):
                    company = job.get('company', 'Company').replace(' ', '_')
                    st.download_button(
                        "Download PDF",
                        data=apply_state['pdf_bytes'],
                        file_name=f"Resume_{company}.pdf",
                        mime="application/pdf"
                    )

            with col_exp2:
                if st.button("Copy Plain Text"):
                    plain_text = resume_renderer.render_resume_plaintext(resume)
                    st.code(plain_text, language=None)

def _apply_rewrite(section, new_content: str, resume_id: int) -> bool:
    """
    Apply a rewrite to the resume in the database.

    Returns True if successful, False otherwise.
    """
    resume = db_utils.get_resume_by_id(resume_id)
    if not resume:
        return False

    try:
        if section.section_type == 'experience':
            idx = section.section_index
            # Bounds check
            if idx >= len(resume.get('experience', [])):
                return False
            bullets = [line.strip().lstrip('- ').lstrip('* ')
                       for line in new_content.strip().split('\n')
                       if line.strip()]
            resume['experience'][idx]['description'] = bullets
            return db_utils.update_resume_sections(resume_id, {'experience': resume['experience']})

        elif section.section_type == 'project':
            idx = section.section_index
            # Bounds check
            if idx >= len(resume.get('projects', [])):
                return False
            bullets = [line.strip().lstrip('- ').lstrip('* ')
                       for line in new_content.strip().split('\n')
                       if line.strip()]
            resume['projects'][idx]['description'] = bullets
            return db_utils.update_resume_sections(resume_id, {'projects': resume['projects']})

        elif section.section_type == 'summary':
            return db_utils.update_resume_sections(resume_id, {'summary': new_content.strip()})

        return False
    except Exception as e:
        print(f"Error applying rewrite: {e}")
        return False
```

**Definition of Done:**
- [ ] Apply tab added to main navigation
- [ ] Job selection dropdown functional
- [ ] Threshold slider with score interpretation guide
- [ ] Top matching resumes found via pgvector (not Python iteration)
- [ ] Resume selector when multiple matches exist
- [ ] Resume preview with three-tier section highlighting
- [ ] Weak sections listed with optimize buttons
- [ ] LLM rewrite generated on demand (with caching)
- [ ] Editable text area for user modifications
- [ ] Accept applies changes AND re-generates embeddings
- [ ] Bounds checking prevents index errors
- [ ] PDF export functional
- [ ] Plain text export for clipboard

---

### **4. Testing Strategy**

#### **Unit Tests**
- `test_text_utils.py` - Verify formatting functions
- `test_embeddings.py` - Verify embedding generation and similarity
- `test_matcher.py` - Test scoring with known resume/job pairs, verify SQL query
- `test_optimizer.py` - Mock LLM responses, verify prompt construction, test cache
- `test_renderer.py` - Verify HTML output structure and XSS escaping

#### **Integration Tests**
- Full pipeline: Job -> Match (via SQL) -> Score -> Optimize -> Re-embed -> Export
- Database round-trip for embeddings
- LLM cache hit/miss scenarios
- Bounds checking edge cases

#### **User Acceptance Testing**
- [ ] Select a job, find best resume matches (should be fast even with many resumes)
- [ ] Verify weak sections are correctly identified
- [ ] Generate and apply optimizations
- [ ] Verify score improves after optimization (embeddings refreshed)
- [ ] Export final PDF
- [ ] Verify PDF looks professional

---

### **5. Dependencies to Add**

```bash
pip install sentence-transformers numpy jinja2

# Optional for PDF export (requires system dependencies)
pip install weasyprint

# macOS system deps for WeasyPrint:
brew install cairo pango gdk-pixbuf libffi

# Ubuntu system deps:
sudo apt install libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0
```

Update `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

### **6. Cost Estimates**

| Component | Cost |
|-----------|------|
| Sentence-BERT embeddings | Free (local CPU, ~10-50ms each) |
| pgvector similarity search | Free (PostgreSQL) |
| OpenAI gpt-4o-mini per section | ~$0.001-0.003 |
| LLM cache hit | Free |
| Optimizing 5 sections (no cache) | ~$0.005-0.015 |

**Verdict:** Very affordable for personal use. Caching further reduces costs.

---

### **7. Key Improvements Over Original Plan**

| Issue | Original | Fixed |
|-------|----------|-------|
| Matching scalability | Python iteration over all resumes | pgvector SQL with HNSW index |
| Stale embeddings | Not addressed | Re-embed after optimization |
| Code duplication | Formatting in db_utils and matcher | Centralized in text_utils.py |
| LLM caching | Not implemented | DB-backed response cache |
| Index bounds | No checking | Explicit bounds validation |
| Score interpretation | Raw percentages | User-friendly guidance |
| Prompt safety | Basic | Explicit anti-hallucination rules |
| Threshold | Single arbitrary value | Three-tier with explanations |

---

### **8. Future Enhancements**

- **Cover Letter Generation:** Extend optimizer to generate matching cover letters
- **A/B Testing:** Track which optimizations lead to interview callbacks
- **Resume Versioning:** Save multiple versions of optimized resumes per job
- **Section-to-Section Matching:** Compare experience to responsibilities, skills to requirements
- **Batch Apply:** Queue multiple job applications with their optimized resumes
- **LaTeX Templates:** Add professional LaTeX resume templates as alternative to HTML

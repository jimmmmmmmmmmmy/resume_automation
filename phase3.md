#### **Phase 3: Resume-Job Matching & Optimization Pipeline**

**1. Overview & Objectives**

*   **1.1. Vision:** To create an intelligent pipeline that takes a saved job listing, finds the best-matching resume from the database, scores each resume section against the job requirements, and provides an interactive "Apply" interface where users can review and optimize underperforming sections with LLM-generated suggestions—similar to Cursor's command-K inline editing experience.

*   **1.2. Key Objectives for Phase 3:**
    *   Implement semantic vectorization for job listings and resume sections
    *   Build a matching algorithm that scores resumes against jobs and ranks them
    *   Create section-level scoring to identify weak areas needing improvement
    *   Integrate LLM API calls to generate targeted rewrite suggestions
    *   Build an "Apply" tab with a PDF-like resume preview and inline editing UI

**2. System Architecture & Design**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLY TAB UI                                    │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │      Resume PDF Preview         │  │      Optimization Panel          │ │
│  │  ┌───────────────────────────┐  │  │  ┌────────────────────────────┐ │ │
│  │  │ Contact Info              │  │  │  │ Match Score: 78%           │ │ │
│  │  ├───────────────────────────┤  │  │  ├────────────────────────────┤ │ │
│  │  │ Summary [Score: 0.85]     │  │  │  │ Weak Sections:             │ │ │
│  │  ├───────────────────────────┤  │  │  │  - Experience #2 (0.62)    │ │ │
│  │  │ Experience                │  │  │  │  - Projects #1 (0.58)      │ │ │
│  │  │  └─ Role 1 [0.91]        │  │  │  ├────────────────────────────┤ │ │
│  │  │  └─ Role 2 [0.62] ← RED  │  │  │  │ [Optimize Selected]        │ │ │
│  │  ├───────────────────────────┤  │  │  └────────────────────────────┘ │ │
│  │  │ Projects                  │  │  │                                  │ │
│  │  │  └─ Proj 1 [0.58] ← RED  │  │  │  Suggested Rewrites:            │ │
│  │  │  └─ Proj 2 [0.88]        │  │  │  ┌────────────────────────────┐ │ │
│  │  ├───────────────────────────┤  │  │  │ Option 1: "Led cross..."  │ │ │
│  │  │ Skills [0.82]             │  │  │  │ [Accept] [Edit] [Reject]  │ │ │
│  │  └───────────────────────────┘  │  │  ├────────────────────────────┤ │ │
│  └─────────────────────────────────┘  │  │ Option 2: "Spearheaded..." │ │ │
│                                        │  │ [Accept] [Edit] [Reject]  │ │ │
│                                        │  └────────────────────────────┘ │ │
│                                        └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

*   **New Components:**
    *   `embeddings.py` - Vectorization service using Sentence-BERT
    *   `matcher.py` - Resume-job matching and scoring logic
    *   `optimizer.py` - LLM-powered section rewriting
    *   `resume_renderer.py` - PDF/HTML resume preview generation

**3. Detailed Task Breakdown**

---

### **Task 1: Embedding Infrastructure**

**Objective:** Set up the vectorization pipeline using Sentence-BERT for semantic similarity comparisons.

#### **Step 1.1: Create the Embeddings Module**

```python
# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

# Load model once at module level
MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text: str) -> np.ndarray:
    """Generate embedding vector for a single text."""
    return MODEL.encode(text, normalize_embeddings=True)

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embedding vectors for multiple texts."""
    return MODEL.encode(texts, normalize_embeddings=True)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2))
```

#### **Step 1.2: Database Schema Updates**

Add vector columns to store embeddings:

```sql
-- scripts/migrations/003_add_embeddings.sql

-- Add embedding column to job_listings
ALTER TABLE job_listings
ADD COLUMN IF NOT EXISTS description_embedding vector(384);

-- Add embedding columns to resumes for each section
ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS summary_embedding vector(384),
ADD COLUMN IF NOT EXISTS skills_embedding vector(384),
ADD COLUMN IF NOT EXISTS experience_embeddings JSONB,  -- Array of {index, embedding}
ADD COLUMN IF NOT EXISTS projects_embeddings JSONB;    -- Array of {index, embedding}

-- Create index for similarity search
CREATE INDEX IF NOT EXISTS idx_job_embedding
ON job_listings USING ivfflat (description_embedding vector_cosine_ops)
WITH (lists = 100);
```

#### **Step 1.3: Embed on Save**

Update `db_utils.py` to generate and store embeddings when saving jobs/resumes:

```python
def insert_job_listing_with_embedding(job: JobListing, job_hash: str) -> bool:
    """Insert job with its embedding vector."""
    embedding = embed_text(job.description)
    # ... insert with embedding
```

**Definition of Done:**
- [ ] `embeddings.py` module created with embed/similarity functions
- [ ] Migration script adds vector columns to database
- [ ] Jobs and resumes store embeddings on save
- [ ] Unit tests verify embedding generation and similarity calculations

---

### **Task 2: Resume-Job Matching Algorithm**

**Objective:** Build the logic to find the best-matching resume for a given job listing.

#### **Step 2.1: Create the Matcher Module**

```python
# matcher.py
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from embeddings import embed_text, cosine_similarity
import db_utils

@dataclass
class MatchResult:
    """Result of matching a resume against a job."""
    resume_id: int
    overall_score: float
    section_scores: Dict[str, float]  # section_name -> score
    weak_sections: List[Dict[str, Any]]  # Sections below threshold

def match_resume_to_job(resume_id: int, job_id: int, threshold: float = 0.75) -> MatchResult:
    """
    Score a single resume against a job listing.

    Returns section-by-section scores and identifies weak areas.
    """
    job = db_utils.get_job_listing_by_id(job_id)
    resume = db_utils.get_resume_by_id(resume_id)

    job_embedding = embed_text(job['description'])

    section_scores = {}
    weak_sections = []

    # Score summary
    if resume.get('summary'):
        score = cosine_similarity(embed_text(resume['summary']), job_embedding)
        section_scores['summary'] = score
        if score < threshold:
            weak_sections.append({
                'section': 'summary',
                'content': resume['summary'],
                'score': score
            })

    # Score each experience entry
    for i, exp in enumerate(resume.get('experience', [])):
        exp_text = _format_experience_for_embedding(exp)
        score = cosine_similarity(embed_text(exp_text), job_embedding)
        section_scores[f'experience_{i}'] = score
        if score < threshold:
            weak_sections.append({
                'section': 'experience',
                'index': i,
                'content': exp,
                'score': score
            })

    # Score each project
    for i, proj in enumerate(resume.get('projects', [])):
        proj_text = _format_project_for_embedding(proj)
        score = cosine_similarity(embed_text(proj_text), job_embedding)
        section_scores[f'project_{i}'] = score
        if score < threshold:
            weak_sections.append({
                'section': 'project',
                'index': i,
                'content': proj,
                'score': score
            })

    # Score skills (aggregate)
    if resume.get('skills'):
        skills_text = ', '.join(resume['skills'])
        section_scores['skills'] = cosine_similarity(embed_text(skills_text), job_embedding)

    # Calculate overall score (weighted average)
    overall_score = _calculate_weighted_score(section_scores)

    return MatchResult(
        resume_id=resume_id,
        overall_score=overall_score,
        section_scores=section_scores,
        weak_sections=sorted(weak_sections, key=lambda x: x['score'])
    )

def find_best_resume(job_id: int) -> Tuple[int, MatchResult]:
    """
    Find the best matching resume for a job from all saved resumes.

    Returns (resume_id, MatchResult) for the top match.
    """
    resumes = db_utils.get_all_resumes(limit=100)

    best_match = None
    best_score = -1

    for resume in resumes:
        result = match_resume_to_job(resume['id'], job_id)
        if result.overall_score > best_score:
            best_score = result.overall_score
            best_match = result

    return best_match
```

#### **Step 2.2: Scoring Weights Configuration**

```python
# In matcher.py or config.py

SECTION_WEIGHTS = {
    'summary': 0.15,
    'experience': 0.40,  # Divided among entries
    'projects': 0.25,    # Divided among entries
    'skills': 0.20
}

SCORE_THRESHOLD = 0.75  # Sections below this are flagged for optimization
```

**Definition of Done:**
- [ ] `matcher.py` module created with matching logic
- [ ] Section-level scoring implemented for all resume sections
- [ ] Weak sections correctly identified based on threshold
- [ ] `find_best_resume()` returns the highest-scoring resume
- [ ] Unit tests with sample job/resume pairs

---

### **Task 3: LLM-Powered Section Optimization**

**Objective:** Use OpenAI API to generate improved versions of weak resume sections.

#### **Step 3.1: Create the Optimizer Module**

```python
# optimizer.py
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_section_rewrites(
    section_content: Dict[str, Any],
    job_description: str,
    num_options: int = 3
) -> List[str]:
    """
    Generate multiple rewrite options for a resume section.

    Args:
        section_content: The current section content (experience/project dict)
        job_description: The target job description
        num_options: Number of alternative versions to generate

    Returns:
        List of rewritten versions
    """

    section_type = section_content.get('section', 'experience')
    content = section_content.get('content', {})

    if section_type == 'experience':
        prompt = _build_experience_rewrite_prompt(content, job_description)
    elif section_type == 'project':
        prompt = _build_project_rewrite_prompt(content, job_description)
    elif section_type == 'summary':
        prompt = _build_summary_rewrite_prompt(content, job_description)
    else:
        return []

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        n=num_options,
        temperature=0.7
    )

    return [choice.message.content for choice in response.choices]

OPTIMIZER_SYSTEM_PROMPT = """You are an expert resume writer and career coach.
Your task is to rewrite resume sections to better match a target job description
while maintaining truthfulness and the candidate's authentic experience.

Guidelines:
- Use strong action verbs (Led, Developed, Implemented, Optimized)
- Include quantifiable metrics where possible
- Mirror keywords from the job description naturally
- Keep bullet points concise (1-2 lines)
- Maintain the factual accuracy of the original content
- Focus on impact and results, not just responsibilities
"""

def _build_experience_rewrite_prompt(exp: Dict, job_desc: str) -> str:
    bullets = '\n'.join(f"- {b}" for b in exp.get('description', []))
    return f"""
Rewrite this work experience section to better align with the target job.

CURRENT EXPERIENCE:
Title: {exp.get('title', '')}
Company: {exp.get('company', '')}
Bullet Points:
{bullets}

TARGET JOB DESCRIPTION:
{job_desc[:2000]}

Provide an improved version of the bullet points that:
1. Uses keywords from the job description naturally
2. Emphasizes relevant skills and achievements
3. Quantifies impact where possible

Return ONLY the rewritten bullet points, one per line starting with "-".
"""
```

#### **Step 3.2: Batch Optimization with Cost Control**

```python
# In optimizer.py

def optimize_weak_sections(
    weak_sections: List[Dict[str, Any]],
    job_description: str,
    max_sections: int = 5
) -> Dict[str, List[str]]:
    """
    Batch optimize multiple weak sections.

    Args:
        weak_sections: List of sections to optimize (sorted by score ascending)
        job_description: Target job description
        max_sections: Maximum sections to optimize (cost control)

    Returns:
        Dict mapping section identifiers to list of rewrite options
    """
    results = {}

    # Only optimize the weakest sections up to max_sections
    for section in weak_sections[:max_sections]:
        section_key = f"{section['section']}_{section.get('index', 0)}"
        results[section_key] = generate_section_rewrites(
            section,
            job_description,
            num_options=3
        )

    return results
```

**Definition of Done:**
- [ ] `optimizer.py` module created with LLM integration
- [ ] Prompts tuned for experience, project, and summary sections
- [ ] Multiple rewrite options generated per section
- [ ] Cost controls in place (max sections, caching)
- [ ] Error handling for API failures

---

### **Task 4: Resume Preview Renderer**

**Objective:** Create a PDF-like visual preview of the resume that can highlight sections and display inline edits.

#### **Step 4.1: HTML-Based Resume Template**

```python
# resume_renderer.py
from typing import Dict, List, Any, Optional
from jinja2 import Template

def render_resume_html(
    resume: Dict[str, Any],
    section_scores: Optional[Dict[str, float]] = None,
    threshold: float = 0.75,
    highlighted_section: Optional[str] = None
) -> str:
    """
    Render resume as styled HTML with optional score highlighting.

    Args:
        resume: Resume data dictionary
        section_scores: Optional dict of section -> score for coloring
        threshold: Score below which sections are highlighted red
        highlighted_section: Currently selected section for editing

    Returns:
        HTML string
    """

    template = Template(RESUME_HTML_TEMPLATE)

    # Prepare section styling based on scores
    section_styles = {}
    if section_scores:
        for section, score in section_scores.items():
            if score < threshold:
                section_styles[section] = 'weak-section'  # Red highlight
            elif score >= 0.85:
                section_styles[section] = 'strong-section'  # Green highlight
            else:
                section_styles[section] = ''

    return template.render(
        resume=resume,
        section_styles=section_styles,
        highlighted_section=highlighted_section,
        section_scores=section_scores
    )

RESUME_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        font-family: 'Georgia', serif;
        max-width: 8.5in;
        margin: 0 auto;
        padding: 0.5in;
        background: white;
        color: #333;
        line-height: 1.4;
    }

    .header {
        text-align: center;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    .header h1 {
        margin: 0;
        font-size: 24pt;
    }

    .contact-info {
        font-size: 10pt;
        color: #666;
    }

    .section {
        margin-bottom: 15px;
        padding: 8px;
        border-radius: 4px;
        transition: all 0.3s ease;
    }

    .section-header {
        font-size: 12pt;
        font-weight: bold;
        text-transform: uppercase;
        border-bottom: 1px solid #333;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }

    .score-badge {
        font-size: 9pt;
        padding: 2px 6px;
        border-radius: 10px;
        font-weight: normal;
    }

    .weak-section {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }

    .weak-section .score-badge {
        background-color: #f44336;
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
    }

    .entry-header {
        display: flex;
        justify-content: space-between;
        font-weight: bold;
    }

    .entry-subheader {
        font-style: italic;
        color: #666;
        font-size: 10pt;
    }

    ul {
        margin: 5px 0;
        padding-left: 20px;
    }

    li {
        margin-bottom: 3px;
        font-size: 10pt;
    }

    .skills-list {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }

    .skill-tag {
        background: #e0e0e0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 9pt;
    }

    .edit-overlay {
        position: absolute;
        right: 5px;
        top: 5px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s;
    }

    .section:hover .edit-overlay {
        opacity: 1;
    }
</style>
</head>
<body>
    <div class="header">
        <h1>{{ resume.full_name or 'Your Name' }}</h1>
        <div class="contact-info">
            {{ resume.email or '' }} | {{ resume.phone or '' }} | {{ resume.location or '' }}
        </div>
    </div>

    {% if resume.summary %}
    <div class="section {{ section_styles.get('summary', '') }} {% if highlighted_section == 'summary' %}highlighted{% endif %}"
         data-section="summary">
        <div class="section-header">
            Summary
            {% if section_scores and section_scores.get('summary') %}
            <span class="score-badge">{{ "%.0f"|format(section_scores.summary * 100) }}%</span>
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
                <span>{{ exp.title }}</span>
                <span>{{ exp.dates }}</span>
            </div>
            <div class="entry-subheader">{{ exp.company }}</div>
            <ul>
            {% for bullet in exp.description %}
                <li>{{ bullet }}</li>
            {% endfor %}
            </ul>
            {% if section_scores and section_scores.get('experience_' ~ loop.index0) %}
            <span class="score-badge">{{ "%.0f"|format(section_scores['experience_' ~ loop.index0] * 100) }}%</span>
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
                <span>{{ proj.name }}</span>
            </div>
            {% if proj.technologies %}
            <div class="skills-list" style="margin-bottom: 5px;">
                {% for tech in proj.technologies %}
                <span class="skill-tag">{{ tech }}</span>
                {% endfor %}
            </div>
            {% endif %}
            <ul>
            {% for bullet in proj.description %}
                <li>{{ bullet }}</li>
            {% endfor %}
            </ul>
            {% if section_scores and section_scores.get('project_' ~ loop.index0) %}
            <span class="score-badge">{{ "%.0f"|format(section_scores['project_' ~ loop.index0] * 100) }}%</span>
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
            {% if section_scores and section_scores.get('skills') %}
            <span class="score-badge">{{ "%.0f"|format(section_scores.skills * 100) }}%</span>
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
                <span>{{ edu.degree }}</span>
                <span>{{ edu.dates }}</span>
            </div>
            <div class="entry-subheader">{{ edu.institution }}</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
"""
```

#### **Step 4.2: PDF Generation Option (using WeasyPrint or similar)**

```python
# In resume_renderer.py

def render_resume_pdf(resume: Dict[str, Any], output_path: str) -> str:
    """
    Render resume as PDF file.

    Uses WeasyPrint to convert HTML to PDF.
    """
    from weasyprint import HTML

    html_content = render_resume_html(resume)
    HTML(string=html_content).write_pdf(output_path)
    return output_path
```

**Definition of Done:**
- [ ] `resume_renderer.py` module created
- [ ] HTML template renders all resume sections
- [ ] Score-based color highlighting working
- [ ] Section selection highlighting for editing
- [ ] PDF export option functional

---

### **Task 5: Apply Tab UI Implementation**

**Objective:** Build the Streamlit "Apply" tab with resume preview, optimization controls, and inline editing.

#### **Step 5.1: Apply Tab Layout**

```python
# In app.py - Add new tab

tab1, tab2, tab3, tab4 = st.tabs(["Job Ingestion", "Resume Analysis", "History", "Apply"])

with tab4:
    st.markdown("### Apply: Resume Optimization")

    # Step 1: Select a job to apply for
    jobs = db_utils.get_all_job_listings(limit=50)
    if not jobs:
        st.info("No saved jobs. Ingest a job listing first.")
        st.stop()

    job_options = {f"{j['job_title']} - {j['company']}": j['id'] for j in jobs}
    selected_job_name = st.selectbox("Select Job to Apply For", options=list(job_options.keys()))
    selected_job_id = job_options[selected_job_name]

    # Step 2: Find or select best-matching resume
    if st.button("Find Best Resume Match"):
        with st.spinner("Analyzing resumes..."):
            match_result = matcher.find_best_resume(selected_job_id)
            st.session_state.match_result = match_result
            st.session_state.selected_job_id = selected_job_id

    if 'match_result' in st.session_state:
        match = st.session_state.match_result
        resume = db_utils.get_resume_by_id(match.resume_id)
        job = db_utils.get_job_listing_by_id(st.session_state.selected_job_id)

        # Display layout
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### Resume Preview")
            # Render resume HTML with score highlighting
            resume_html = resume_renderer.render_resume_html(
                resume,
                section_scores=match.section_scores,
                threshold=0.75
            )
            st.components.v1.html(resume_html, height=800, scrolling=True)

        with col2:
            st.markdown("#### Optimization Panel")

            # Overall score
            score_color = "green" if match.overall_score >= 0.8 else "orange" if match.overall_score >= 0.7 else "red"
            st.markdown(f"**Match Score:** :{score_color}[{match.overall_score:.0%}]")

            # Threshold slider
            threshold = st.slider("Optimization Threshold", 0.5, 0.95, 0.75, 0.05)

            # Weak sections list
            if match.weak_sections:
                st.markdown("**Sections Needing Improvement:**")
                for ws in match.weak_sections:
                    section_name = f"{ws['section'].title()}"
                    if 'index' in ws:
                        section_name += f" #{ws['index'] + 1}"

                    if st.button(f"Optimize: {section_name} ({ws['score']:.0%})",
                                key=f"opt_{ws['section']}_{ws.get('index', 0)}"):
                        st.session_state.optimizing_section = ws
            else:
                st.success("All sections score above threshold!")

            # Optimization results
            if 'optimizing_section' in st.session_state:
                section = st.session_state.optimizing_section
                st.markdown("---")
                st.markdown(f"**Optimizing: {section['section'].title()}**")

                if 'rewrite_options' not in st.session_state:
                    with st.spinner("Generating suggestions..."):
                        options = optimizer.generate_section_rewrites(
                            section,
                            job['description'],
                            num_options=3
                        )
                        st.session_state.rewrite_options = options

                for i, option in enumerate(st.session_state.rewrite_options):
                    with st.expander(f"Option {i + 1}", expanded=(i == 0)):
                        st.markdown(option)
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button("Accept", key=f"accept_{i}"):
                                # Apply the rewrite
                                _apply_rewrite(section, option)
                                st.success("Applied!")
                                del st.session_state.rewrite_options
                                del st.session_state.optimizing_section
                                st.rerun()
                        with col_b:
                            if st.button("Edit", key=f"edit_{i}"):
                                st.session_state.editing_option = option
                        with col_c:
                            if st.button("Reject", key=f"reject_{i}"):
                                pass  # Just close/move on

                if 'editing_option' in st.session_state:
                    edited = st.text_area("Edit suggestion:",
                                         value=st.session_state.editing_option,
                                         height=200)
                    if st.button("Apply Edited Version"):
                        _apply_rewrite(section, edited)
                        st.success("Applied!")
                        del st.session_state.editing_option
                        st.rerun()
```

#### **Step 5.2: Inline Editing (Cursor-like Command-K)**

For a more advanced inline editing experience, consider using custom Streamlit components:

```python
# For future enhancement: Custom component for inline editing
# This would render the resume and allow clicking on sections to edit

# Simpler approach: Use callbacks with section selection
def _apply_rewrite(section: Dict, new_content: str):
    """Apply a rewrite to the resume in the database."""
    resume_id = st.session_state.match_result.resume_id
    resume = db_utils.get_resume_by_id(resume_id)

    if section['section'] == 'experience':
        idx = section['index']
        # Parse new_content (bullet points) back into list
        bullets = [line.strip().lstrip('- ') for line in new_content.strip().split('\n') if line.strip()]
        resume['experience'][idx]['description'] = bullets

    elif section['section'] == 'project':
        idx = section['index']
        bullets = [line.strip().lstrip('- ') for line in new_content.strip().split('\n') if line.strip()]
        resume['projects'][idx]['description'] = bullets

    elif section['section'] == 'summary':
        resume['summary'] = new_content.strip()

    # Update in database
    db_utils.update_resume_sections(resume_id, resume)
```

**Definition of Done:**
- [ ] Apply tab added to main navigation
- [ ] Job selection dropdown functional
- [ ] Best resume matching with score display
- [ ] Resume preview with section highlighting
- [ ] Weak sections listed with optimize buttons
- [ ] LLM rewrite options displayed
- [ ] Accept/Edit/Reject workflow complete
- [ ] Changes persisted to database

---

### **Task 6: PDF Export for Final Submission**

**Objective:** Generate a polished PDF resume for manual job application submission.

#### **Step 6.1: Export Button**

```python
# In Apply tab
st.markdown("---")
st.markdown("#### Export")

col1, col2 = st.columns(2)
with col1:
    if st.button("Download Optimized Resume (PDF)"):
        pdf_path = resume_renderer.render_resume_pdf(
            resume,
            output_path=f"/tmp/resume_{resume['id']}_{job['id']}.pdf"
        )
        with open(pdf_path, 'rb') as f:
            st.download_button(
                "Download PDF",
                data=f,
                file_name=f"Resume_{job['company']}.pdf",
                mime="application/pdf"
            )

with col2:
    if st.button("Copy to Clipboard (Plain Text)"):
        plain_text = resume_renderer.render_resume_plaintext(resume)
        st.code(plain_text)
        st.info("Copy the text above")
```

---

### **4. Testing Strategy**

#### **Unit Tests**
- `test_embeddings.py` - Verify embedding generation and similarity calculations
- `test_matcher.py` - Test scoring logic with known resume/job pairs
- `test_optimizer.py` - Mock LLM responses, verify prompt construction
- `test_renderer.py` - Verify HTML output structure

#### **Integration Tests**
- Full pipeline: Job → Match → Score → Optimize → Export
- Database round-trip for embeddings
- LLM API integration (with rate limiting)

#### **User Acceptance Testing**
- [ ] Select a job, find best resume match
- [ ] Verify weak sections are correctly identified
- [ ] Generate and apply optimizations
- [ ] Export final PDF
- [ ] Verify PDF looks professional and is ATS-friendly

---

### **5. Dependencies to Add**

```bash
pip install sentence-transformers numpy jinja2 weasyprint
```

Update `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

### **6. Future Enhancements**

- **Cover Letter Generation:** Extend optimizer to generate matching cover letters
- **A/B Testing:** Track which optimizations lead to interview callbacks
- **Resume Versioning:** Save multiple versions of optimized resumes
- **Batch Apply:** Queue multiple job applications with their optimized resumes
- **LaTeX Templates:** Add professional LaTeX resume templates as alternative to HTML
- **Custom Component:** Build a Streamlit custom component for true inline editing

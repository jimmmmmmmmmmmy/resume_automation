"""
Resume Preview Renderer Module

This module creates PDF-like visual previews of resumes with section highlighting
based on match scores.

Features:
- HTML template with professional styling
- Three-tier color highlighting (red/orange/green)
- Plain text export for clipboard
- PDF export (requires WeasyPrint)
"""

from typing import Dict, List, Any, Optional
import html

from jinja2 import Template


def render_resume_html(
    resume: Dict[str, Any],
    section_scores: Optional[Dict[str, float]] = None,
    threshold: float = 0.50,
    highlighted_section: Optional[str] = None,
    matching_skills: Optional[List[str]] = None
) -> str:
    """
    Render resume as styled HTML with optional score highlighting.

    Args:
        resume: Resume data dictionary.
        section_scores: Optional dict of section -> score for coloring.
        threshold: Score below which sections are highlighted red (default 0.50).
        highlighted_section: Currently selected section for editing.
        matching_skills: Optional list of skills that match job requirements.

    Returns:
        HTML string safe for embedding in Streamlit.
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

    # Normalize matching skills for comparison
    matching_skills_normalized = set()
    if matching_skills:
        matching_skills_normalized = {s.lower().strip() for s in matching_skills}

    return template.render(
        resume=safe_resume,
        section_styles=section_styles,
        highlighted_section=highlighted_section,
        section_scores=section_scores or {},
        matching_skills=matching_skills_normalized
    )


def _escape_resume_content(resume: Dict[str, Any]) -> Dict[str, Any]:
    """
    Escape HTML in all string fields to prevent XSS.

    Args:
        resume: Resume dictionary with potentially unsafe content.

    Returns:
        Resume dictionary with all strings HTML-escaped.
    """
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
    """
    Render resume as plain text for clipboard copying.

    Args:
        resume: Resume data dictionary.

    Returns:
        Plain text representation of the resume.
    """
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
    experience = resume.get('experience', []) or []
    if experience:
        lines.append('EXPERIENCE')
        for exp in experience:
            if not exp:
                continue
            title = exp.get('title', '')
            company = exp.get('company', '')
            lines.append(f"{title} - {company}")
            if exp.get('dates'):
                lines.append(exp['dates'])
            for bullet in exp.get('description', []) or []:
                lines.append(f"  - {bullet}")
            lines.append('')

    # Projects
    projects = resume.get('projects', []) or []
    if projects:
        lines.append('PROJECTS')
        for proj in projects:
            if not proj:
                continue
            lines.append(proj.get('name', ''))
            tech = proj.get('technologies', []) or []
            if tech:
                lines.append(f"Technologies: {', '.join(tech)}")
            for bullet in proj.get('description', []) or []:
                lines.append(f"  - {bullet}")
            lines.append('')

    # Skills
    skills = resume.get('skills', []) or []
    if skills:
        lines.append('SKILLS')
        lines.append(', '.join(skills))
        lines.append('')

    # Education
    education = resume.get('education', []) or []
    if education:
        lines.append('EDUCATION')
        for edu in education:
            if not edu:
                continue
            degree = edu.get('degree', 'Degree')
            institution = edu.get('institution', 'Institution')
            lines.append(f"{degree} - {institution}")
            if edu.get('dates'):
                lines.append(edu['dates'])
        lines.append('')

    return '\n'.join(lines)


# HTML template for resume preview
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
        padding: 8px;
        padding-right: 50px;
        border-radius: 3px;
        position: relative;
    }

    .entry-score {
        position: absolute;
        right: 4px;
        top: 4px;
        font-size: 9pt;
        padding: 2px 6px;
        border-radius: 8px;
        font-weight: 500;
    }

    .entry-score.weak {
        background-color: #f44336;
        color: white;
    }

    .entry-score.moderate {
        background-color: #ff9800;
        color: white;
    }

    .entry-score.strong {
        background-color: #4caf50;
        color: white;
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

    .skill-tag.matched {
        background: #c8e6c9;
        color: #2e7d32;
        font-weight: 500;
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
        {% if exp %}
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
            {% set exp_score = section_scores.get('experience_' ~ loop.index0, 0) %}
            {% if exp_score %}
            <span class="entry-score {% if exp_score < 0.5 %}weak{% elif exp_score < 0.7 %}moderate{% else %}strong{% endif %}">{{ "%.0f"|format(exp_score * 100) }}%</span>
            {% endif %}
        </div>
        {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    {% if resume.projects %}
    <div class="section">
        <div class="section-header">Projects</div>
        {% for proj in resume.projects %}
        {% if proj %}
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
            {% set proj_score = section_scores.get('project_' ~ loop.index0, 0) %}
            {% if proj_score %}
            <span class="entry-score {% if proj_score < 0.5 %}weak{% elif proj_score < 0.7 %}moderate{% else %}strong{% endif %}">{{ "%.0f"|format(proj_score * 100) }}%</span>
            {% endif %}
        </div>
        {% endif %}
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
            <span class="skill-tag{% if skill.lower() in matching_skills %} matched{% endif %}">{{ skill }}</span>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    {% if resume.education %}
    <div class="section">
        <div class="section-header">Education</div>
        {% for edu in resume.education %}
        {% if edu %}
        <div class="entry">
            <div class="entry-header">
                <span>{{ edu.degree or 'Degree' }}</span>
                <span>{{ edu.dates or '' }}</span>
            </div>
            <div class="entry-subheader">{{ edu.institution or '' }}</div>
        </div>
        {% endif %}
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

    Args:
        resume: Resume data dictionary.
        output_path: Path where PDF will be written.

    Returns:
        Path to generated PDF.

    Raises:
        ImportError: If WeasyPrint is not installed.
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
    """
    Render resume as PDF bytes (for Streamlit download button).

    Args:
        resume: Resume data dictionary.

    Returns:
        PDF file as bytes.

    Raises:
        ImportError: If WeasyPrint is not installed.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError("WeasyPrint not installed")

    html_content = render_resume_html(resume, section_scores=None)
    return HTML(string=html_content).write_pdf()


def check_pdf_available() -> bool:
    """
    Check if PDF generation is available.

    Returns:
        True if WeasyPrint can be imported, False otherwise.
    """
    try:
        from weasyprint import HTML
        return True
    except ImportError:
        return False

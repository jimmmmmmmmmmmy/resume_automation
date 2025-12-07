"""
Shared text formatting utilities for embeddings and display.

This module centralizes text formatting to prevent code duplication
between db_utils, matcher, and other modules.
"""

from typing import Dict, List, Any


def format_experience_text(exp: Dict[str, Any]) -> str:
    """
    Format experience entry as text for embedding.

    Used by both db_utils (on save) and matcher (on score).

    Args:
        exp: Experience dictionary with title, company, description, etc.

    Returns:
        Formatted text string suitable for embedding.
    """
    parts = [
        exp.get('title', ''),
        exp.get('company', ''),
    ]
    if exp.get('dates'):
        parts.append(exp['dates'])
    if exp.get('description'):
        if isinstance(exp['description'], list):
            parts.extend(exp['description'])
        else:
            parts.append(str(exp['description']))
    return ' '.join(filter(None, parts))


def format_project_text(proj: Dict[str, Any]) -> str:
    """
    Format project entry as text for embedding.

    Args:
        proj: Project dictionary with name, technologies, description, etc.

    Returns:
        Formatted text string suitable for embedding.
    """
    parts = [proj.get('name', '')]
    if proj.get('technologies'):
        if isinstance(proj['technologies'], list):
            parts.append(', '.join(proj['technologies']))
        else:
            parts.append(str(proj['technologies']))
    if proj.get('description'):
        if isinstance(proj['description'], list):
            parts.extend(proj['description'])
        else:
            parts.append(str(proj['description']))
    return ' '.join(filter(None, parts))


def format_skills_text(skills: List[str]) -> str:
    """
    Format skills list as text for embedding.

    Args:
        skills: List of skill strings.

    Returns:
        Comma-separated skills string.
    """
    if not skills:
        return ''
    if isinstance(skills, list):
        return ', '.join(str(s) for s in skills)
    return str(skills)


def format_education_text(edu: Dict[str, Any]) -> str:
    """
    Format education entry as text for embedding.

    Args:
        edu: Education dictionary with degree, institution, field, etc.

    Returns:
        Formatted text string suitable for embedding.
    """
    parts = [
        edu.get('degree', ''),
        edu.get('institution', ''),
        edu.get('field', ''),
    ]
    if edu.get('dates'):
        parts.append(edu['dates'])
    return ' '.join(filter(None, parts))


def format_resume_full_text(resume: Dict[str, Any]) -> str:
    """
    Format entire resume as a single text for overall embedding.

    Args:
        resume: Full resume dictionary.

    Returns:
        Formatted text string of the entire resume.
    """
    parts = []

    # Add name and contact
    if resume.get('full_name'):
        parts.append(resume['full_name'])

    # Add summary
    if resume.get('summary'):
        parts.append(resume['summary'])

    # Add skills
    if resume.get('skills'):
        parts.append(format_skills_text(resume['skills']))

    # Add experience
    for exp in resume.get('experience', []):
        parts.append(format_experience_text(exp))

    # Add projects
    for proj in resume.get('projects', []):
        parts.append(format_project_text(proj))

    # Add education
    for edu in resume.get('education', []):
        parts.append(format_education_text(edu))

    return ' '.join(filter(None, parts))

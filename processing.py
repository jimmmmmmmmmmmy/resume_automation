"""
Information Extraction Service

This module provides a tiered extraction strategy for parsing job listing text:
1. Regex-based extraction for explicitly labeled information
2. spaCy NER for entity recognition in unstructured text

The module also provides hashing utilities for deduplication.
"""
import re
import hashlib
from typing import Optional

# Import shared data structure
from models import JobListing

# Try to load spaCy model
try:
    import spacy
    NLP_MODEL = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    print("Warning: spaCy model not available. Run: 'python -m spacy download en_core_web_sm'")
    NLP_MODEL = None


# --- REGEX PATTERNS FOR EXTRACTION ---

REGEX_PATTERNS = {
    'job_title': [
        re.compile(r"(?:Job\s*Title|Position|Role)\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
        re.compile(r"^(.+?)\s+at\s+.+$", re.MULTILINE),
    ],
    'company': [
        re.compile(r"(?:Company|Employer|Organization)\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
        re.compile(r"(?:at|@)\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:\n|$|\s+is)", re.MULTILINE),
    ],
    'location': [
        re.compile(r"(?:Location|Where|Office)\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
        re.compile(r"(Remote|Hybrid|On-?site)", re.IGNORECASE),
        re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2})", re.MULTILINE),  # City, ST format
    ],
    'apply_url': [
        re.compile(r"(?:Apply\s*(?:here|now|at)?\s*[:\-]?\s*)?(https?://\S+)", re.IGNORECASE),
        re.compile(r"(https?://[^\s]+(?:careers|jobs|apply|lever|greenhouse|workday)[^\s]*)", re.IGNORECASE),
    ]
}


def extract_job_details(raw_text: str) -> JobListing:
    """
    Orchestrates the extraction process using a tiered approach.

    Args:
        raw_text: The raw job listing text to parse.

    Returns:
        A JobListing object with extracted fields populated.
    """
    job = JobListing(description=raw_text)

    # Track extraction methods in metadata
    extraction_methods = {}

    # Layer 1: Regex extraction
    job, regex_found = _extract_with_regex(raw_text, job)
    extraction_methods['regex'] = regex_found

    # Layer 2: spaCy NER extraction (fills gaps)
    job, spacy_found = _extract_with_spacy(raw_text, job)
    extraction_methods['spacy'] = spacy_found

    job.metadata['extraction_methods'] = extraction_methods

    return job


def _extract_with_regex(raw_text: str, job: JobListing) -> tuple[JobListing, list]:
    """
    Extract information using regex patterns.

    Returns the updated JobListing and a list of fields that were found.
    """
    found_fields = []

    for field, patterns in REGEX_PATTERNS.items():
        if getattr(job, field) is not None:
            continue

        for pattern in patterns:
            match = pattern.search(raw_text)
            if match:
                value = match.group(1).strip()
                # Clean up extracted value
                value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                if value and len(value) > 1:  # Avoid single-char matches
                    setattr(job, field, value)
                    found_fields.append(field)
                    break

    return job, found_fields


def _extract_with_spacy(raw_text: str, job: JobListing) -> tuple[JobListing, list]:
    """
    Extract information using spaCy Named Entity Recognition.

    Focuses on filling in company (ORG entities) and location (GPE entities)
    that weren't found by regex.

    Returns the updated JobListing and a list of fields that were found.
    """
    found_fields = []

    if NLP_MODEL is None:
        return job, found_fields

    # Only process if we're missing company or location
    if job.company is not None and job.location is not None:
        return job, found_fields

    # Process the text with spaCy
    doc = NLP_MODEL(raw_text[:5000])  # Limit text length for performance

    # Collect entities by type
    orgs = []
    locations = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.append(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC"):
            locations.append(ent.text.strip())

    # Use the first ORG entity as company if not already set
    if job.company is None and orgs:
        job.company = orgs[0]
        found_fields.append('company')

    # Use the first GPE/LOC entity as location if not already set
    if job.location is None and locations:
        job.location = locations[0]
        found_fields.append('location')

    return job, found_fields


def hash_description(text: str) -> str:
    """
    Normalize and hash a job description string using SHA-256.

    The normalization process:
    1. Converts to lowercase
    2. Removes all whitespace

    This creates a consistent fingerprint for deduplication.

    Args:
        text: The job description text to hash.

    Returns:
        A 64-character hexadecimal SHA-256 hash string.
    """
    # Normalize: lowercase and remove all whitespace
    normalized_text = "".join(text.lower().split())
    # Encode and hash
    encoded_text = normalized_text.encode('utf-8')
    return hashlib.sha256(encoded_text).hexdigest()


def clean_job_text(raw_text: str) -> str:
    """
    Clean and normalize job listing text for better extraction.

    Removes common boilerplate, normalizes whitespace, etc.

    Args:
        raw_text: The raw job listing text.

    Returns:
        Cleaned text ready for extraction.
    """
    # Remove common footer patterns
    boilerplate_patterns = [
        r"Equal\s+Opportunity\s+Employer.*",
        r"We\s+are\s+an?\s+equal\s+opportunity.*",
        r"All\s+qualified\s+applicants.*",
        r"E-Verify.*",
    ]

    cleaned = raw_text
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Normalize whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)

    return cleaned.strip()

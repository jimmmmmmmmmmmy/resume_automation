"""
Scraper Utilities Module

Utility functions for URL validation, HTML cleaning, and scraping helpers.
"""
import re
import logging
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# Common job board domains for validation
JOB_BOARD_DOMAINS = {
    'linkedin.com', 'indeed.com', 'glassdoor.com', 'monster.com',
    'greenhouse.io', 'lever.co', 'workday.com', 'myworkdayjobs.com',
    'icims.com', 'smartrecruiters.com', 'jobvite.com', 'ultipro.com',
    'taleo.net', 'brassring.com', 'successfactors.com', 'phenom.com',
    'jobs.ashbyhq.com', 'boards.greenhouse.io', 'apply.workable.com',
}

# Tracking parameters to strip from URLs
TRACKING_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'ref', 'refId', 'referrer', 'source', 'trk', 'trackingId', 'fbclid',
    'gclid', 'mc_cid', 'mc_eid', '_ga', 'srsltid',
}


def validate_job_url(url: str) -> Tuple[bool, str]:
    """
    Validate that a URL is properly formatted and likely a job posting.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not url.strip():
        return False, "URL is required"

    url = url.strip()

    # Must start with http:// or https://
    if not url.startswith(('http://', 'https://')):
        return False, "URL must start with http:// or https://"

    try:
        parsed = urlparse(url)

        if not parsed.netloc:
            return False, "Invalid URL format - no domain found"

        if not parsed.scheme in ('http', 'https'):
            return False, "URL must use http or https protocol"

        # Basic length check
        if len(url) > 2000:
            return False, "URL is too long (max 2000 characters)"

        return True, ""

    except Exception as e:
        return False, f"Invalid URL: {str(e)}"


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing tracking parameters and standardizing format.

    Args:
        url: The URL to normalize

    Returns:
        Normalized URL string
    """
    try:
        parsed = urlparse(url)

        # Remove tracking parameters from query string
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            filtered_params = {
                k: v for k, v in params.items()
                if k.lower() not in TRACKING_PARAMS
            }
            new_query = urlencode(filtered_params, doseq=True)
        else:
            new_query = ''

        # Reconstruct URL without tracking params
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),  # Lowercase domain
            parsed.path.rstrip('/'),  # Remove trailing slash
            parsed.params,
            new_query,
            ''  # Remove fragment
        ))

        return normalized

    except Exception:
        return url


def clean_html_for_llm(html: str, max_chars: int = 12000) -> str:
    """
    Clean HTML content for LLM processing by removing scripts, styles,
    and other non-content elements while preserving semantic structure.

    Args:
        html: Raw HTML string
        max_chars: Maximum characters to return

    Returns:
        Cleaned HTML string suitable for LLM context
    """
    if not html:
        return ""

    # Remove script tags and content
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove style tags and content
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove noscript tags
    html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Remove SVG content (often icons)
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove inline event handlers
    html = re.sub(r'\s+on\w+="[^"]*"', '', html)

    # Remove data attributes (except data-ph which may have content)
    html = re.sub(r'\s+data-(?!ph)[a-z-]+="[^"]*"', '', html, flags=re.IGNORECASE)

    # Simplify class attributes (keep for structure hints)
    html = re.sub(r'class="([^"]{50,})"', lambda m: f'class="{m.group(1)[:50]}..."', html)

    # Remove empty elements
    html = re.sub(r'<(\w+)[^>]*>\s*</\1>', '', html)

    # Collapse multiple whitespace
    html = re.sub(r'\s+', ' ', html)

    # Truncate if needed
    if len(html) > max_chars:
        html = html[:max_chars] + "\n... [truncated]"

    return html.strip()


def extract_visible_text(html: str) -> str:
    """
    Extract visible text content from HTML, stripping all tags.

    Args:
        html: Raw HTML string

    Returns:
        Plain text content
    """
    if not html:
        return ""

    # Remove script and style content first
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Replace block elements with newlines
    text = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)

    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&apos;', "'")

    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


def detect_ats_platform(url: str, html: str) -> Optional[str]:
    """
    Detect the Applicant Tracking System (ATS) platform based on URL and HTML.

    Args:
        url: The job posting URL
        html: The page HTML content

    Returns:
        Platform name string or None if unknown
    """
    url_lower = url.lower()
    html_lower = html.lower() if html else ""

    # URL-based detection
    if 'greenhouse.io' in url_lower or 'boards.greenhouse' in url_lower:
        return 'greenhouse'
    if 'lever.co' in url_lower:
        return 'lever'
    if 'workday.com' in url_lower or 'myworkdayjobs.com' in url_lower:
        return 'workday'
    if 'icims.com' in url_lower:
        return 'icims'
    if 'smartrecruiters.com' in url_lower:
        return 'smartrecruiters'
    if 'jobvite.com' in url_lower:
        return 'jobvite'
    if 'taleo' in url_lower:
        return 'taleo'
    if 'successfactors' in url_lower or 'myflorida.com' in url_lower:
        return 'successfactors'
    if 'phenom' in url_lower:
        return 'phenom'
    if 'ashbyhq.com' in url_lower:
        return 'ashby'

    # HTML-based detection
    if 'icims_content_iframe' in html_lower or 'icims-widget' in html_lower:
        return 'icims'
    if 'phapp.ddo' in html_lower or 'data-ph-id' in html_lower:
        return 'phenom'
    if 'greenhouse' in html_lower and 'application/ld+json' in html_lower:
        return 'greenhouse'
    if 'lever-jobs' in html_lower:
        return 'lever'

    return None


def format_description_sections(sections: list) -> str:
    """
    Format description sections into a readable text format.

    Args:
        sections: List of dicts with 'header' and 'content' keys

    Returns:
        Formatted description string
    """
    if not sections:
        return ""

    parts = []
    for section in sections:
        header = section.get('header', 'Description')
        content = section.get('content', [])

        if header:
            parts.append(f"\n{header.upper()}\n")

        if isinstance(content, list):
            for item in content:
                if item and item.strip():
                    parts.append(f"- {item.strip()}")
        elif content:
            parts.append(content)

    return '\n'.join(parts).strip()

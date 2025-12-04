"""
Job Scraper Module

Playwright-based web scraper for job postings with multi-strategy extraction:
1. JSON-LD schema.org JobPosting (most reliable)
2. CSS selector-based extraction (fallback)
3. Two-pass LLM extraction (for complete description capture)
"""
import asyncio
import json
import logging
import re
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv

from models import JobListing
from scraper_utils import (
    validate_job_url,
    normalize_url,
    clean_html_for_llm,
    extract_visible_text,
    detect_ats_platform,
    format_description_sections,
)

logger = logging.getLogger(__name__)
load_dotenv()


# =============================================================================
# Custom Exceptions
# =============================================================================

class ScrapingError(Exception):
    """Base exception for scraping errors."""
    pass


class PageLoadError(ScrapingError):
    """Failed to load page (timeout, network error)."""
    pass


class BlockedError(ScrapingError):
    """Site blocked the scraper (CAPTCHA, rate limit)."""
    pass


class ExtractionError(ScrapingError):
    """Failed to extract job data from page."""
    pass


# =============================================================================
# Playwright Browser Management
# =============================================================================

class JobScraper:
    """
    Playwright-based job page scraper with lazy browser initialization.

    Usage:
        scraper = JobScraper()
        html, text = await scraper.fetch_page("https://example.com/job/123")
        await scraper.close()

    Or as context manager:
        async with JobScraper() as scraper:
            html, text = await scraper.fetch_page(url)
    """

    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize scraper settings.

        Args:
            headless: Run browser without UI (default True)
            timeout: Page load timeout in milliseconds (default 30s)
        """
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _init_browser(self):
        """Lazy browser initialization."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless
                )
            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                raise PageLoadError(f"Browser initialization failed: {e}")

    async def fetch_page(self, url: str) -> Tuple[str, str]:
        """
        Fetch and render page, return HTML and visible text.

        Uses a multi-strategy approach:
        1. Try "domcontentloaded" first (fast, works for most sites)
        2. Wait for job content selectors to appear
        3. Fall back to longer waits if needed

        Args:
            url: The URL to fetch

        Returns:
            Tuple of (html_content, visible_text)

        Raises:
            PageLoadError: If page fails to load
            BlockedError: If site blocks the scraper
        """
        await self._init_browser()

        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={'width': 1920, 'height': 1080},
            java_script_enabled=True,
        )

        page = await context.new_page()

        try:
            # Strategy 1: Use "domcontentloaded" - faster and more reliable
            # Many sites have continuous network activity that prevents "networkidle"
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.timeout
            )

            if response is None:
                raise PageLoadError("No response received from page")

            # Check for blocking indicators
            status = response.status
            if status == 403:
                raise BlockedError("Access forbidden (403) - site may be blocking scrapers")
            if status == 429:
                raise BlockedError("Rate limited (429) - too many requests")
            if status >= 400:
                raise PageLoadError(f"HTTP error {status}")

            # Strategy 2: Wait for job content to load (with timeout)
            await self._wait_for_job_content(page)

            # Strategy 3: Additional wait for JS-heavy sites
            # Give React/Vue/Angular apps time to hydrate
            await page.wait_for_timeout(2000)

            # Extract content
            html = await page.content()
            text = await page.evaluate("() => document.body.innerText")

            # Check for CAPTCHA or blocking pages
            if self._is_blocked_page(html, text):
                raise BlockedError("Page appears to be blocked or requires CAPTCHA")

            return html, text

        except BlockedError:
            raise
        except PageLoadError:
            raise
        except Exception as e:
            logger.error(f"Error fetching page: {e}")
            raise PageLoadError(f"Failed to load page: {e}")
        finally:
            await context.close()

    async def _wait_for_job_content(self, page):
        """Wait for common job content selectors to appear."""
        # Common selectors for job content across different ATS platforms
        selectors = [
            # JSON-LD (most reliable indicator)
            'script[type="application/ld+json"]',
            # Generic job description selectors
            '[class*="job-description"]',
            '[class*="jobDescription"]',
            '[class*="jobdescription"]',
            '[id*="job-description"]',
            '[id*="job-html"]',
            '[class*="posting-requirements"]',
            '[class*="responsibilities"]',
            # ATS-specific selectors
            '[class*="jd-info"]',  # Workday
            '[class*="posting-page"]',  # Greenhouse
            '[class*="content-wrapper"]',  # Lever
            '[data-automation="jobDescription"]',  # Various
            'article[class*="job"]',
            'main[class*="job"]',
            # Fallback to any main content area
            'main',
            'article',
            '[role="main"]',
        ]

        # Try each selector with a short timeout
        for selector in selectors:
            try:
                await page.wait_for_selector(selector, timeout=3000)
                logger.debug(f"Found content with selector: {selector}")
                return
            except:
                continue

        # If no specific selector found, that's okay - page may still have content
        logger.debug("No specific job selector found, continuing with page content")

    def _is_blocked_page(self, html: str, text: str) -> bool:
        """Check if page shows blocking indicators."""
        # Only check visible text, not HTML (which may contain false positives)
        text_lower = text.lower()

        # Strong indicators - if these appear prominently, likely blocked
        strong_indicators = [
            'captcha',
            'verify you are human',
            'verify you are not a robot',
            'access denied',
            'bot detected',
            'automated access',
        ]

        for pattern in strong_indicators:
            if pattern in text_lower:
                return True

        # Weak indicators - only flag if text is very short (blocked page)
        # Normal job pages have substantial text
        if len(text) < 500:
            weak_indicators = [
                'please enable javascript',
                'unusual traffic',
                'security check',
                'checking your browser',
            ]
            for pattern in weak_indicators:
                if pattern in text_lower:
                    return True

        return False

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


# =============================================================================
# JSON-LD Extraction (Priority 1)
# =============================================================================

def extract_json_ld(html: str) -> Optional[Dict[str, Any]]:
    """
    Extract schema.org JobPosting data from JSON-LD scripts.

    Args:
        html: Raw HTML content

    Returns:
        Dict with job data if found, None otherwise
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'lxml')
        ld_scripts = soup.find_all('script', type='application/ld+json')

        for script in ld_scripts:
            if not script.string:
                continue

            try:
                data = json.loads(script.string)

                # Handle array of schemas
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'JobPosting':
                            return _normalize_job_posting(item)
                elif isinstance(data, dict):
                    if data.get('@type') == 'JobPosting':
                        return _normalize_job_posting(data)
                    # Check for nested graph structure
                    if '@graph' in data:
                        for item in data['@graph']:
                            if isinstance(item, dict) and item.get('@type') == 'JobPosting':
                                return _normalize_job_posting(item)

            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.warning(f"JSON-LD extraction error: {e}")

    return None


def _normalize_job_posting(data: Dict) -> Dict[str, Any]:
    """Normalize schema.org JobPosting to our JobListing format."""
    result = {'raw_json_ld': data}

    # Extract title
    result['job_title'] = data.get('title')

    # Extract company from hiringOrganization
    if 'hiringOrganization' in data:
        org = data['hiringOrganization']
        if isinstance(org, dict):
            result['company'] = org.get('name')
        elif isinstance(org, str):
            result['company'] = org

    # Extract location
    location = _extract_location_from_schema(data.get('jobLocation'))
    if location:
        result['location'] = location

    # Extract apply URL
    result['apply_url'] = data.get('url') or data.get('directApply')

    # Extract and clean description
    description = data.get('description', '')
    if description:
        # Description may be HTML-encoded
        description = _clean_html_description(description)
        result['description'] = description

    # Additional metadata
    result['date_posted'] = data.get('datePosted')
    result['employment_type'] = data.get('employmentType')

    # Salary info
    if 'baseSalary' in data:
        salary = data['baseSalary']
        if isinstance(salary, dict):
            value = salary.get('value', {})
            if isinstance(value, dict):
                min_val = value.get('minValue')
                max_val = value.get('maxValue')
                if min_val and max_val:
                    result['salary_range'] = f"${min_val:,} - ${max_val:,}"

    return result


def _extract_location_from_schema(job_location) -> Optional[str]:
    """Extract location string from schema.org jobLocation field."""
    if not job_location:
        return None

    # Handle single location
    if isinstance(job_location, dict):
        return _parse_single_location(job_location)

    # Handle multiple locations
    if isinstance(job_location, list):
        locations = []
        for loc in job_location:
            if isinstance(loc, dict):
                parsed = _parse_single_location(loc)
                if parsed:
                    locations.append(parsed)
        return ' | '.join(locations) if locations else None

    return None


def _parse_single_location(loc: Dict) -> Optional[str]:
    """Parse a single location object."""
    if 'address' in loc:
        addr = loc['address']
        if isinstance(addr, dict):
            parts = []
            if addr.get('addressLocality'):
                parts.append(addr['addressLocality'])
            if addr.get('addressRegion'):
                parts.append(addr['addressRegion'])
            return ', '.join(parts) if parts else None
        elif isinstance(addr, str):
            return addr
    return None


def _clean_html_description(description: str) -> str:
    """Clean HTML from description field."""
    if not description:
        return ""

    # Unescape HTML entities
    description = description.replace('&lt;', '<')
    description = description.replace('&gt;', '>')
    description = description.replace('&amp;', '&')
    description = description.replace('&quot;', '"')
    description = description.replace('&#39;', "'")
    description = description.replace('&nbsp;', ' ')

    # Extract text from HTML
    return extract_visible_text(description)


# =============================================================================
# CSS Selector Extraction (Priority 2)
# =============================================================================

def extract_with_css_selectors(html: str) -> Dict[str, Any]:
    """
    Fallback extraction using common CSS selectors.

    Args:
        html: Raw HTML content

    Returns:
        Dict with extracted job data (may be partial)
    """
    result = {}

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')

        # Extract title
        title_selectors = [
            '.jobTitle h1', '.job-title h1', '#jobTitle',
            'h1[class*="title"]', 'h1[class*="job"]',
            '[data-automation="job-title"]',
            '.posting-headline h2',
        ]
        for selector in title_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                result['job_title'] = elem.get_text(strip=True)
                break

        # Try page title as fallback for job title
        if 'job_title' not in result:
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                # Often format: "Job Title - Company | Site"
                if ' - ' in title_text:
                    result['job_title'] = title_text.split(' - ')[0].strip()
                elif ' | ' in title_text:
                    result['job_title'] = title_text.split(' | ')[0].strip()

        # Extract location
        location_selectors = [
            '.jobLocation', '.job-location', '#job-location',
            '[class*="location"]', '[data-automation="location"]',
            '.jobGeoLocation',
        ]
        for selector in location_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                result['location'] = elem.get_text(strip=True)
                break

        # Extract company
        company_selectors = [
            '.company-name', '.employer-name', '[class*="company"]',
            '[data-automation="company"]', '.jobCompany',
        ]
        for selector in company_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                result['company'] = elem.get_text(strip=True)
                break

        # Extract description
        description_selectors = [
            '.jobdescription', '.job-description', '#job-description',
            '#job-html', '[class*="description"]', '.posting-requirements',
            'article', '.job-details', '.job-content',
        ]
        for selector in description_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator='\n', strip=True)
                if len(text) > 100:  # Ensure it's substantial
                    result['description'] = text
                    break

    except Exception as e:
        logger.warning(f"CSS selector extraction error: {e}")

    return result


# =============================================================================
# LLM Extraction (Priority 3 / Enhancement)
# =============================================================================

def get_openai_client():
    """Get OpenAI client, returns None if not configured."""
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('open_ai_api_key')
    if not api_key:
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def _call_llm(
    client,
    prompt: str,
    system_msg: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 3000
) -> Optional[Dict]:
    """Helper to make LLM call and parse JSON response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)

        # Parse JSON response
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Attempting extraction...")
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return None

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def extract_job_with_llm(
    html: str,
    visible_text: str,
    url: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Two-pass LLM extraction for job posting data.

    Pass 1: Extract all structured data from page content
    Pass 2: Verify and refine extraction for completeness

    Args:
        html: Raw HTML content
        visible_text: Extracted visible text
        url: Source URL

    Returns:
        Dict with extracted job data
    """
    client = get_openai_client()
    if not client:
        logger.warning("OpenAI client not available, skipping LLM extraction")
        return {}

    # Prepare content for LLM
    cleaned_html = clean_html_for_llm(html, max_chars=6000)
    truncated_text = visible_text[:6000] if visible_text else ""

    content = f"""URL: {url}

VISIBLE TEXT:
{truncated_text}

HTML STRUCTURE (cleaned):
{cleaned_html}"""

    # =========================================================================
    # PASS 1: Full extraction
    # =========================================================================
    pass1_prompt = f"""Extract the job posting information from this page.

{content}

Return a JSON object with these fields:

{{
  "job_title": "The position title (not the page title)",
  "company": "The hiring company name",
  "location": "Job location (city, state) or 'Remote' or 'Hybrid'",
  "apply_url": "Direct application URL if visible, otherwise null",
  "description_sections": [
    {{
      "header": "Section header (e.g., 'About the Role', 'Responsibilities', 'Requirements')",
      "content": ["bullet point 1", "bullet point 2", ...]
    }}
  ],
  "salary_range": "Salary if mentioned, null otherwise",
  "job_type": "Full-time/Part-time/Contract/Internship",
  "experience_level": "Entry/Mid/Senior/Lead if mentioned, null otherwise"
}}

CRITICAL RULES:
1. Extract ALL bullet points from each section - do not summarize or skip any
2. Preserve the exact section headers from the posting (About, Responsibilities, Requirements, Qualifications, What You'll Do, etc.)
3. Do NOT include EEO statements, legal boilerplate, or company-wide benefits in description_sections
4. The job_title should be the actual position name, NOT the page title or company name
5. If apply_url requires JavaScript/navigation, set to null
6. Use null for any field not clearly present in the posting

Respond with ONLY the JSON object, no explanation."""

    pass1_result = _call_llm(
        client,
        pass1_prompt,
        "You are a precise job posting parser. Extract structured data accurately. Respond with valid JSON only.",
        model
    )

    if not pass1_result:
        return {}

    # =========================================================================
    # PASS 2: Verification and refinement
    # =========================================================================
    sections_summary = []
    for section in pass1_result.get('description_sections', []):
        header = section.get('header', 'Unknown')
        count = len(section.get('content', []))
        sections_summary.append(f"  - {header}: {count} items")

    pass2_prompt = f"""Verify this job extraction is complete. Compare against the original content.

ORIGINAL PAGE TEXT (truncated):
{truncated_text[:4000]}

CURRENT EXTRACTION:
Job Title: {pass1_result.get('job_title')}
Company: {pass1_result.get('company')}
Location: {pass1_result.get('location')}
Sections extracted:
{chr(10).join(sections_summary) if sections_summary else '  None'}

VERIFICATION CHECKLIST:
1. Is the job title the actual position name (not a page title or "Careers at X")?
2. Is the company name correct (the actual employer, not a parent company or job board)?
3. Are ALL section headers from the original posting captured?
4. Does each section have ALL its bullet points? Count them in the original.
5. Are requirements/qualifications completely extracted?

If ANY data is missing or incorrect, return the COMPLETE corrected extraction in the same JSON format.

If extraction is already complete and accurate, return {{}}.

Respond with ONLY valid JSON."""

    pass2_result = _call_llm(
        client,
        pass2_prompt,
        "You are a job extraction verifier. Check for completeness and accuracy. Respond with valid JSON only.",
        model
    )

    # Merge pass 2 corrections into pass 1 results
    if pass2_result:
        for key, value in pass2_result.items():
            if value is not None:
                pass1_result[key] = value

    return pass1_result


# =============================================================================
# Main Orchestrator
# =============================================================================

def scrape_job_url(url: str, use_llm: bool = True) -> JobListing:
    """
    Main entry point: Scrape URL and return JobListing.

    Extraction priority:
    1. JSON-LD schema.org data (most reliable)
    2. CSS selector-based extraction
    3. LLM extraction (always used for complete description)

    Args:
        url: Job posting URL
        use_llm: Whether to use LLM for extraction/enhancement (default True)

    Returns:
        JobListing with extracted data

    Raises:
        ValueError: If URL is invalid
        PageLoadError: If page fails to load
        BlockedError: If site blocks the scraper
        ExtractionError: If no job data could be extracted
    """
    # Validate URL
    is_valid, error = validate_job_url(url)
    if not is_valid:
        raise ValueError(f"Invalid job URL: {error}")

    # Normalize URL
    normalized_url = normalize_url(url)

    # Fetch page with Playwright
    async def _fetch():
        async with JobScraper() as scraper:
            return await scraper.fetch_page(url)

    try:
        html, visible_text = asyncio.run(_fetch())
    except (PageLoadError, BlockedError):
        raise
    except Exception as e:
        raise PageLoadError(f"Failed to fetch page: {e}")

    # Initialize metadata
    metadata = {
        'scrape_timestamp': datetime.utcnow().isoformat(),
        'source_url': url,
        'normalized_url': normalized_url,
        'ats_platform': detect_ats_platform(url, html),
    }

    # Track what we've extracted
    job_data = {}

    # =========================================================================
    # Strategy 1: JSON-LD extraction (most reliable)
    # =========================================================================
    json_ld_data = extract_json_ld(html)
    if json_ld_data:
        metadata['scrape_source'] = 'json_ld'
        metadata['extraction_methods'] = {'json_ld': list(json_ld_data.keys())}
        job_data.update(json_ld_data)
        logger.info(f"JSON-LD extraction successful: {list(json_ld_data.keys())}")

    # =========================================================================
    # Strategy 2: CSS selector extraction (fill gaps)
    # =========================================================================
    css_data = extract_with_css_selectors(html)
    if css_data:
        filled_fields = []
        for key, value in css_data.items():
            if key not in job_data or not job_data[key]:
                job_data[key] = value
                filled_fields.append(key)
        if filled_fields:
            if 'extraction_methods' not in metadata:
                metadata['extraction_methods'] = {}
            metadata['extraction_methods']['css'] = filled_fields
            if 'scrape_source' not in metadata:
                metadata['scrape_source'] = 'css_selectors'

    # =========================================================================
    # Strategy 3: LLM extraction (enhance description / fill gaps)
    # =========================================================================
    if use_llm:
        llm_data = extract_job_with_llm(html, visible_text, url)
        if llm_data:
            metadata['llm_extraction'] = True
            metadata['llm_raw'] = llm_data

            # Fill missing fields from LLM
            for key in ['job_title', 'company', 'location', 'apply_url']:
                if key in llm_data and llm_data[key] and not job_data.get(key):
                    job_data[key] = llm_data[key]

            # Use LLM sections for structured description
            if llm_data.get('description_sections'):
                formatted_desc = format_description_sections(llm_data['description_sections'])
                if formatted_desc and len(formatted_desc) > 100:
                    job_data['description'] = formatted_desc
                    metadata['description_source'] = 'llm_sections'

    # =========================================================================
    # Fallback: Use visible text as description
    # =========================================================================
    if not job_data.get('description') and visible_text:
        job_data['description'] = visible_text[:10000]
        metadata['description_source'] = 'visible_text'

    # Validate we got something useful
    if not job_data.get('description') and not job_data.get('job_title'):
        raise ExtractionError("Could not extract job title or description from page")

    # Create JobListing
    return JobListing(
        job_title=job_data.get('job_title'),
        company=job_data.get('company'),
        location=job_data.get('location'),
        apply_url=job_data.get('apply_url') or url,
        description=job_data.get('description', ''),
        source_url=url,
        metadata=metadata
    )

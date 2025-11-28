"""
Unit Tests for the Processing Module

Tests the extraction logic and hashing functions.
Run with: pytest test_processing.py -v
"""
import pytest
from processing import (
    extract_job_details,
    hash_description,
    clean_job_text,
    _extract_with_regex,
)
from models import JobListing


class TestHashDescription:
    """Tests for the hash_description function."""

    def test_basic_hashing(self):
        """Test that hashing produces a consistent 64-char hex string."""
        result = hash_description("Hello World")
        assert len(result) == 64
        assert result.isalnum()

    def test_normalization_case(self):
        """Test that hashing is case-insensitive."""
        hash1 = hash_description("Hello World")
        hash2 = hash_description("hello world")
        hash3 = hash_description("HELLO WORLD")
        assert hash1 == hash2 == hash3

    def test_normalization_whitespace(self):
        """Test that hashing ignores whitespace differences."""
        hash1 = hash_description("Hello World")
        hash2 = hash_description("Hello    World")
        hash3 = hash_description("Hello\n\tWorld")
        hash4 = hash_description("  Hello World  ")
        assert hash1 == hash2 == hash3 == hash4

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = hash_description("Data Scientist")
        hash2 = hash_description("Data Engineer")
        assert hash1 != hash2

    def test_empty_string(self):
        """Test hashing an empty string."""
        result = hash_description("")
        assert len(result) == 64


class TestExtractJobDetails:
    """Tests for the main extract_job_details function."""

    def test_structured_input(self):
        """Test extraction from well-formatted job listing."""
        text = """
        Job Title: Senior Data Scientist
        Company: TechCorp Inc.
        Location: San Francisco, CA

        About the role:
        We are looking for a senior data scientist to join our team.

        Apply here: https://techcorp.com/careers/apply
        """
        result = extract_job_details(text)

        assert result.job_title == "Senior Data Scientist"
        assert result.company == "TechCorp Inc."
        assert "San Francisco" in result.location
        assert "techcorp.com" in result.apply_url

    def test_description_preserved(self):
        """Test that the full description is preserved."""
        text = "This is a test job description with important details."
        result = extract_job_details(text)
        assert result.description == text

    def test_partial_extraction(self):
        """Test extraction when only some fields are present."""
        text = """
        Job Title: Python Developer
        We need someone who knows Python and Django.
        """
        result = extract_job_details(text)

        assert result.job_title == "Python Developer"
        assert result.description is not None
        # Company and location may or may not be found

    def test_metadata_populated(self):
        """Test that extraction methods are tracked in metadata."""
        text = """
        Job Title: ML Engineer
        Location: Remote
        """
        result = extract_job_details(text)

        assert 'extraction_methods' in result.metadata


class TestRegexExtraction:
    """Tests for the regex extraction layer."""

    def test_extract_job_title(self):
        """Test job title extraction patterns."""
        text = "Job Title: Software Engineer"
        job = JobListing(description=text)
        result, found = _extract_with_regex(text, job)

        assert result.job_title == "Software Engineer"
        assert 'job_title' in found

    def test_extract_location_city_state(self):
        """Test location extraction for City, ST format."""
        text = "This position is based in Austin, TX."
        job = JobListing(description=text)
        result, found = _extract_with_regex(text, job)

        assert result.location == "Austin, TX"

    def test_extract_location_remote(self):
        """Test extraction of Remote location."""
        text = "This is a fully Remote position."
        job = JobListing(description=text)
        result, found = _extract_with_regex(text, job)

        assert result.location.lower() == "remote"

    def test_extract_apply_url(self):
        """Test URL extraction."""
        text = "Apply at https://company.com/jobs/12345"
        job = JobListing(description=text)
        result, found = _extract_with_regex(text, job)

        assert "company.com/jobs" in result.apply_url


class TestCleanJobText:
    """Tests for the text cleaning function."""

    def test_removes_eeo_boilerplate(self):
        """Test that Equal Opportunity Employer text is removed."""
        text = """
        Great job description here.

        Equal Opportunity Employer. All qualified applicants will receive consideration.
        """
        result = clean_job_text(text)

        assert "Equal Opportunity" not in result
        assert "Great job description" in result

    def test_normalizes_whitespace(self):
        """Test that excessive newlines are reduced."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_job_text(text)

        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestJobListingModel:
    """Tests for the JobListing dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        job = JobListing()
        assert job.job_title is None
        assert job.company is None
        assert job.location is None
        assert job.description == ""
        assert job.metadata == {}

    def test_to_dict(self):
        """Test dictionary conversion."""
        job = JobListing(
            job_title="Engineer",
            company="Acme",
            location="NYC"
        )
        result = job.to_dict()

        assert result['job_title'] == "Engineer"
        assert result['company'] == "Acme"
        assert result['location'] == "NYC"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'job_title': 'Analyst',
            'company': 'DataCo',
            'description': 'Analyze data'
        }
        job = JobListing.from_dict(data)

        assert job.job_title == "Analyst"
        assert job.company == "DataCo"
        assert job.description == "Analyze data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

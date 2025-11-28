"""
Core data structures (models) used across the application.

This file serves as the single source of truth for shared data structures,
preventing circular dependencies between modules.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class JobListing:
    """A structured representation of a job listing."""
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    apply_url: Optional[str] = None
    description: str = ""
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the JobListing to a dictionary."""
        return {
            'job_title': self.job_title,
            'company': self.company,
            'location': self.location,
            'apply_url': self.apply_url,
            'description': self.description,
            'source_url': self.source_url,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobListing':
        """Create a JobListing from a dictionary."""
        return cls(
            job_title=data.get('job_title'),
            company=data.get('company'),
            location=data.get('location'),
            apply_url=data.get('apply_url'),
            description=data.get('description', ''),
            source_url=data.get('source_url'),
            metadata=data.get('metadata', {})
        )

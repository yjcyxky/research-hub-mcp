"""
Pydantic models for pdf2text.

The models are intentionally minimal and focused on the PDF extraction
workflow (structured text, figures, and tables).
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ServerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"
    NOT_INSTALLED = "not_installed"


class GrobidConfig(BaseModel):
    """Configuration for a GROBID server instance."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    grobid_home: Optional[Path] = Field(default=None, description="GROBID home directory")
    port: int = Field(default=8070, ge=1024, le=65535, description="Server port number")
    host: str = Field(default="localhost", description="Server host address")
    version: str = Field(default="0.8.0", description="GROBID version")
    memory: str = Field(default="4g", pattern=r"^\d+[gmGM]$", description="Java heap memory")
    config_path: Optional[Path] = Field(default=None, description="Custom GROBID config path")
    auto_download: bool = Field(default=True, description="Download GROBID if missing")

    @field_validator("grobid_home")
    @classmethod
    def validate_home(cls, value: Optional[Path]) -> Optional[Path]:
        return value.resolve() if value and not value.is_absolute() else value


class ServerInfo(BaseModel):
    """Status information for a running GROBID server."""

    model_config = ConfigDict(validate_assignment=True)

    url: str = Field(description="Server URL")
    health_check_url: str = Field(description="Health check endpoint")
    status: ServerStatus = Field(description="Current server status")
    pid: Optional[int] = Field(default=None, description="Process ID")
    version: str = Field(description="GROBID version")
    config: GrobidConfig = Field(description="Server configuration")


class PDFExtractionRequest(BaseModel):
    """Request model for PDF extraction."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pdf_file: Path = Field(description="PDF file or directory path")
    output_dir: Path = Field(description="Output directory")
    grobid_url: Optional[HttpUrl] = Field(default=None, description="GROBID server URL")
    extract_figures: bool = Field(default=True, description="Extract figures from PDFs")
    extract_tables: bool = Field(default=True, description="Extract tables from PDFs")
    consolidate_citations: bool = Field(default=True, description="Consolidate citations")
    include_raw_citations: bool = Field(default=False, description="Include raw citations")
    segment_sentences: bool = Field(default=False, description="Segment text into sentences")
    batch_size: int = Field(default=1, ge=1, le=100, description="Batch size")

    @field_validator("pdf_file")
    @classmethod
    def validate_pdf_file(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Path does not exist: {value}")
        if value.is_file() and value.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {value}")
        return value.resolve()

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        resolved = value.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


class DocumentSection(BaseModel):
    """A section of the extracted document."""

    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(description="Section title")
    text: str = Field(description="Section text")
    section_number: Optional[str] = Field(default=None, description="Section number")
    subsections: List["DocumentSection"] = Field(
        default_factory=list, description="Nested subsections"
    )


class Reference(BaseModel):
    """Bibliographic reference entry."""

    model_config = ConfigDict(validate_assignment=True)

    title: Optional[str] = Field(default=None, description="Reference title")
    authors: List[str] = Field(default_factory=list, description="Authors")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Publication venue")
    doi: Optional[str] = Field(default=None, description="DOI")
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    raw_text: str = Field(description="Raw reference text")


class Figure(BaseModel):
    """Extracted figure information."""

    model_config = ConfigDict(validate_assignment=True)

    caption: Optional[str] = Field(default=None, description="Figure caption")
    label: Optional[str] = Field(default=None, description="Figure label")
    file_path: Optional[Path] = Field(default=None, description="Path to figure file")
    page: Optional[int] = Field(default=None, ge=1, description="Page number")


class Table(BaseModel):
    """Extracted table information."""

    model_config = ConfigDict(validate_assignment=True)

    caption: Optional[str] = Field(default=None, description="Table caption")
    label: Optional[str] = Field(default=None, description="Table label")
    content: Any = Field(description="Table content")
    page: Optional[int] = Field(default=None, ge=1, description="Page number")


class ExtractedDocument(BaseModel):
    """Structured representation of an extracted PDF."""

    model_config = ConfigDict(validate_assignment=True)

    title: Optional[str] = Field(default=None, description="Document title")
    abstract: Optional[str] = Field(default=None, description="Abstract text")
    sections: List[DocumentSection] = Field(default_factory=list, description="Sections")
    authors: List[str] = Field(default_factory=list, description="Authors")
    affiliations: List[str] = Field(default_factory=list, description="Affiliations")
    references: List[Reference] = Field(default_factory=list, description="References")
    figures: List[Figure] = Field(default_factory=list, description="Figures")
    tables: List[Table] = Field(default_factory=list, description="Tables")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_file: Optional[Path] = Field(default=None, description="Source PDF path")


DocumentSection.model_rebuild()
ExtractedDocument.model_rebuild()

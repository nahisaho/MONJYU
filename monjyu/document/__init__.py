# Document Processing Module
"""
Document processing components for MONJYU.

This module handles:
- File loading and format detection
- Document parsing (PDF, text, etc.)
- Text chunking
- Metadata extraction
"""

from monjyu.document.models import (
    AcademicPaperDocument,
    AcademicSection,
    Author,
    Figure,
    Reference,
    Table,
    TextUnit,
)
from monjyu.document.loader import FileLoader
from monjyu.document.chunker import TextChunker
from monjyu.document.parser import DocumentParser
from monjyu.document.pipeline import DocumentProcessingPipeline, PipelineConfig, ProcessingResult, BatchResult

__all__ = [
    # Models
    "AcademicPaperDocument",
    "AcademicSection",
    "Author",
    "Figure",
    "Reference",
    "Table",
    "TextUnit",
    # Components
    "FileLoader",
    "TextChunker",
    "DocumentParser",
    "DocumentProcessingPipeline",
    "PipelineConfig",
    "ProcessingResult",
    "BatchResult",
]

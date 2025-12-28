# PDF Processing Module
"""
PDF processing strategies for MONJYU.

Supports:
- UnstructuredPDFProcessor: Local development using unstructured library
- AzureDocIntelPDFProcessor: Production using Azure Document Intelligence
"""

from monjyu.document.pdf.base import PDFProcessor, PDFProcessorProtocol
from monjyu.document.pdf.unstructured_processor import UnstructuredPDFProcessor

__all__ = [
    "PDFProcessor",
    "PDFProcessorProtocol",
    "UnstructuredPDFProcessor",
]

# Azure processor is conditionally imported
HAS_AZURE = False
try:
    from monjyu.document.pdf.azure_processor import AzureDocIntelPDFProcessor
    __all__.append("AzureDocIntelPDFProcessor")
    HAS_AZURE = True
except ImportError:
    AzureDocIntelPDFProcessor = None  # type: ignore

__all__.append("HAS_AZURE")

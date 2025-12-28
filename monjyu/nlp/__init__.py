# NLP Processing Module
"""
NLP components for Index Level 1.

Provides keyword extraction, noun phrase extraction, and named entity recognition
without LLM costs using spaCy, NLTK/RAKE.
"""

from monjyu.nlp.base import (
    NLPProcessorProtocol,
    NLPProcessor,
    NLPFeatures,
)
from monjyu.nlp.spacy_processor import SpacyNLPProcessor
from monjyu.nlp.rake_extractor import RAKEKeywordExtractor

__all__ = [
    "NLPProcessorProtocol",
    "NLPProcessor",
    "NLPFeatures",
    "SpacyNLPProcessor",
    "RAKEKeywordExtractor",
]

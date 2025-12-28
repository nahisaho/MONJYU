# Document Processing Data Models
"""
Data models for document processing.

These dataclasses define the structure of:
- Academic papers and their components
- Text units (chunks)
- Metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Author:
    """著者情報"""
    
    name: str
    affiliation: str | None = None
    email: str | None = None
    orcid: str | None = None
    
    def __str__(self) -> str:
        return self.name


@dataclass
class AcademicSection:
    """学術論文セクション"""
    
    heading: str
    level: int
    section_type: str  # introduction, methods, results, discussion, conclusion, other
    content: str
    page_numbers: list[int] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"[{self.section_type}] {self.heading}"


@dataclass
class Table:
    """テーブル"""
    
    table_id: str
    caption: str | None
    content: str  # HTML or Markdown format
    page_number: int = 0
    
    def __str__(self) -> str:
        return f"Table: {self.caption or self.table_id}"


@dataclass
class Figure:
    """図"""
    
    figure_id: str
    caption: str | None
    image_path: str | None = None
    page_number: int = 0
    
    def __str__(self) -> str:
        return f"Figure: {self.caption or self.figure_id}"


@dataclass
class Reference:
    """参考文献"""
    
    ref_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    
    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            authors_str += " et al."
        return f"{authors_str} ({self.year}). {self.title}"


@dataclass
class AcademicPaperDocument:
    """学術論文ドキュメント"""
    
    # 基本情報
    file_name: str
    file_type: str
    title: str
    
    # 著者情報
    authors: list[Author] = field(default_factory=list)
    
    # 識別子
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    
    # 出版情報
    publication_year: int | None = None
    venue: str | None = None
    venue_type: str = "unknown"  # journal, conference, preprint
    
    # 構造化コンテンツ
    abstract: str = ""
    sections: list[AcademicSection] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    
    # 参考文献
    references: list[Reference] = field(default_factory=list)
    
    # メタデータ
    keywords: list[str] = field(default_factory=list)
    language: str = "en"
    page_count: int = 0
    raw_text: str = ""  # Full text for fallback
    
    def __str__(self) -> str:
        return f"{self.title} ({self.file_name})"
    
    @property
    def full_text(self) -> str:
        """全テキストを取得"""
        parts = []
        
        if self.abstract:
            parts.append(f"Abstract:\n{self.abstract}")
        
        for section in self.sections:
            parts.append(f"\n{section.heading}:\n{section.content}")
        
        return "\n".join(parts) if parts else self.raw_text
    
    @property
    def author_names(self) -> list[str]:
        """著者名リストを取得"""
        return [author.name for author in self.authors]
    
    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "title": self.title,
            "authors": [a.name for a in self.authors],
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "publication_year": self.publication_year,
            "venue": self.venue,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "language": self.language,
            "page_count": self.page_count,
            "section_count": len(self.sections),
            "reference_count": len(self.references),
        }


@dataclass
class TextUnit:
    """テキストユニット（チャンク）"""
    
    id: str
    text: str
    n_tokens: int
    document_id: str
    
    # 位置情報
    chunk_index: int
    start_char: int
    end_char: int
    
    # セクション情報
    section_type: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    
    # メタデータ
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextUnit[{self.id}]: {preview}"
    
    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "section_type": self.section_type,
            "page_numbers": self.page_numbers,
            "metadata": self.metadata,
        }


@dataclass
class ProcessedPDF:
    """PDF処理結果（内部用）"""
    
    file_name: str
    title: str
    authors: list[Author]
    doi: str | None
    arxiv_id: str | None
    abstract: str
    sections: list[AcademicSection]
    tables: list[Table]
    figures: list[Figure]
    references: list[Reference]
    keywords: list[str]
    language: str
    page_count: int
    raw_text: str = ""

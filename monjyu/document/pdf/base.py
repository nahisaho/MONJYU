# PDF Processor Base
"""
Base class and protocol for PDF processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from monjyu.document.models import ProcessedPDF


class PDFProcessorProtocol(Protocol):
    """PDF処理プロトコル"""
    
    def process(self, content: bytes) -> ProcessedPDF:
        """PDFを処理して構造化データを返す
        
        Args:
            content: PDFファイルの内容（バイト列）
            
        Returns:
            処理されたPDFデータ
        """
        ...


class PDFProcessor(ABC):
    """PDF処理の抽象基底クラス"""
    
    @abstractmethod
    def process(self, content: bytes) -> ProcessedPDF:
        """PDFを処理して構造化データを返す
        
        Args:
            content: PDFファイルの内容（バイト列）
            
        Returns:
            処理されたPDFデータ
        """
        ...
    
    def _classify_section(self, heading: str) -> str:
        """セクションタイプを分類（IMRaD）
        
        Args:
            heading: セクション見出し
            
        Returns:
            セクションタイプ
        """
        heading_lower = heading.lower()
        
        keywords = {
            "introduction": ["introduction", "背景", "はじめに", "序論"],
            "methods": ["method", "approach", "methodology", "手法", "方法"],
            "results": ["result", "experiment", "evaluation", "結果", "実験"],
            "discussion": ["discussion", "考察", "議論", "analysis"],
            "conclusion": ["conclusion", "summary", "結論", "まとめ"],
            "related_work": ["related", "先行研究", "関連研究", "background"],
        }
        
        for section_type, kws in keywords.items():
            if any(kw in heading_lower for kw in kws):
                return section_type
        
        return "other"

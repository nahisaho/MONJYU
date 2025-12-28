# Unstructured PDF Processor
"""
PDF processing using the unstructured library.
Used for local development without Azure dependencies.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from monjyu.document.models import (
    AcademicSection,
    Author,
    Figure,
    ProcessedPDF,
    Reference,
    Table,
)
from monjyu.document.pdf.base import PDFProcessor


class UnstructuredPDFProcessor(PDFProcessor):
    """unstructured による PDF 処理（ローカル開発用）
    
    unstructured ライブラリを使用してPDFを解析し、
    学術論文の構造を抽出する。
    
    Example:
        >>> processor = UnstructuredPDFProcessor(strategy="hi_res")
        >>> result = processor.process(pdf_bytes)
    """
    
    def __init__(
        self,
        strategy: str = "fast",
        extract_images: bool = True,
        infer_table_structure: bool = True,
        languages: list[str] | None = None,
    ) -> None:
        """初期化
        
        Args:
            strategy: 処理戦略 ("fast", "hi_res", "ocr_only")
            extract_images: 画像を抽出するか
            infer_table_structure: テーブル構造を推論するか
            languages: OCR言語リスト
        """
        self.strategy = strategy
        self.extract_images = extract_images
        self.infer_table_structure = infer_table_structure
        self.languages = languages or ["eng"]
    
    def process(self, content: bytes) -> ProcessedPDF:
        """PDFを処理
        
        Args:
            content: PDFファイル内容
            
        Returns:
            処理されたPDF
        """
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            file=BytesIO(content),
            strategy=self.strategy,
            extract_images_in_pdf=self.extract_images,
            infer_table_structure=self.infer_table_structure,
            languages=self.languages,
        )
        
        return self._structure_elements(elements)
    
    def _structure_elements(self, elements: list[Any]) -> ProcessedPDF:
        """要素を構造化
        
        Args:
            elements: unstructuredの要素リスト
            
        Returns:
            構造化されたPDF
        """
        title = ""
        authors: list[Author] = []
        abstract = ""
        sections: list[AcademicSection] = []
        tables: list[Table] = []
        figures: list[Figure] = []
        references: list[Reference] = []
        raw_text_parts: list[str] = []
        
        current_section: AcademicSection | None = None
        in_abstract = False
        in_references = False
        
        for element in elements:
            text = element.text.strip() if hasattr(element, "text") else ""
            category = getattr(element, "category", "")
            metadata = getattr(element, "metadata", {})
            
            if not text:
                continue
            
            raw_text_parts.append(text)
            page_number = self._get_page_number(metadata)
            
            # タイトル検出（最初のTitleカテゴリ）
            if category == "Title" and not title:
                title = text
                continue
            
            # 著者検出（タイトル直後の短いテキスト）
            if not authors and title and len(text) < 500 and not abstract:
                potential_authors = self._extract_authors_from_text(text)
                if potential_authors:
                    authors = potential_authors
                    continue
            
            # アブストラクト検出
            if "abstract" in text.lower() and len(text) < 50:
                in_abstract = True
                continue
            
            if in_abstract and category == "NarrativeText":
                abstract = text
                in_abstract = False
                continue
            
            # References セクション検出
            if any(kw in text.lower() for kw in ["references", "bibliography", "参考文献"]):
                in_references = True
                if current_section:
                    sections.append(current_section)
                    current_section = None
                continue
            
            # 参考文献の処理
            if in_references and category in ("NarrativeText", "ListItem"):
                ref = self._parse_reference(text, len(references) + 1)
                if ref:
                    references.append(ref)
                continue
            
            # テーブル
            if category == "Table":
                tables.append(Table(
                    table_id=f"table_{len(tables)+1}",
                    caption=self._get_metadata_value(metadata, "caption"),
                    content=text,
                    page_number=page_number,
                ))
                continue
            
            # 図（キャプション）
            if category == "Image" or (category == "FigureCaption"):
                figures.append(Figure(
                    figure_id=f"figure_{len(figures)+1}",
                    caption=text if category == "FigureCaption" else None,
                    image_path=None,
                    page_number=page_number,
                ))
                continue
            
            # セクションヘッダー
            if category == "Header" or self._is_section_header(text, category):
                if current_section:
                    sections.append(current_section)
                
                current_section = AcademicSection(
                    heading=text,
                    level=self._detect_heading_level(text),
                    section_type=self._classify_section(text),
                    content="",
                    page_numbers=[page_number] if page_number else [],
                )
                continue
            
            # 本文（現在のセクションに追加）
            if current_section and category in ("NarrativeText", "ListItem", "Text"):
                current_section.content += text + "\n"
                if page_number and page_number not in current_section.page_numbers:
                    current_section.page_numbers.append(page_number)
        
        # 最後のセクションを追加
        if current_section:
            sections.append(current_section)
        
        # 識別子を抽出
        raw_text = "\n".join(raw_text_parts)
        doi = self._extract_doi(raw_text)
        arxiv_id = self._extract_arxiv_id(raw_text)
        
        # 言語検出
        language = self._detect_language(raw_text)
        
        # ページ数
        page_count = self._get_max_page_number(elements)
        
        return ProcessedPDF(
            file_name="",
            title=title,
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            keywords=[],
            language=language,
            page_count=page_count,
            raw_text=raw_text,
        )
    
    def _get_page_number(self, metadata: Any) -> int:
        """メタデータからページ番号を取得"""
        if isinstance(metadata, dict):
            return metadata.get("page_number", 0)
        return getattr(metadata, "page_number", 0)
    
    def _get_metadata_value(self, metadata: Any, key: str) -> str | None:
        """メタデータから値を取得"""
        if isinstance(metadata, dict):
            return metadata.get(key)
        return getattr(metadata, key, None)
    
    def _get_max_page_number(self, elements: list[Any]) -> int:
        """最大ページ番号を取得"""
        max_page = 0
        for element in elements:
            metadata = getattr(element, "metadata", {})
            page = self._get_page_number(metadata)
            if page > max_page:
                max_page = page
        return max_page
    
    def _is_section_header(self, text: str, category: str) -> bool:
        """セクションヘッダーかどうか判定"""
        # カテゴリがTitle/Headerで短いテキスト
        if category in ("Title", "Header") and len(text) < 100:
            return True
        
        # 番号付きセクション（例: "1. Introduction"）
        if re.match(r"^\d+\.?\s+[A-Z]", text) and len(text) < 100:
            return True
        
        # IMRaDキーワード
        text_lower = text.lower()
        imrad_keywords = [
            "introduction", "method", "result", "discussion",
            "conclusion", "related work", "background", "experiment"
        ]
        if any(kw in text_lower for kw in imrad_keywords) and len(text) < 100:
            return True
        
        return False
    
    def _detect_heading_level(self, text: str) -> int:
        """ヘッダーレベルを検出"""
        # 番号からレベルを推定
        match = re.match(r"^(\d+)(\.(\d+))?(\.(\d+))?", text)
        if match:
            if match.group(5):  # 1.2.3
                return 3
            if match.group(3):  # 1.2
                return 2
            return 1
        return 1
    
    def _extract_authors_from_text(self, text: str) -> list[Author]:
        """テキストから著者を抽出"""
        authors = []
        
        # カンマやandで区切られた名前を検出
        # 単純なパターン: "Name1, Name2 and Name3"
        names = re.split(r",\s*|\s+and\s+", text)
        
        for name in names:
            name = name.strip()
            # 著者名らしいか判定（2-4語、数字なし）
            words = name.split()
            if 1 <= len(words) <= 5 and not any(c.isdigit() for c in name):
                # アフィリエーション記号を除去
                name = re.sub(r"[*†‡§∗]+", "", name).strip()
                if name:
                    authors.append(Author(name=name))
        
        # 著者が多すぎる場合は誤検出の可能性
        if len(authors) > 20:
            return []
        
        return authors
    
    def _parse_reference(self, text: str, index: int) -> Reference | None:
        """参考文献をパース"""
        if len(text) < 20:
            return None
        
        # 簡易パース：タイトル、著者、年を抽出
        year_match = re.search(r"\(?(19|20)\d{2}\)?", text)
        year = int(year_match.group().strip("()")) if year_match else None
        
        doi = self._extract_doi(text)
        arxiv_id = self._extract_arxiv_id(text)
        
        return Reference(
            ref_id=f"ref_{index}",
            title=text[:200],  # 最初の200文字をタイトルとして
            authors=[],
            year=year,
            venue=None,
            doi=doi,
            arxiv_id=arxiv_id,
        )
    
    def _extract_doi(self, text: str) -> str | None:
        """DOIを抽出"""
        doi_pattern = r"10\.\d{4,}/[^\s\]>\"']+"
        match = re.search(doi_pattern, text)
        return match.group() if match else None
    
    def _extract_arxiv_id(self, text: str) -> str | None:
        """arXiv IDを抽出"""
        arxiv_pattern = r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)"
        match = re.search(arxiv_pattern, text, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _detect_language(self, text: str) -> str:
        """言語を検出"""
        try:
            from langdetect import detect
            return detect(text[:1000])  # 最初の1000文字で判定
        except Exception:
            return "en"  # デフォルト英語

# Azure Document Intelligence PDF Processor
"""
PDF processing using Azure Document Intelligence (formerly Form Recognizer).
Used for production with higher accuracy and additional features.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from monjyu.document.models import (
    AcademicSection,
    Author,
    Figure,
    ProcessedPDF,
    Reference,
    Table,
)
from monjyu.document.pdf.base import PDFProcessor

if TYPE_CHECKING:
    from azure.ai.documentintelligence import DocumentIntelligenceClient


class AzureDocIntelPDFProcessor(PDFProcessor):
    """Azure Document Intelligence による PDF 処理（本番用）
    
    Azure Document Intelligence を使用してPDFを解析し、
    高精度な構造抽出を行う。
    
    Example:
        >>> processor = AzureDocIntelPDFProcessor(
        ...     endpoint="https://xxx.cognitiveservices.azure.com/",
        ...     api_key="your-api-key"
        ... )
        >>> result = processor.process(pdf_bytes)
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        model: str = "prebuilt-layout",
        features: list[str] | None = None,
    ) -> None:
        """初期化
        
        Args:
            endpoint: Azure Document Intelligence エンドポイント
            api_key: APIキー（Noneの場合はDefaultAzureCredentialを使用）
            model: 使用するモデル
            features: 追加機能（"tables", "figures", "formulas" など）
        """
        self.endpoint = endpoint
        self.model = model
        self.features = features or ["tables", "figures"]
        self._client: DocumentIntelligenceClient | None = None
        self._api_key = api_key
    
    @property
    def client(self) -> "DocumentIntelligenceClient":
        """クライアントを取得（遅延初期化）"""
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            
            if self._api_key:
                from azure.core.credentials import AzureKeyCredential
                credential = AzureKeyCredential(self._api_key)
            else:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
            
            self._client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=credential,
            )
        return self._client
    
    def process(self, content: bytes) -> ProcessedPDF:
        """PDFを処理
        
        Args:
            content: PDFファイル内容
            
        Returns:
            処理されたPDF
        """
        from azure.ai.documentintelligence.models import (
            AnalyzeDocumentRequest,
            DocumentAnalysisFeature,
        )
        
        # 機能フラグをマッピング
        feature_map = {
            "tables": DocumentAnalysisFeature.TABLES if hasattr(DocumentAnalysisFeature, 'TABLES') else None,
            "figures": DocumentAnalysisFeature.FIGURES if hasattr(DocumentAnalysisFeature, 'FIGURES') else None,
            "formulas": DocumentAnalysisFeature.FORMULAS if hasattr(DocumentAnalysisFeature, 'FORMULAS') else None,
        }
        
        features = [
            feature_map[f] for f in self.features
            if f in feature_map and feature_map[f] is not None
        ]
        
        poller = self.client.begin_analyze_document(
            model_id=self.model,
            body=AnalyzeDocumentRequest(bytes_source=content),
            features=features if features else None,
        )
        result = poller.result()
        
        return self._structure_result(result)
    
    def _structure_result(self, result: Any) -> ProcessedPDF:
        """Azure Document Intelligenceの結果を構造化
        
        Args:
            result: Azure Document Intelligenceの解析結果
            
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
        
        # ページごとに処理
        for page in result.pages:
            page_number = page.page_number
            
            # 行を処理
            for line in page.lines or []:
                text = line.content.strip()
                if not text:
                    continue
                
                raw_text_parts.append(text)
                
                # タイトル検出（最初のページの最初の大きなテキスト）
                if not title and page_number == 1:
                    # フォントサイズや位置から推定
                    if self._is_likely_title(line, page):
                        title = text
                        continue
                
                # アブストラクト検出
                if "abstract" in text.lower() and len(text) < 50:
                    in_abstract = True
                    continue
                
                if in_abstract:
                    abstract += text + " "
                    # 次のセクションまでアブストラクト
                    if self._is_section_header(text):
                        in_abstract = False
                        abstract = abstract.strip()
                    continue
                
                # References検出
                if any(kw in text.lower() for kw in ["references", "bibliography"]):
                    in_references = True
                    if current_section:
                        sections.append(current_section)
                        current_section = None
                    continue
                
                if in_references:
                    ref = self._parse_reference(text, len(references) + 1)
                    if ref:
                        references.append(ref)
                    continue
                
                # セクションヘッダー
                if self._is_section_header(text):
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = AcademicSection(
                        heading=text,
                        level=self._detect_heading_level(text),
                        section_type=self._classify_section(text),
                        content="",
                        page_numbers=[page_number],
                    )
                    continue
                
                # 本文
                if current_section:
                    current_section.content += text + "\n"
                    if page_number not in current_section.page_numbers:
                        current_section.page_numbers.append(page_number)
        
        # 最後のセクションを追加
        if current_section:
            sections.append(current_section)
        
        # テーブルを処理
        for idx, table in enumerate(result.tables or []):
            tables.append(self._process_table(table, idx + 1))
        
        # 図を処理（figuresが利用可能な場合）
        if hasattr(result, "figures"):
            for idx, figure in enumerate(result.figures or []):
                figures.append(self._process_figure(figure, idx + 1))
        
        # 識別子を抽出
        raw_text = "\n".join(raw_text_parts)
        doi = self._extract_doi(raw_text)
        arxiv_id = self._extract_arxiv_id(raw_text)
        
        # 言語検出
        language = self._detect_language(raw_text)
        
        # ページ数
        page_count = len(result.pages) if result.pages else 0
        
        return ProcessedPDF(
            file_name="",
            title=title,
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
            abstract=abstract.strip(),
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            keywords=[],
            language=language,
            page_count=page_count,
            raw_text=raw_text,
        )
    
    def _is_likely_title(self, line: Any, page: Any) -> bool:
        """タイトルらしいか判定"""
        text = line.content
        
        # 短すぎるまたは長すぎる
        if len(text) < 10 or len(text) > 300:
            return False
        
        # 最初の行に近い位置
        if hasattr(line, "polygon") and line.polygon:
            y_position = line.polygon[1]  # 上端のY座標
            page_height = page.height if hasattr(page, "height") else 1000
            if y_position > page_height * 0.3:  # ページ上部30%以内
                return False
        
        # 大文字で始まる
        if not text[0].isupper():
            return False
        
        return True
    
    def _is_section_header(self, text: str) -> bool:
        """セクションヘッダーかどうか判定"""
        if len(text) > 100:
            return False
        
        # 番号付きセクション
        if re.match(r"^\d+\.?\s+[A-Z]", text):
            return True
        
        # IMRaDキーワード
        text_lower = text.lower()
        keywords = [
            "introduction", "method", "result", "discussion",
            "conclusion", "related work", "background", "experiment",
            "abstract", "references", "acknowledgment"
        ]
        return any(kw in text_lower for kw in keywords)
    
    def _detect_heading_level(self, text: str) -> int:
        """ヘッダーレベルを検出"""
        match = re.match(r"^(\d+)(\.(\d+))?(\.(\d+))?", text)
        if match:
            if match.group(5):
                return 3
            if match.group(3):
                return 2
            return 1
        return 1
    
    def _process_table(self, table: Any, index: int) -> Table:
        """テーブルを処理"""
        # Markdownテーブル形式に変換
        content_parts = []
        
        if hasattr(table, "cells") and table.cells:
            # ヘッダー行
            header_cells = [c for c in table.cells if c.row_index == 0]
            if header_cells:
                header_row = " | ".join(c.content for c in sorted(header_cells, key=lambda x: x.column_index))
                content_parts.append(f"| {header_row} |")
                content_parts.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
            
            # データ行
            max_row = max(c.row_index for c in table.cells)
            for row_idx in range(1, max_row + 1):
                row_cells = [c for c in table.cells if c.row_index == row_idx]
                row = " | ".join(c.content for c in sorted(row_cells, key=lambda x: x.column_index))
                content_parts.append(f"| {row} |")
        
        return Table(
            table_id=f"table_{index}",
            caption=None,
            content="\n".join(content_parts),
            page_number=table.bounding_regions[0].page_number if table.bounding_regions else 0,
        )
    
    def _process_figure(self, figure: Any, index: int) -> Figure:
        """図を処理"""
        caption = None
        if hasattr(figure, "caption") and figure.caption:
            caption = figure.caption.content
        
        page_number = 0
        if hasattr(figure, "bounding_regions") and figure.bounding_regions:
            page_number = figure.bounding_regions[0].page_number
        
        return Figure(
            figure_id=f"figure_{index}",
            caption=caption,
            image_path=None,
            page_number=page_number,
        )
    
    def _parse_reference(self, text: str, index: int) -> Reference | None:
        """参考文献をパース"""
        if len(text) < 20:
            return None
        
        year_match = re.search(r"\(?(19|20)\d{2}\)?", text)
        year = int(year_match.group().strip("()")) if year_match else None
        
        return Reference(
            ref_id=f"ref_{index}",
            title=text[:200],
            authors=[],
            year=year,
            venue=None,
            doi=self._extract_doi(text),
            arxiv_id=self._extract_arxiv_id(text),
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
            return detect(text[:1000])
        except Exception:
            return "en"

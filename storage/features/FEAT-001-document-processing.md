# FEAT-001: Document Processing

**フィーチャーID**: FEAT-001  
**名称**: ドキュメント処理  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

学術論文を含む多様なドキュメントを入力し、構造化されたTextUnitに変換するフィーチャー。

### 1.1 スコープ

```
入力 → 前処理 → 出力
```

- **入力**: ファイル読み込み、形式検出
- **前処理**: 要素分類、メタデータ抽出、学術論文特有処理
- **出力**: AcademicPaperDocument、TextUnit分割

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-DOC-INP-001 | ファイル形式検出 | P0 |
| FR-DOC-INP-002 | テキストファイル読込 | P0 |
| FR-DOC-INP-003 | 構造化文書読込 | P0 |
| FR-DOC-INP-004 | Office文書読込 | P1 |
| FR-DOC-INP-005 | PDF読込 | P0 |
| FR-DOC-INP-006 | バッチ入力 | P0 |
| FR-DOC-PRE-001 | 要素分類 | P0 |
| FR-DOC-PRE-002 | メタデータ抽出 | P0 |
| FR-DOC-PRE-003 | テーブル変換 | P0 |
| FR-DOC-PRE-004 | 言語検出 | P0 |
| FR-DOC-PRE-005 | テキストクリーニング | P0 |
| FR-DOC-PRE-101 | IMRaD構造認識 | P0 |
| FR-DOC-PRE-102 | 2カラム→1カラム変換 | P0 |
| FR-DOC-PRE-103 | 数式抽出 | P1 |
| FR-DOC-PRE-104 | 図表キャプション抽出 | P0 |
| FR-DOC-PRE-105 | 参考文献抽出 | P0 |
| FR-DOC-PRE-106 | 学術識別子抽出 | P0 |
| FR-DOC-PRE-107 | 著者情報抽出 | P0 |
| FR-DOC-OUT-001 | 標準化文書出力 | P0 |
| FR-DOC-OUT-002 | チャンク分割 | P0 |
| FR-DOC-OUT-003 | オーバーラップ設定 | P0 |
| FR-DOC-OUT-004 | チャンクメタデータ | P0 |

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Document Processing Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ FileLoader   │───▶│ DocumentParser   │───▶│ TextChunker      │  │
│  │              │    │                  │    │                  │  │
│  │ - detect()   │    │ - parse()        │    │ - chunk()        │  │
│  │ - load()     │    │ - extract_meta() │    │ - overlap()      │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│         │                    │                        │             │
│         ▼                    ▼                        ▼             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ RawFile      │    │ AcademicPaper    │    │ TextUnit[]       │  │
│  │              │    │ Document         │    │                  │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    PDF Processing Strategy                    │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐   │  │
│  │  │ Azure Doc Intel     │  │ unstructured (Local)        │   │  │
│  │  │ (Production)        │  │ (Development)               │   │  │
│  │  └─────────────────────┘  └─────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 クラス図

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# === Protocols ===

class FileLoaderProtocol(Protocol):
    """ファイルローダープロトコル"""
    def detect_format(self, path: Path) -> str: ...
    def load(self, path: Path) -> bytes: ...
    def load_batch(self, directory: Path, pattern: str = "*") -> list[tuple[Path, bytes]]: ...

class DocumentParserProtocol(Protocol):
    """ドキュメントパーサープロトコル"""
    def parse(self, content: bytes, file_type: str) -> "AcademicPaperDocument": ...
    def extract_metadata(self, content: bytes, file_type: str) -> dict: ...

class TextChunkerProtocol(Protocol):
    """テキストチャンカープロトコル"""
    def chunk(
        self, 
        document: "AcademicPaperDocument",
        chunk_size: int = 300,
        overlap: int = 100
    ) -> list["TextUnit"]: ...

class PDFProcessorProtocol(Protocol):
    """PDF処理プロトコル（Strategy）"""
    def process(self, content: bytes) -> "ProcessedPDF": ...

# === Data Classes ===

@dataclass
class Author:
    """著者"""
    name: str
    affiliation: str | None = None
    email: str | None = None
    orcid: str | None = None

@dataclass
class AcademicSection:
    """学術論文セクション"""
    heading: str
    level: int
    section_type: str  # introduction, methods, results, discussion, other
    content: str
    page_numbers: list[int]

@dataclass
class Table:
    """テーブル"""
    table_id: str
    caption: str | None
    content: str  # HTML or Markdown
    page_number: int

@dataclass
class Figure:
    """図"""
    figure_id: str
    caption: str | None
    image_path: str | None
    page_number: int

@dataclass
class Reference:
    """参考文献"""
    ref_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None

@dataclass
class AcademicPaperDocument:
    """学術論文ドキュメント"""
    # 基本情報
    file_name: str
    file_type: str
    title: str
    
    # 著者情報
    authors: list[Author]
    
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
    sections: list[AcademicSection] = None
    tables: list[Table] = None
    figures: list[Figure] = None
    
    # 参考文献
    references: list[Reference] = None
    
    # メタデータ
    keywords: list[str] = None
    language: str = "en"
    page_count: int = 0

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
    page_numbers: list[int] = None
    
    # メタデータ
    metadata: dict = None
```

---

## 3. 詳細設計

### 3.1 FileLoader

```python
import mimetypes
from pathlib import Path

class FileLoader:
    """ファイルローダー"""
    
    SUPPORTED_FORMATS = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".xml": "application/xml",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    
    def detect_format(self, path: Path) -> str:
        """ファイル形式を検出"""
        suffix = path.suffix.lower()
        if suffix in self.SUPPORTED_FORMATS:
            return suffix
        
        # MIMEタイプからフォールバック
        mime_type, _ = mimetypes.guess_type(str(path))
        for ext, mt in self.SUPPORTED_FORMATS.items():
            if mt == mime_type:
                return ext
        
        raise ValueError(f"Unsupported file format: {path}")
    
    def load(self, path: Path) -> bytes:
        """ファイルを読み込み"""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_bytes()
    
    def load_batch(
        self, 
        directory: Path, 
        pattern: str = "*",
        recursive: bool = True
    ) -> list[tuple[Path, bytes]]:
        """ディレクトリからバッチ読み込み"""
        glob_method = directory.rglob if recursive else directory.glob
        results = []
        
        for path in glob_method(pattern):
            if path.is_file():
                try:
                    self.detect_format(path)  # サポート確認
                    results.append((path, self.load(path)))
                except ValueError:
                    continue  # 非サポート形式はスキップ
        
        return results
```

### 3.2 DocumentParser

```python
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

class DocumentParser:
    """ドキュメントパーサー"""
    
    def __init__(self, pdf_processor: PDFProcessorProtocol):
        self.pdf_processor = pdf_processor
    
    def parse(self, content: bytes, file_type: str) -> AcademicPaperDocument:
        """ドキュメントをパース"""
        if file_type == ".pdf":
            return self._parse_pdf(content)
        else:
            return self._parse_text(content, file_type)
    
    def _parse_pdf(self, content: bytes) -> AcademicPaperDocument:
        """PDFをパース"""
        processed = self.pdf_processor.process(content)
        
        return AcademicPaperDocument(
            file_name=processed.file_name,
            file_type=".pdf",
            title=processed.title,
            authors=processed.authors,
            doi=processed.doi,
            arxiv_id=processed.arxiv_id,
            abstract=processed.abstract,
            sections=processed.sections,
            tables=processed.tables,
            figures=processed.figures,
            references=processed.references,
            keywords=processed.keywords,
            language=processed.language,
            page_count=processed.page_count,
        )
    
    def _parse_text(self, content: bytes, file_type: str) -> AcademicPaperDocument:
        """テキスト系ファイルをパース"""
        # unstructuredでパース
        elements = partition(
            file=BytesIO(content),
            content_type=FileLoader.SUPPORTED_FORMATS.get(file_type)
        )
        
        # 要素を分類
        title = ""
        sections = []
        current_section = None
        
        for element in elements:
            if element.category == "Title":
                title = element.text
            elif element.category == "Header":
                if current_section:
                    sections.append(current_section)
                current_section = AcademicSection(
                    heading=element.text,
                    level=1,
                    section_type=self._classify_section(element.text),
                    content="",
                    page_numbers=[]
                )
            elif element.category in ("NarrativeText", "ListItem"):
                if current_section:
                    current_section.content += element.text + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return AcademicPaperDocument(
            file_name="",
            file_type=file_type,
            title=title,
            authors=[],
            sections=sections,
        )
    
    def _classify_section(self, heading: str) -> str:
        """セクションタイプを分類（IMRaD）"""
        heading_lower = heading.lower()
        
        if any(kw in heading_lower for kw in ["introduction", "背景", "はじめに"]):
            return "introduction"
        elif any(kw in heading_lower for kw in ["method", "approach", "手法", "方法"]):
            return "methods"
        elif any(kw in heading_lower for kw in ["result", "experiment", "結果", "実験"]):
            return "results"
        elif any(kw in heading_lower for kw in ["discussion", "考察", "議論"]):
            return "discussion"
        elif any(kw in heading_lower for kw in ["conclusion", "結論", "まとめ"]):
            return "conclusion"
        elif any(kw in heading_lower for kw in ["related", "先行研究", "関連研究"]):
            return "related_work"
        else:
            return "other"
```

### 3.3 PDF Processor (Strategy Pattern)

```python
from abc import ABC, abstractmethod

class PDFProcessor(ABC):
    """PDF処理の抽象基底クラス"""
    
    @abstractmethod
    def process(self, content: bytes) -> "ProcessedPDF": ...

@dataclass
class ProcessedPDF:
    """PDF処理結果"""
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


class UnstructuredPDFProcessor(PDFProcessor):
    """unstructured による PDF 処理（ローカル開発用）"""
    
    def __init__(self, strategy: str = "fast"):
        self.strategy = strategy  # fast, hi_res, ocr_only
    
    def process(self, content: bytes) -> ProcessedPDF:
        from unstructured.partition.pdf import partition_pdf
        from io import BytesIO
        
        elements = partition_pdf(
            file=BytesIO(content),
            strategy=self.strategy,
            extract_images_in_pdf=True,
            infer_table_structure=True,
        )
        
        # 要素を分類して構造化
        return self._structure_elements(elements)
    
    def _structure_elements(self, elements) -> ProcessedPDF:
        """要素を構造化"""
        title = ""
        authors = []
        abstract = ""
        sections = []
        tables = []
        figures = []
        references = []
        
        current_section = None
        in_abstract = False
        in_references = False
        
        for element in elements:
            text = element.text.strip()
            
            # タイトル検出
            if element.category == "Title" and not title:
                title = text
                continue
            
            # アブストラクト検出
            if "abstract" in text.lower() and len(text) < 50:
                in_abstract = True
                continue
            
            if in_abstract and element.category == "NarrativeText":
                abstract = text
                in_abstract = False
                continue
            
            # References セクション検出
            if any(kw in text.lower() for kw in ["references", "bibliography"]):
                in_references = True
                continue
            
            # テーブル
            if element.category == "Table":
                tables.append(Table(
                    table_id=f"table_{len(tables)+1}",
                    caption=element.metadata.get("caption"),
                    content=text,
                    page_number=element.metadata.get("page_number", 0)
                ))
                continue
            
            # 図
            if element.category == "Image":
                figures.append(Figure(
                    figure_id=f"figure_{len(figures)+1}",
                    caption=element.metadata.get("caption"),
                    image_path=None,
                    page_number=element.metadata.get("page_number", 0)
                ))
                continue
            
            # セクション
            if element.category == "Header":
                if current_section:
                    sections.append(current_section)
                current_section = AcademicSection(
                    heading=text,
                    level=1,
                    section_type=self._classify_section(text),
                    content="",
                    page_numbers=[]
                )
            elif current_section and element.category in ("NarrativeText", "ListItem"):
                current_section.content += text + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return ProcessedPDF(
            file_name="",
            title=title,
            authors=authors,
            doi=self._extract_doi(elements),
            arxiv_id=self._extract_arxiv_id(elements),
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            keywords=[],
            language="en",
            page_count=max(e.metadata.get("page_number", 0) for e in elements) if elements else 0
        )
    
    def _classify_section(self, heading: str) -> str:
        """セクションタイプを分類"""
        # DocumentParser と同じロジック
        ...
    
    def _extract_doi(self, elements) -> str | None:
        """DOIを抽出"""
        import re
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        
        for element in elements[:10]:  # 最初の要素から検索
            match = re.search(doi_pattern, element.text)
            if match:
                return match.group()
        return None
    
    def _extract_arxiv_id(self, elements) -> str | None:
        """arXiv IDを抽出"""
        import re
        arxiv_pattern = r'arXiv:(\d{4}\.\d{4,5})'
        
        for element in elements[:10]:
            match = re.search(arxiv_pattern, element.text)
            if match:
                return match.group(1)
        return None


class AzureDocIntelPDFProcessor(PDFProcessor):
    """Azure Document Intelligence による PDF 処理（本番用）"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "prebuilt-layout"
    ):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        self.model = model
    
    def process(self, content: bytes) -> ProcessedPDF:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        
        poller = self.client.begin_analyze_document(
            model_id=self.model,
            body=AnalyzeDocumentRequest(bytes_source=content),
            features=["tables", "figures", "formulas"]
        )
        result = poller.result()
        
        return self._structure_result(result)
    
    def _structure_result(self, result) -> ProcessedPDF:
        """Azure Document Intelligence の結果を構造化"""
        # 詳細実装は省略
        ...
```

### 3.4 TextChunker

```python
import tiktoken

class TextChunker:
    """テキストチャンカー"""
    
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def chunk(
        self,
        document: AcademicPaperDocument,
        chunk_size: int = 300,
        overlap: int = 100
    ) -> list[TextUnit]:
        """ドキュメントをチャンクに分割"""
        chunks = []
        chunk_index = 0
        
        # アブストラクトをチャンク
        if document.abstract:
            abstract_chunks = self._split_text(
                document.abstract,
                chunk_size,
                overlap,
                section_type="abstract"
            )
            for text, n_tokens, start, end in abstract_chunks:
                chunks.append(TextUnit(
                    id=f"{document.file_name}_{chunk_index}",
                    text=text,
                    n_tokens=n_tokens,
                    document_id=document.file_name,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    section_type="abstract",
                    metadata={"title": document.title}
                ))
                chunk_index += 1
        
        # 各セクションをチャンク
        for section in document.sections or []:
            section_chunks = self._split_text(
                section.content,
                chunk_size,
                overlap,
                section_type=section.section_type
            )
            for text, n_tokens, start, end in section_chunks:
                chunks.append(TextUnit(
                    id=f"{document.file_name}_{chunk_index}",
                    text=text,
                    n_tokens=n_tokens,
                    document_id=document.file_name,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    section_type=section.section_type,
                    page_numbers=section.page_numbers,
                    metadata={
                        "title": document.title,
                        "section_heading": section.heading
                    }
                ))
                chunk_index += 1
        
        return chunks
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        section_type: str
    ) -> list[tuple[str, int, int, int]]:
        """テキストを分割"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # 文字位置を計算（概算）
            char_start = len(self.tokenizer.decode(tokens[:start]))
            char_end = len(self.tokenizer.decode(tokens[:end]))
            
            chunks.append((chunk_text, len(chunk_tokens), char_start, char_end))
            
            # 次のチャンク開始位置（オーバーラップ考慮）
            start = end - overlap if end < len(tokens) else end
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """トークン数をカウント"""
        return len(self.tokenizer.encode(text))
```

### 3.5 DocumentProcessingPipeline (Facade)

```python
class DocumentProcessingPipeline:
    """ドキュメント処理パイプライン（ファサード）"""
    
    def __init__(
        self,
        pdf_processor: PDFProcessorProtocol | None = None,
        chunk_size: int = 300,
        overlap: int = 100
    ):
        self.file_loader = FileLoader()
        self.pdf_processor = pdf_processor or UnstructuredPDFProcessor()
        self.document_parser = DocumentParser(self.pdf_processor)
        self.text_chunker = TextChunker()
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_file(self, path: Path) -> tuple[AcademicPaperDocument, list[TextUnit]]:
        """単一ファイルを処理"""
        # 1. 形式検出 & 読み込み
        file_type = self.file_loader.detect_format(path)
        content = self.file_loader.load(path)
        
        # 2. パース
        document = self.document_parser.parse(content, file_type)
        document.file_name = path.name
        
        # 3. チャンク分割
        text_units = self.text_chunker.chunk(
            document,
            self.chunk_size,
            self.overlap
        )
        
        return document, text_units
    
    def process_directory(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True
    ) -> list[tuple[AcademicPaperDocument, list[TextUnit]]]:
        """ディレクトリを処理"""
        files = self.file_loader.load_batch(directory, pattern, recursive)
        results = []
        
        for path, content in files:
            try:
                file_type = self.file_loader.detect_format(path)
                document = self.document_parser.parse(content, file_type)
                document.file_name = path.name
                text_units = self.text_chunker.chunk(
                    document,
                    self.chunk_size,
                    self.overlap
                )
                results.append((document, text_units))
            except Exception as e:
                # ログ出力、処理継続
                print(f"Error processing {path}: {e}")
                continue
        
        return results
```

---

## 4. 設定

```yaml
# config/document_processing.yaml

document_processing:
  # チャンク設定
  chunk_size: 300
  overlap: 100
  tokenizer: cl100k_base
  
  # PDF処理設定
  pdf:
    provider: unstructured  # unstructured | azure_document_intelligence
    
    # unstructured 設定
    unstructured:
      strategy: fast  # fast | hi_res | ocr_only
      extract_images: true
      infer_table_structure: true
    
    # Azure Document Intelligence 設定
    azure_document_intelligence:
      model: prebuilt-layout
      api_version: "2024-02-29-preview"
      features:
        - tables
        - figures
        - formulas
  
  # 対応ファイル形式
  supported_formats:
    - .txt
    - .md
    - .json
    - .html
    - .xml
    - .csv
    - .pdf
    - .docx
    - .pptx
    - .xlsx
  
  # 学術論文設定
  academic:
    extract_doi: true
    extract_arxiv_id: true
    extract_references: true
    imrad_recognition: true
    language_detection: true
```

---

## 5. テスト計画

### 5.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_detect_format_txt | FileLoader.detect_format | ".txt" を返す |
| test_detect_format_pdf | FileLoader.detect_format | ".pdf" を返す |
| test_detect_format_unsupported | FileLoader.detect_format | ValueError を raise |
| test_load_file | FileLoader.load | bytes を返す |
| test_load_file_not_found | FileLoader.load | FileNotFoundError を raise |
| test_chunk_text | TextChunker.chunk | chunk_size 以下のチャンクを返す |
| test_chunk_overlap | TextChunker.chunk | オーバーラップが正しい |
| test_parse_text_file | DocumentParser.parse | AcademicPaperDocument を返す |
| test_classify_section_introduction | DocumentParser._classify_section | "introduction" を返す |
| test_extract_doi | PDFProcessor._extract_doi | DOI文字列を返す |

### 5.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_process_pdf_file | DocumentProcessingPipeline | Document + TextUnits を返す |
| test_process_directory | DocumentProcessingPipeline | 複数ファイルを処理 |
| test_azure_pdf_processor | AzureDocIntelPDFProcessor | Azure API 呼び出し成功 |

### 5.3 E2Eテスト

| テストケース | シナリオ | 期待結果 |
|-------------|---------|---------|
| test_academic_paper_pipeline | arXiv論文PDFを入力 | 正しくIMRaD構造を認識 |
| test_batch_processing | 10件の論文を処理 | 全件正常終了、エラーファイルはスキップ |

---

## 6. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-001-01 | FileLoader 実装 | 2h | - |
| TASK-001-02 | TextChunker 実装 | 2h | - |
| TASK-001-03 | DocumentParser 実装 | 3h | TASK-001-01 |
| TASK-001-04 | UnstructuredPDFProcessor 実装 | 4h | TASK-001-03 |
| TASK-001-05 | AzureDocIntelPDFProcessor 実装 | 4h | TASK-001-03 |
| TASK-001-06 | DocumentProcessingPipeline 実装 | 2h | TASK-001-01~05 |
| TASK-001-07 | 単体テスト作成 | 3h | TASK-001-01~06 |
| TASK-001-08 | 統合テスト作成 | 2h | TASK-001-07 |
| **合計** | | **22h** | |

---

## 7. 受入基準

- [ ] 全対応ファイル形式を読み込める
- [ ] PDFの2カラムレイアウトを正しく1カラムに変換できる
- [ ] IMRaD構造（Introduction/Methods/Results/Discussion）を認識できる
- [ ] DOI、arXiv IDを正しく抽出できる
- [ ] 参考文献を構造化して抽出できる
- [ ] 設定可能なチャンクサイズ・オーバーラップでTextUnitを生成できる
- [ ] 100MB以上のドキュメントセットを処理できる
- [ ] ローカル（unstructured）と本番（Azure）の切り替えが可能

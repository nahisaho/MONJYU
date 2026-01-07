# MONJYU コンポーネント仕様書 v3.2

**文書番号**: MONJYU-COMP-003  
**バージョン**: 3.2.0  
**作成日**: 2026-01-07  
**ステータス**: Approved  
**準拠要件**: [01_requirements_v3.md](01_requirements_v3.md) v3.1.0 (Approved)  
**準拠アーキテクチャ**: [02_architecture_v3.md](02_architecture_v3.md) v3.2.0 (Approved)  
**テスト状況**: 2,417テスト / 83% カバレッジ (2026-01-07時点)

---

## 1. 概要

### 1.1 目的

本書は、MONJYUシステムを構成する各コンポーネントの詳細仕様を定義します。
学術論文（AI for Science）を対象とした Progressive GraphRAG システムのコンポーネント設計を記述します。

### 1.2 対象読者

- 開発者
- アーキテクト
- テスト担当者

### 1.3 コンポーネント一覧（実装ステータス付き）

| カテゴリ | コンポーネント | 対応要件 | 実装ステータス |
|---------|---------------|---------|---------------|
| ドキュメント処理 | PDFProcessor | REQ-IDX-001a | ✅ 実装済 |
| ドキュメント処理 | AcademicPaperParser | REQ-IDX-001a | ✅ 実装済 |
| ドキュメント処理 | PreprocessingPipeline | REQ-IDX-001c | ✅ 実装済 |
| Index | TextChunker | REQ-IDX-002 | ✅ 実装済 |
| Index | Embedder | REQ-IDX-003 | ✅ 実装済 |
| Index | NLPExtractor | REQ-IDX-004 | ✅ 実装済 |
| Index | EntityExtractor | REQ-IDX-005 | ✅ 実装済 |
| Index | RelationshipExtractor | REQ-IDX-006 | ✅ 実装済 |
| Index | CommunityDetector | REQ-IDX-007 | ✅ 実装済 |
| Index | ReportGenerator | REQ-IDX-008 | ✅ 実装済 |
| Index | ProgressiveIndexManager | REQ-IDX-009 | ✅ 実装済 |
| Query | VectorSearch | REQ-QRY-001 | ✅ 実装済 |
| Query | GlobalSearch | REQ-QRY-002 | ✅ 実装済 |
| Query | LocalSearch | REQ-QRY-003 | ✅ 実装済 |
| Query | LazySearch | REQ-QRY-004 | ✅ 実装済 |
| Query | HybridSearch | REQ-QRY-005 | ✅ 実装済 |
| Query | QueryRouter | REQ-QRY-006 | ✅ 実装済 |
| Controller | UnifiedController | REQ-ARC-001 | ✅ 実装済 |
| Controller | ProgressiveController | REQ-ARC-002 | ✅ 実装済 |
| Controller | HybridController | REQ-ARC-003 | ✅ 実装済 |
| Citation | CitationNetworkBuilder | REQ-IDX-005a | ✅ 実装済 |
| Citation | CoCitationAnalyzer | REQ-IDX-005a | ✅ 実装済 |
| Storage | FileStorage | REQ-STG-001 | ✅ 実装済 |
| Storage | VectorStore | REQ-STG-002 | ✅ 実装済 |
| Storage | CacheManager | REQ-STG-003 | ✅ 実装済 |
| LLM | AzureOpenAIClient | 環境設定 | ✅ 実装済 |
| LLM | OllamaClient | 環境設定 | ✅ 実装済 |
| API | MONJYUFacade | REQ-API-001 | ✅ 実装済 |
| API | CLI | REQ-API-002 | ✅ 実装済 |
| API | MCPServer | REQ-API-004 | ✅ 実装済 |
| External | SemanticScholarClient | NFR-INT-001 | ✅ 実装済 |
| External | CrossRefClient | NFR-INT-001 | ✅ 実装済 |

**実装サマリ**: 32/32 コンポーネント完了 (100%)

### 1.4 レイヤー構成

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │   CLI   │  │  API    │  │   MCP   │  │  REST (Future)  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MONJYU Facade                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐   │
│  │  Unified   │  │Progressive │  │       Hybrid           │   │
│  │ Controller │  │ Controller │  │      Controller        │   │
│  └────────────┘  └────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Domain Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │    Index     │  │    Query     │  │     Citation       │   │
│  │   Domain     │  │   Domain     │  │      Domain        │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │   LLM   │  │ Storage │  │External │  │   PDF Process   │   │
│  │ Clients │  │         │  │  APIs   │  │                 │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. ドキュメント処理コンポーネント群

### 2.1 PDFProcessor

学術論文PDF処理コンポーネント。環境に応じてAzure Document IntelligenceまたはUnstructuredを使用。

#### 2.1.1 インターフェース定義

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ProcessorType(Enum):
    """PDF処理エンジンタイプ"""
    AZURE_DOCUMENT_INTELLIGENCE = "azure_di"
    UNSTRUCTURED = "unstructured"


@dataclass
class PDFElement:
    """PDF要素の基底クラス"""
    page_number: int
    bounding_box: Optional[tuple[float, float, float, float]] = None


@dataclass
class TextBlock(PDFElement):
    """テキストブロック"""
    content: str
    confidence: float = 1.0


@dataclass
class Table(PDFElement):
    """表"""
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None


@dataclass
class Figure(PDFElement):
    """図"""
    caption: Optional[str] = None
    image_data: Optional[bytes] = None


@dataclass
class Equation(PDFElement):
    """数式"""
    latex: str
    inline: bool = False


@dataclass
class PDFProcessResult:
    """PDF処理結果"""
    file_path: str
    text_blocks: List[TextBlock]
    tables: List[Table]
    figures: List[Figure]
    equations: List[Equation]
    metadata: dict = field(default_factory=dict)
    page_count: int = 0
    processing_time_ms: float = 0


class PDFProcessorProtocol(ABC):
    """PDF処理プロトコル"""
    
    @abstractmethod
    async def process(self, file_path: str) -> PDFProcessResult:
        """PDFファイルを処理"""
        ...
    
    @abstractmethod
    async def process_batch(
        self, 
        file_paths: List[str],
        max_concurrent: int = 5
    ) -> List[PDFProcessResult]:
        """複数PDFを並列処理"""
        ...
    
    @property
    @abstractmethod
    def processor_type(self) -> ProcessorType:
        """処理エンジンタイプ"""
        ...
```

#### 2.1.2 Azure Document Intelligence 実装

```python
class AzureDocumentIntelligenceProcessor(PDFProcessorProtocol):
    """Azure Document Intelligence 実装（本番環境用）"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_id: str = "prebuilt-layout"
    ):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        self.model_id = model_id
    
    @property
    def processor_type(self) -> ProcessorType:
        return ProcessorType.AZURE_DOCUMENT_INTELLIGENCE
    
    async def process(self, file_path: str) -> PDFProcessResult:
        """
        Azure Document Intelligence でPDFを処理
        
        Features:
        - 高精度なレイアウト解析
        - 表・数式の構造化抽出
        - 多言語対応
        """
        import asyncio
        from pathlib import Path
        import time
        
        start_time = time.time()
        
        with open(file_path, "rb") as f:
            # 同期APIを非同期で実行
            poller = await asyncio.to_thread(
                self.client.begin_analyze_document,
                self.model_id,
                f,
                content_type="application/pdf"
            )
            result = await asyncio.to_thread(poller.result)
        
        # 結果をPDFProcessResultに変換
        text_blocks = []
        tables = []
        figures = []
        equations = []
        
        for page in result.pages:
            for line in page.lines:
                text_blocks.append(TextBlock(
                    page_number=page.page_number,
                    content=line.content,
                    confidence=line.confidence if hasattr(line, 'confidence') else 1.0,
                    bounding_box=self._convert_bbox(line.polygon)
                ))
        
        for table in result.tables or []:
            tables.append(self._convert_table(table))
        
        processing_time = (time.time() - start_time) * 1000
        
        return PDFProcessResult(
            file_path=file_path,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            equations=equations,
            page_count=len(result.pages),
            processing_time_ms=processing_time
        )
    
    def _convert_bbox(self, polygon) -> Optional[tuple]:
        """ポリゴンをバウンディングボックスに変換"""
        if not polygon:
            return None
        xs = [p.x for p in polygon]
        ys = [p.y for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def _convert_table(self, table) -> Table:
        """Azure形式のテーブルを変換"""
        # 実装詳細
        ...
```

#### 2.1.3 Unstructured 実装

```python
class UnstructuredProcessor(PDFProcessorProtocol):
    """Unstructured 実装（ローカル環境用）"""
    
    def __init__(
        self,
        strategy: str = "hi_res",
        languages: List[str] = None
    ):
        self.strategy = strategy
        self.languages = languages or ["eng", "jpn"]
    
    @property
    def processor_type(self) -> ProcessorType:
        return ProcessorType.UNSTRUCTURED
    
    async def process(self, file_path: str) -> PDFProcessResult:
        """
        Unstructured でPDFを処理
        
        Features:
        - ローカル実行可能
        - オープンソース
        - カスタマイズ可能
        """
        import asyncio
        import time
        from unstructured.partition.pdf import partition_pdf
        
        start_time = time.time()
        
        # 同期APIを非同期で実行
        elements = await asyncio.to_thread(
            partition_pdf,
            filename=file_path,
            strategy=self.strategy,
            languages=self.languages,
            extract_images_in_pdf=True,
            infer_table_structure=True
        )
        
        text_blocks = []
        tables = []
        figures = []
        equations = []
        
        for element in elements:
            element_type = type(element).__name__
            
            if element_type == "Table":
                tables.append(self._convert_table(element))
            elif element_type == "Image":
                figures.append(self._convert_figure(element))
            elif element_type == "Formula":
                equations.append(self._convert_equation(element))
            else:
                text_blocks.append(TextBlock(
                    page_number=element.metadata.page_number or 1,
                    content=str(element),
                    confidence=1.0
                ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return PDFProcessResult(
            file_path=file_path,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            equations=equations,
            page_count=self._count_pages(elements),
            processing_time_ms=processing_time
        )
```

#### 2.1.4 ファクトリー

```python
class PDFProcessorFactory:
    """PDF処理エンジンファクトリー"""
    
    @staticmethod
    def create(
        processor_type: ProcessorType,
        config: dict
    ) -> PDFProcessorProtocol:
        """
        環境に応じたPDF処理エンジンを生成
        
        Args:
            processor_type: 処理エンジンタイプ
            config: エンジン固有設定
        
        Returns:
            PDFProcessorProtocol 実装
        """
        if processor_type == ProcessorType.AZURE_DOCUMENT_INTELLIGENCE:
            return AzureDocumentIntelligenceProcessor(
                endpoint=config["endpoint"],
                api_key=config["api_key"],
                model_id=config.get("model_id", "prebuilt-layout")
            )
        elif processor_type == ProcessorType.UNSTRUCTURED:
            return UnstructuredProcessor(
                strategy=config.get("strategy", "hi_res"),
                languages=config.get("languages", ["eng", "jpn"])
            )
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
    
    @staticmethod
    def create_from_environment() -> PDFProcessorProtocol:
        """環境変数から自動選択"""
        import os
        
        if os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"):
            return PDFProcessorFactory.create(
                ProcessorType.AZURE_DOCUMENT_INTELLIGENCE,
                {
                    "endpoint": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
                    "api_key": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
                }
            )
        else:
            return PDFProcessorFactory.create(
                ProcessorType.UNSTRUCTURED,
                {}
            )
```

---

### 2.2 AcademicPaperParser

学術論文構造解析コンポーネント。IMRaD構造・引用・メタデータを抽出。

#### 2.2.1 データモデル

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import date


@dataclass
class Author:
    """著者情報"""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None
    is_corresponding: bool = False


@dataclass
class Section:
    """論文セクション"""
    heading: str
    level: int  # 1=H1, 2=H2, etc.
    section_type: Literal[
        "abstract",
        "introduction", 
        "related_work",
        "methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "acknowledgments",
        "references",
        "appendix",
        "other"
    ]
    content: str
    subsections: List['Section'] = field(default_factory=list)
    start_page: Optional[int] = None
    end_page: Optional[int] = None


@dataclass
class Reference:
    """参考文献"""
    ref_id: str  # [1], [Smith2020] など
    raw_text: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


@dataclass
class InlineCitation:
    """本文中の引用"""
    ref_id: str
    context: str  # 引用周辺のテキスト
    section: str  # 引用が出現するセクション
    position: int  # 文書内の位置


@dataclass
class AcademicPaper:
    """学術論文"""
    # === 識別子 ===
    id: str
    file_path: str
    
    # === 基本情報 ===
    title: str
    authors: List[Author]
    abstract: str
    
    # === 外部識別子 ===
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    
    # === 出版情報 ===
    publication_date: Optional[date] = None
    venue: Optional[str] = None
    venue_type: Literal["journal", "conference", "preprint", "other"] = "other"
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    
    # === 構造化コンテンツ ===
    sections: List[Section] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    
    # === 引用情報 ===
    references: List[Reference] = field(default_factory=list)
    inline_citations: List[InlineCitation] = field(default_factory=list)
    
    # === メタデータ ===
    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    page_count: int = 0
    
    # === 外部メトリクス（後から付与） ===
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    
    def get_full_text(self) -> str:
        """全文テキストを取得"""
        return "\n\n".join(
            section.content for section in self.sections
        )
    
    def get_section_by_type(
        self, 
        section_type: str
    ) -> Optional[Section]:
        """セクションタイプで検索"""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
```

#### 2.2.2 パーサー実装

```python
class AcademicPaperParser:
    """学術論文パーサー"""
    
    def __init__(
        self,
        pdf_processor: PDFProcessorProtocol,
        llm_client: Optional['ChatModelProtocol'] = None
    ):
        self.pdf_processor = pdf_processor
        self.llm_client = llm_client  # メタデータ抽出用（オプション）
    
    async def parse(self, file_path: str) -> AcademicPaper:
        """
        学術論文PDFを解析
        
        処理フロー:
        1. PDF処理（テキスト・表・図抽出）
        2. セクション構造解析（IMRaD）
        3. 参考文献解析
        4. メタデータ抽出（タイトル・著者等）
        5. 本文中引用の検出
        """
        # 1. PDF処理
        pdf_result = await self.pdf_processor.process(file_path)
        
        # 2. 全文テキスト構築
        full_text = self._build_full_text(pdf_result)
        
        # 3. セクション構造解析
        sections = self._parse_sections(full_text)
        
        # 4. 参考文献解析
        references = self._parse_references(pdf_result, sections)
        
        # 5. メタデータ抽出
        metadata = await self._extract_metadata(
            pdf_result, sections, full_text
        )
        
        # 6. 本文中引用検出
        inline_citations = self._detect_inline_citations(
            sections, references
        )
        
        return AcademicPaper(
            id=self._generate_id(file_path),
            file_path=file_path,
            title=metadata["title"],
            authors=metadata["authors"],
            abstract=metadata["abstract"],
            doi=metadata.get("doi"),
            arxiv_id=metadata.get("arxiv_id"),
            sections=sections,
            tables=pdf_result.tables,
            figures=pdf_result.figures,
            equations=pdf_result.equations,
            references=references,
            inline_citations=inline_citations,
            keywords=metadata.get("keywords", []),
            page_count=pdf_result.page_count
        )
    
    def _parse_sections(self, text: str) -> List[Section]:
        """
        IMRaD構造を解析
        
        パターン:
        - Abstract / 概要
        - Introduction / はじめに / 1. Introduction
        - Related Work / 関連研究 / 2. Related Work
        - Methods / 手法 / Methodology / 3. Proposed Method
        - Experiments / 実験 / 4. Experiments
        - Results / 結果 / 5. Results
        - Discussion / 考察 / 6. Discussion
        - Conclusion / 結論 / 7. Conclusion
        - References / 参考文献
        """
        import re
        
        # セクションヘッダーパターン
        section_patterns = {
            "abstract": r"(?i)^(abstract|概要)\s*$",
            "introduction": r"(?i)^(\d+\.?\s*)?(introduction|はじめに|序論)\s*$",
            "related_work": r"(?i)^(\d+\.?\s*)?(related\s+work|関連研究|先行研究)\s*$",
            "methods": r"(?i)^(\d+\.?\s*)?(method|手法|methodology|proposed|提案手法)\s*",
            "experiments": r"(?i)^(\d+\.?\s*)?(experiment|実験|evaluation|評価)\s*$",
            "results": r"(?i)^(\d+\.?\s*)?(result|結果)\s*$",
            "discussion": r"(?i)^(\d+\.?\s*)?(discussion|考察|議論)\s*$",
            "conclusion": r"(?i)^(\d+\.?\s*)?(conclusion|結論|まとめ)\s*$",
            "references": r"(?i)^(references|参考文献|bibliography)\s*$",
        }
        
        sections = []
        # 実装詳細...
        return sections
    
    def _parse_references(
        self,
        pdf_result: PDFProcessResult,
        sections: List[Section]
    ) -> List[Reference]:
        """参考文献を解析"""
        import re
        
        # Referencesセクションを特定
        ref_section = None
        for section in sections:
            if section.section_type == "references":
                ref_section = section
                break
        
        if not ref_section:
            return []
        
        references = []
        
        # 参考文献パターン（番号形式）
        numbered_pattern = r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)'
        
        # 参考文献パターン（著者年形式）
        author_year_pattern = r'([A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?)\s*\((\d{4})\)'
        
        matches = re.findall(numbered_pattern, ref_section.content, re.DOTALL)
        
        for ref_id, raw_text in matches:
            ref = self._parse_single_reference(ref_id, raw_text.strip())
            references.append(ref)
        
        return references
    
    def _parse_single_reference(
        self,
        ref_id: str,
        raw_text: str
    ) -> Reference:
        """単一の参考文献を解析"""
        import re
        
        # DOI抽出
        doi_match = re.search(r'10\.\d{4,}/[^\s]+', raw_text)
        doi = doi_match.group(0) if doi_match else None
        
        # arXiv ID抽出
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', raw_text)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        
        # 年抽出
        year_match = re.search(r'\((\d{4})\)', raw_text)
        year = int(year_match.group(1)) if year_match else None
        
        return Reference(
            ref_id=ref_id,
            raw_text=raw_text,
            doi=doi,
            arxiv_id=arxiv_id,
            year=year
        )
    
    async def _extract_metadata(
        self,
        pdf_result: PDFProcessResult,
        sections: List[Section],
        full_text: str
    ) -> dict:
        """メタデータを抽出"""
        # 最初のページからタイトル・著者を抽出
        first_page_blocks = [
            b for b in pdf_result.text_blocks 
            if b.page_number == 1
        ]
        
        # ヒューリスティック: 最初の大きなテキストブロックがタイトル
        title = first_page_blocks[0].content if first_page_blocks else ""
        
        # Abstract抽出
        abstract = ""
        for section in sections:
            if section.section_type == "abstract":
                abstract = section.content
                break
        
        # LLMを使用した高精度抽出（オプション）
        if self.llm_client:
            return await self._extract_metadata_with_llm(full_text)
        
        return {
            "title": title,
            "authors": [],
            "abstract": abstract,
        }
```

---

### 2.3 PreprocessingPipeline

ドキュメント前処理パイプライン。複数形式対応。

#### 2.3.1 パイプライン定義

```python
from dataclasses import dataclass
from typing import List, Union, Callable
from pathlib import Path
from enum import Enum


class DocumentType(Enum):
    """ドキュメントタイプ"""
    PDF = "pdf"
    WORD = "word"
    POWERPOINT = "powerpoint"
    TEXT = "text"
    MARKDOWN = "markdown"


@dataclass
class ProcessedDocument:
    """前処理済みドキュメント"""
    id: str
    source_path: str
    document_type: DocumentType
    content: Union[AcademicPaper, str]  # 論文 or プレーンテキスト
    is_academic_paper: bool
    processing_time_ms: float
    errors: List[str] = field(default_factory=list)


class PreprocessingPipeline:
    """ドキュメント前処理パイプライン"""
    
    def __init__(
        self,
        pdf_processor: PDFProcessorProtocol,
        paper_parser: AcademicPaperParser,
        word_processor: Optional['WordProcessor'] = None,
        pptx_processor: Optional['PowerPointProcessor'] = None
    ):
        self.pdf_processor = pdf_processor
        self.paper_parser = paper_parser
        self.word_processor = word_processor
        self.pptx_processor = pptx_processor
        
        self._processors = {
            DocumentType.PDF: self._process_pdf,
            DocumentType.WORD: self._process_word,
            DocumentType.POWERPOINT: self._process_powerpoint,
            DocumentType.TEXT: self._process_text,
            DocumentType.MARKDOWN: self._process_markdown,
        }
    
    async def process(
        self,
        file_path: str,
        force_type: Optional[DocumentType] = None
    ) -> ProcessedDocument:
        """
        単一ドキュメントを処理
        
        Args:
            file_path: ファイルパス
            force_type: ドキュメントタイプを強制指定
        
        Returns:
            ProcessedDocument
        """
        import time
        
        start_time = time.time()
        path = Path(file_path)
        
        # ドキュメントタイプ判定
        doc_type = force_type or self._detect_type(path)
        
        # 処理実行
        processor = self._processors.get(doc_type)
        if not processor:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        try:
            content, is_academic = await processor(file_path)
            errors = []
        except Exception as e:
            content = ""
            is_academic = False
            errors = [str(e)]
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessedDocument(
            id=self._generate_id(file_path),
            source_path=file_path,
            document_type=doc_type,
            content=content,
            is_academic_paper=is_academic,
            processing_time_ms=processing_time,
            errors=errors
        )
    
    async def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: List[str] = None,
        max_concurrent: int = 5
    ) -> List[ProcessedDocument]:
        """
        ディレクトリ内のドキュメントを一括処理
        """
        import asyncio
        
        extensions = extensions or [".pdf", ".docx", ".pptx", ".txt", ".md"]
        path = Path(directory)
        
        # ファイル一覧取得
        if recursive:
            files = [
                f for f in path.rglob("*") 
                if f.suffix.lower() in extensions
            ]
        else:
            files = [
                f for f in path.glob("*") 
                if f.suffix.lower() in extensions
            ]
        
        # 並列処理
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process(str(file_path))
        
        tasks = [process_with_semaphore(f) for f in files]
        return await asyncio.gather(*tasks)
    
    def _detect_type(self, path: Path) -> DocumentType:
        """ファイル拡張子からドキュメントタイプを判定"""
        suffix = path.suffix.lower()
        mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.WORD,
            ".doc": DocumentType.WORD,
            ".pptx": DocumentType.POWERPOINT,
            ".ppt": DocumentType.POWERPOINT,
            ".txt": DocumentType.TEXT,
            ".md": DocumentType.MARKDOWN,
        }
        return mapping.get(suffix, DocumentType.TEXT)
    
    async def _process_pdf(
        self,
        file_path: str
    ) -> tuple[Union[AcademicPaper, str], bool]:
        """PDF処理"""
        # 学術論文判定
        is_academic = await self._is_academic_paper(file_path)
        
        if is_academic:
            paper = await self.paper_parser.parse(file_path)
            return paper, True
        else:
            result = await self.pdf_processor.process(file_path)
            text = "\n".join(b.content for b in result.text_blocks)
            return text, False
    
    async def _is_academic_paper(self, file_path: str) -> bool:
        """学術論文かどうかを判定"""
        # ヒューリスティック:
        # - "Abstract"セクションがある
        # - "References"セクションがある
        # - DOIまたはarXiv IDがある
        # - 著者・所属情報がある
        
        result = await self.pdf_processor.process(file_path)
        text = "\n".join(b.content for b in result.text_blocks[:50])  # 最初の50ブロック
        
        import re
        
        has_abstract = bool(re.search(r'(?i)\babstract\b', text))
        has_references = bool(re.search(r'(?i)\breferences\b', text))
        has_doi = bool(re.search(r'10\.\d{4,}/', text))
        has_arxiv = bool(re.search(r'arXiv:', text))
        
        score = sum([has_abstract, has_references, has_doi or has_arxiv])
        return score >= 2
```

---

## 3. Indexコンポーネント群

### 3.1 TextChunker

テキストチャンク分割コンポーネント。学術論文構造を考慮した分割をサポート。

#### 3.1.1 インターフェース定義

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum


class ChunkingStrategy(Enum):
    """チャンキング戦略"""
    FIXED_SIZE = "fixed_size"        # 固定サイズ
    SENTENCE = "sentence"            # 文単位
    PARAGRAPH = "paragraph"          # 段落単位
    SECTION = "section"              # セクション単位
    SEMANTIC = "semantic"            # セマンティック分割


@dataclass
class TextChunk:
    """テキストチャンク"""
    id: str
    content: str
    source_id: str                   # 元文書ID
    
    # 位置情報
    start_offset: int
    end_offset: int
    
    # トークン情報
    token_count: int
    
    # メタデータ
    metadata: dict = field(default_factory=dict)
    
    # 学術論文固有
    section_type: Optional[str] = None   # introduction, methods, etc.
    page_number: Optional[int] = None
    paper_id: Optional[str] = None


@dataclass
class ChunkerConfig:
    """チャンカー設定"""
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 300            # トークン数
    chunk_overlap: int = 100         # オーバーラップ
    encoding_model: str = "cl100k_base"
    
    # セクション分割オプション
    respect_section_boundaries: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 500


class TextChunkerProtocol(ABC):
    """テキストチャンカープロトコル"""
    
    @abstractmethod
    def chunk(
        self,
        text: str,
        source_id: str,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """テキストをチャンクに分割"""
        ...
    
    @abstractmethod
    def chunk_paper(
        self,
        paper: AcademicPaper
    ) -> List[TextChunk]:
        """学術論文をチャンクに分割（構造考慮）"""
        ...
    
    @abstractmethod
    def chunk_batch(
        self,
        documents: List[dict]
    ) -> List[TextChunk]:
        """複数文書を一括チャンク分割"""
        ...
```

#### 3.1.2 実装

```python
class TiktokenChunker(TextChunkerProtocol):
    """tiktoken基づくチャンク分割"""
    
    def __init__(self, config: ChunkerConfig):
        import tiktoken
        
        self.config = config
        self._encoder = tiktoken.get_encoding(config.encoding_model)
    
    def chunk(
        self,
        text: str,
        source_id: str,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """固定サイズチャンク分割"""
        tokens = self._encoder.encode(text)
        chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            end = min(start + self.config.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            chunk_text = self._encoder.decode(chunk_tokens)
            
            # 文字オフセット計算
            char_start = len(self._encoder.decode(tokens[:start]))
            char_end = char_start + len(chunk_text)
            
            chunks.append(TextChunk(
                id=f"{source_id}_chunk_{chunk_idx}",
                content=chunk_text,
                source_id=source_id,
                start_offset=char_start,
                end_offset=char_end,
                token_count=len(chunk_tokens),
                metadata=metadata or {}
            ))
            
            start = end - self.config.chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def chunk_paper(self, paper: AcademicPaper) -> List[TextChunk]:
        """
        学術論文をセクション構造を考慮してチャンク分割
        
        Strategy:
        1. セクションごとに分割
        2. セクション内で固定サイズ分割
        3. セクションタイプをメタデータに付与
        """
        chunks = []
        
        for section in paper.sections:
            section_chunks = self._chunk_section(
                section,
                paper.id,
                base_metadata={
                    "paper_title": paper.title,
                    "paper_doi": paper.doi,
                }
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_section(
        self,
        section: Section,
        paper_id: str,
        base_metadata: dict
    ) -> List[TextChunk]:
        """セクションをチャンク分割"""
        chunks = []
        
        # セクションが小さければそのまま1チャンク
        tokens = self._encoder.encode(section.content)
        if len(tokens) <= self.config.max_chunk_size:
            chunks.append(TextChunk(
                id=f"{paper_id}_{section.section_type}_0",
                content=section.content,
                source_id=paper_id,
                start_offset=0,
                end_offset=len(section.content),
                token_count=len(tokens),
                metadata={
                    **base_metadata,
                    "section_heading": section.heading,
                },
                section_type=section.section_type,
                paper_id=paper_id
            ))
        else:
            # 固定サイズで分割
            base_chunks = self.chunk(
                section.content,
                f"{paper_id}_{section.section_type}",
                metadata={
                    **base_metadata,
                    "section_heading": section.heading,
                }
            )
            for c in base_chunks:
                c.section_type = section.section_type
                c.paper_id = paper_id
            chunks.extend(base_chunks)
        
        # サブセクションも処理
        for subsection in section.subsections:
            sub_chunks = self._chunk_section(
                subsection, paper_id, base_metadata
            )
            chunks.extend(sub_chunks)
        
        return chunks
```

---

### 3.2 Embedder

ベクトルエンベディング生成コンポーネント。Azure OpenAI / Ollama 対応。

#### 3.2.1 インターフェース定義

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class EmbeddingResult:
    """エンベディング結果"""
    id: str
    vector: np.ndarray
    model: str
    dimensions: int
    token_count: int


@dataclass
class EmbedderConfig:
    """エンベディング設定"""
    model: str
    dimensions: int
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 60.0


class EmbedderProtocol(ABC):
    """エンベディングプロトコル"""
    
    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """単一テキストをエンベディング"""
        ...
    
    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """複数テキストを一括エンベディング"""
        ...
    
    @abstractmethod
    async def embed_chunks(
        self,
        chunks: List[TextChunk]
    ) -> List[EmbeddingResult]:
        """チャンクをエンベディング"""
        ...
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """エンベディング次元数"""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """モデル名"""
        ...
```

#### 3.2.2 Azure OpenAI 実装

```python
class AzureOpenAIEmbedder(EmbedderProtocol):
    """Azure OpenAI Embeddings 実装（本番環境）"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-02-15-preview",
        config: EmbedderConfig = None
    ):
        from openai import AsyncAzureOpenAI
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment = deployment
        self.config = config or EmbedderConfig(
            model=deployment,
            dimensions=3072  # text-embedding-3-large default
        )
    
    @property
    def dimensions(self) -> int:
        return self.config.dimensions
    
    @property
    def model_name(self) -> str:
        return self.deployment
    
    async def embed(self, text: str) -> np.ndarray:
        """単一テキストをエンベディング"""
        response = await self.client.embeddings.create(
            model=self.deployment,
            input=text,
            dimensions=self.config.dimensions
        )
        return np.array(response.data[0].embedding)
    
    async def embed_batch(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """バッチエンベディング"""
        results = []
        
        # バッチサイズで分割
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            response = await self.client.embeddings.create(
                model=self.deployment,
                input=batch,
                dimensions=self.config.dimensions
            )
            
            for item in response.data:
                results.append(np.array(item.embedding))
        
        return results
    
    async def embed_chunks(
        self,
        chunks: List[TextChunk]
    ) -> List[EmbeddingResult]:
        """チャンクをエンベディング"""
        texts = [c.content for c in chunks]
        vectors = await self.embed_batch(texts)
        
        return [
            EmbeddingResult(
                id=chunk.id,
                vector=vector,
                model=self.deployment,
                dimensions=self.config.dimensions,
                token_count=chunk.token_count
            )
            for chunk, vector in zip(chunks, vectors)
        ]
```

#### 3.2.3 Ollama 実装

```python
class OllamaEmbedder(EmbedderProtocol):
    """Ollama Embeddings 実装（ローカル環境）"""
    
    def __init__(
        self,
        host: str = "http://192.168.224.1:11434",
        model: str = "nomic-embed-text",
        config: EmbedderConfig = None
    ):
        import httpx
        
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=60.0)
        self.config = config or EmbedderConfig(
            model=model,
            dimensions=768  # nomic-embed-text default
        )
        self._dimensions = None
    
    @property
    def dimensions(self) -> int:
        return self._dimensions or self.config.dimensions
    
    @property
    def model_name(self) -> str:
        return self.model
    
    async def embed(self, text: str) -> np.ndarray:
        """単一テキストをエンベディング"""
        response = await self._client.post(
            f"{self.host}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )
        response.raise_for_status()
        data = response.json()
        
        embedding = np.array(data["embedding"])
        self._dimensions = len(embedding)
        return embedding
    
    async def embed_batch(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """バッチエンベディング（Ollamaは1つずつ）"""
        import asyncio
        
        # Ollamaは並列リクエストに制限があるため、逐次処理
        results = []
        for text in texts:
            vector = await self.embed(text)
            results.append(vector)
        
        return results
```

#### 3.2.4 ファクトリー

```python
class EmbedderFactory:
    """エンベディングファクトリー"""
    
    @staticmethod
    def create(
        provider: str,
        config: dict
    ) -> EmbedderProtocol:
        """
        プロバイダーに応じたEmbedderを生成
        
        Args:
            provider: "azure_openai" | "ollama" | "openai"
            config: プロバイダー固有設定
        """
        if provider == "azure_openai":
            return AzureOpenAIEmbedder(
                endpoint=config["endpoint"],
                api_key=config["api_key"],
                deployment=config["deployment"],
                api_version=config.get("api_version", "2024-02-15-preview")
            )
        elif provider == "ollama":
            return OllamaEmbedder(
                host=config.get("host", "http://192.168.224.1:11434"),
                model=config.get("model", "nomic-embed-text")
            )
        else:
            raise ValueError(f"Unknown embedder provider: {provider}")
    
    @staticmethod
    def create_from_environment() -> EmbedderProtocol:
        """環境変数から自動生成"""
        import os
        
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            return EmbedderFactory.create("azure_openai", {
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", 
                                        "text-embedding-3-large")
            })
        else:
            return EmbedderFactory.create("ollama", {
                "host": os.getenv("OLLAMA_HOST", "http://192.168.224.1:11434"),
                "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            })
```

---

### 3.3 NLPExtractor

NLPベースの軽量インデックス抽出（LazyGraphRAG用）。

#### 3.3.1 インターフェース定義

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import networkx as nx


@dataclass
class NounPhrase:
    """名詞句"""
    text: str
    lemma: str               # 原形
    count: int               # 出現回数
    chunk_ids: List[str]     # 出現チャンクID
    tfidf_score: float = 0.0


@dataclass
class NLPFeatures:
    """NLP抽出結果"""
    chunk_id: str
    noun_phrases: List[str]
    entities: List[dict]     # {"text": str, "label": str}
    keywords: List[str]
    language: str = "en"


@dataclass
class NounGraph:
    """名詞句共起グラフ"""
    graph: nx.Graph
    noun_phrases: List[NounPhrase]
    
    def get_top_phrases(self, n: int = 100) -> List[NounPhrase]:
        """上位N個の名詞句を取得"""
        return sorted(
            self.noun_phrases,
            key=lambda x: x.tfidf_score,
            reverse=True
        )[:n]
    
    def get_cooccurrences(
        self,
        phrase: str,
        min_weight: float = 0.1
    ) -> List[Tuple[str, float]]:
        """共起する名詞句を取得"""
        if phrase not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(phrase):
            weight = self.graph[phrase][neighbor].get("weight", 0)
            if weight >= min_weight:
                neighbors.append((neighbor, weight))
        
        return sorted(neighbors, key=lambda x: x[1], reverse=True)


class NLPExtractorProtocol(ABC):
    """NLP抽出プロトコル"""
    
    @abstractmethod
    def extract(self, chunk: TextChunk) -> NLPFeatures:
        """単一チャンクからNLP特徴を抽出"""
        ...
    
    @abstractmethod
    def extract_batch(
        self,
        chunks: List[TextChunk]
    ) -> List[NLPFeatures]:
        """複数チャンクから一括抽出"""
        ...
    
    @abstractmethod
    def build_noun_graph(
        self,
        features: List[NLPFeatures]
    ) -> NounGraph:
        """名詞句共起グラフを構築"""
        ...
```

#### 3.3.2 spaCy 実装

```python
class SpacyNLPExtractor(NLPExtractorProtocol):
    """spaCy を使用したNLP抽出"""
    
    def __init__(
        self,
        model: str = "en_core_web_sm",
        languages: List[str] = None
    ):
        import spacy
        
        self.models = {}
        self.default_model = model
        
        # 言語ごとにモデルをロード
        self.models["en"] = spacy.load(model)
        
        if languages and "ja" in languages:
            try:
                self.models["ja"] = spacy.load("ja_core_news_sm")
            except OSError:
                pass  # 日本語モデルがない場合はスキップ
    
    def extract(self, chunk: TextChunk) -> NLPFeatures:
        """NLP特徴を抽出"""
        # 言語検出（簡易）
        lang = self._detect_language(chunk.content)
        nlp = self.models.get(lang, self.models["en"])
        
        doc = nlp(chunk.content)
        
        # 名詞句抽出
        noun_phrases = [
            nc.text for nc in doc.noun_chunks
        ]
        
        # エンティティ抽出
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
        
        # キーワード抽出（名詞）
        keywords = [
            token.lemma_ for token in doc
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop
        ]
        
        return NLPFeatures(
            chunk_id=chunk.id,
            noun_phrases=noun_phrases,
            entities=entities,
            keywords=list(set(keywords)),
            language=lang
        )
    
    def extract_batch(
        self,
        chunks: List[TextChunk]
    ) -> List[NLPFeatures]:
        """バッチ抽出（pipe使用で高速化）"""
        features = []
        
        # 言語ごとにグループ化
        lang_groups = {}
        for chunk in chunks:
            lang = self._detect_language(chunk.content)
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append(chunk)
        
        # 言語ごとにpipe処理
        for lang, lang_chunks in lang_groups.items():
            nlp = self.models.get(lang, self.models["en"])
            texts = [c.content for c in lang_chunks]
            
            for chunk, doc in zip(lang_chunks, nlp.pipe(texts)):
                noun_phrases = [nc.text for nc in doc.noun_chunks]
                entities = [
                    {"text": ent.text, "label": ent.label_}
                    for ent in doc.ents
                ]
                keywords = [
                    token.lemma_ for token in doc
                    if token.pos_ in ("NOUN", "PROPN") and not token.is_stop
                ]
                
                features.append(NLPFeatures(
                    chunk_id=chunk.id,
                    noun_phrases=noun_phrases,
                    entities=entities,
                    keywords=list(set(keywords)),
                    language=lang
                ))
        
        return features
    
    def build_noun_graph(
        self,
        features: List[NLPFeatures]
    ) -> NounGraph:
        """名詞句共起グラフを構築"""
        from collections import defaultdict
        import math
        
        # 名詞句カウント
        phrase_counts = defaultdict(int)
        phrase_chunks = defaultdict(list)
        
        for f in features:
            for phrase in f.noun_phrases:
                phrase_lower = phrase.lower()
                phrase_counts[phrase_lower] += 1
                phrase_chunks[phrase_lower].append(f.chunk_id)
        
        # TF-IDF計算
        n_docs = len(features)
        noun_phrases = []
        
        for phrase, count in phrase_counts.items():
            df = len(set(phrase_chunks[phrase]))
            idf = math.log(n_docs / (df + 1))
            tfidf = count * idf
            
            noun_phrases.append(NounPhrase(
                text=phrase,
                lemma=phrase,
                count=count,
                chunk_ids=phrase_chunks[phrase],
                tfidf_score=tfidf
            ))
        
        # 共起グラフ構築
        graph = nx.Graph()
        
        for f in features:
            phrases = [p.lower() for p in f.noun_phrases]
            # 同一チャンク内の名詞句間にエッジ
            for i, p1 in enumerate(phrases):
                for p2 in phrases[i+1:]:
                    if graph.has_edge(p1, p2):
                        graph[p1][p2]["weight"] += 1
                    else:
                        graph.add_edge(p1, p2, weight=1)
        
        # 重み正規化
        max_weight = max(
            (d["weight"] for _, _, d in graph.edges(data=True)),
            default=1
        )
        for _, _, d in graph.edges(data=True):
            d["weight"] /= max_weight
        
        return NounGraph(graph=graph, noun_phrases=noun_phrases)
    
    def _detect_language(self, text: str) -> str:
        """簡易言語検出"""
        # 日本語文字があれば日本語
        import re
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text):
            return "ja"
        return "en"
```

---

### 3.4 EntityExtractor

LLMベースのエンティティ抽出（GraphRAG Level 2+）。

#### 3.4.1 インターフェース定義

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class AcademicEntityType(Enum):
    """学術エンティティタイプ"""
    RESEARCHER = "researcher"
    ORGANIZATION = "organization"
    METHOD = "method"
    MODEL = "model"
    DATASET = "dataset"
    METRIC = "metric"
    TASK = "task"
    CONCEPT = "concept"
    TOOL = "tool"
    PAPER = "paper"


@dataclass
class Entity:
    """エンティティ"""
    id: str
    name: str
    type: AcademicEntityType
    description: str
    aliases: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    
    # 学術固有
    first_mentioned_year: Optional[int] = None
    external_ids: dict = field(default_factory=dict)  # {"doi": ..., "arxiv": ...}


class EntityExtractorProtocol(ABC):
    """エンティティ抽出プロトコル"""
    
    @abstractmethod
    async def extract(
        self,
        chunk: TextChunk
    ) -> List[Entity]:
        """単一チャンクからエンティティ抽出"""
        ...
    
    @abstractmethod
    async def extract_batch(
        self,
        chunks: List[TextChunk],
        max_concurrent: int = 5
    ) -> List[Entity]:
        """複数チャンクから一括抽出"""
        ...
    
    @abstractmethod
    async def merge_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """重複エンティティをマージ"""
        ...
```

#### 3.4.2 LLM実装

```python
class LLMEntityExtractor(EntityExtractorProtocol):
    """LLMベースのエンティティ抽出"""
    
    EXTRACTION_PROMPT = '''
You are an expert at extracting academic entities from scientific papers.
Extract all entities from the following text and categorize them.

Entity Types:
- RESEARCHER: People (researchers, authors)
- ORGANIZATION: Institutions, companies, labs
- METHOD: Algorithms, techniques, approaches
- MODEL: ML models (GPT-4, BERT, ResNet, etc.)
- DATASET: Datasets (ImageNet, COCO, etc.)
- METRIC: Evaluation metrics (accuracy, F1, BLEU, etc.)
- TASK: Tasks (classification, translation, etc.)
- CONCEPT: Abstract concepts, theories
- TOOL: Tools, frameworks, libraries

Text:
{text}

Output JSON format:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "ENTITY_TYPE",
      "description": "brief description",
      "aliases": ["alias1", "alias2"]
    }}
  ]
}}
'''
    
    def __init__(
        self,
        llm_client: 'ChatModelProtocol',
        max_tokens_per_request: int = 2000
    ):
        self.llm_client = llm_client
        self.max_tokens = max_tokens_per_request
    
    async def extract(self, chunk: TextChunk) -> List[Entity]:
        """エンティティ抽出"""
        import json
        import uuid
        
        prompt = self.EXTRACTION_PROMPT.format(text=chunk.content)
        
        response = await self.llm_client.chat(prompt)
        
        try:
            # JSON抽出
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return []
            
            entities = []
            for item in data.get("entities", []):
                try:
                    entity_type = AcademicEntityType[item["type"]]
                except KeyError:
                    entity_type = AcademicEntityType.CONCEPT
                
                entities.append(Entity(
                    id=str(uuid.uuid4()),
                    name=item["name"],
                    type=entity_type,
                    description=item.get("description", ""),
                    aliases=item.get("aliases", []),
                    source_chunk_ids=[chunk.id]
                ))
            
            return entities
        
        except json.JSONDecodeError:
            return []
    
    async def extract_batch(
        self,
        chunks: List[TextChunk],
        max_concurrent: int = 5
    ) -> List[Entity]:
        """バッチ抽出"""
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_limit(chunk):
            async with semaphore:
                return await self.extract(chunk)
        
        tasks = [extract_with_limit(c) for c in chunks]
        results = await asyncio.gather(*tasks)
        
        # フラット化
        all_entities = []
        for entities in results:
            all_entities.extend(entities)
        
        # マージ
        return await self.merge_entities(all_entities)
    
    async def merge_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """重複エンティティをマージ"""
        from collections import defaultdict
        
        # 名前でグループ化
        groups = defaultdict(list)
        for entity in entities:
            key = entity.name.lower()
            groups[key].append(entity)
        
        merged = []
        for name, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 同じ名前のエンティティをマージ
                primary = group[0]
                for other in group[1:]:
                    primary.aliases.extend(other.aliases)
                    primary.source_chunk_ids.extend(other.source_chunk_ids)
                
                primary.aliases = list(set(primary.aliases))
                primary.source_chunk_ids = list(set(primary.source_chunk_ids))
                merged.append(primary)
        
        return merged
```

---

### 3.5 CommunityDetector

コミュニティ検出（Leiden Algorithm）。

#### 3.5.1 実装

```python
from dataclasses import dataclass
from typing import List, Optional
import networkx as nx


@dataclass
class Community:
    """コミュニティ"""
    id: str
    level: int                    # 階層レベル
    member_ids: List[str]         # メンバーID（名詞句 or エンティティ）
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # サマリー（Level 3で生成）
    title: Optional[str] = None
    summary: Optional[str] = None


class CommunityDetector:
    """コミュニティ検出"""
    
    def __init__(
        self,
        resolution: float = 1.0,
        n_iterations: int = 10,
        seed: int = 42
    ):
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.seed = seed
    
    def detect(
        self,
        graph: nx.Graph,
        hierarchical: bool = True,
        max_levels: int = 3
    ) -> List[Community]:
        """
        Leiden アルゴリズムでコミュニティ検出
        
        Args:
            graph: 入力グラフ（名詞句 or エンティティグラフ）
            hierarchical: 階層的検出
            max_levels: 最大階層数
        
        Returns:
            コミュニティリスト
        """
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            # フォールバック: NetworkX の Louvain
            return self._detect_louvain(graph)
        
        # NetworkX -> iGraph 変換
        ig_graph = ig.Graph.from_networkx(graph)
        
        communities = []
        
        if hierarchical:
            # 階層的検出
            partitions = []
            current_graph = ig_graph
            
            for level in range(max_levels):
                partition = leidenalg.find_partition(
                    current_graph,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=self.resolution,
                    n_iterations=self.n_iterations,
                    seed=self.seed
                )
                
                level_communities = self._partition_to_communities(
                    partition, graph, level
                )
                communities.extend(level_communities)
                
                # 次のレベル用に集約
                if len(partition) <= 1:
                    break
                
                current_graph = partition.cluster_graph()
        else:
            # フラット検出
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution,
                n_iterations=self.n_iterations,
                seed=self.seed
            )
            communities = self._partition_to_communities(partition, graph, 0)
        
        return communities
    
    def _partition_to_communities(
        self,
        partition,
        original_graph: nx.Graph,
        level: int
    ) -> List[Community]:
        """パーティションをCommunityオブジェクトに変換"""
        import uuid
        
        nodes = list(original_graph.nodes())
        communities = []
        
        for i, members in enumerate(partition):
            member_ids = [nodes[m] for m in members]
            communities.append(Community(
                id=f"community_L{level}_{i}",
                level=level,
                member_ids=member_ids
            ))
        
        return communities
    
    def _detect_louvain(self, graph: nx.Graph) -> List[Community]:
        """フォールバック: Louvain法"""
        from networkx.algorithms.community import louvain_communities
        
        partitions = louvain_communities(
            graph,
            resolution=self.resolution,
            seed=self.seed
        )
        
        communities = []
        for i, members in enumerate(partitions):
            communities.append(Community(
                id=f"community_L0_{i}",
                level=0,
                member_ids=list(members)
            ))
        
        return communities
```

---

### 3.6 ProgressiveIndexManager

Progressive GraphRAG のレベル管理。

#### 3.6.1 実装

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import IntEnum
import json


class IndexLevel(IntEnum):
    """インデックスレベル"""
    RAW = 0          # Baseline RAG
    LAZY = 1         # LazyGraphRAG (NLP-based)
    PARTIAL = 2      # GraphRAG with Entities
    FULL = 3         # GraphRAG with Community Reports
    ENHANCED = 4     # Pre-extracted Claims


@dataclass
class LevelStatus:
    """レベルステータス"""
    level: IndexLevel
    is_built: bool
    built_at: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    
    # レベル固有統計
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressiveIndex:
    """Progressive インデックス"""
    id: str
    name: str
    current_level: IndexLevel
    level_status: Dict[IndexLevel, LevelStatus]
    
    # Level 0
    documents: List[AcademicPaper] = field(default_factory=list)
    chunks: List[TextChunk] = field(default_factory=list)
    embeddings: Optional[Dict[str, Any]] = None  # chunk_id -> vector
    
    # Level 1
    noun_graph: Optional[NounGraph] = None
    nlp_communities: List[Community] = field(default_factory=list)
    
    # Level 2
    entities: List[Entity] = field(default_factory=list)
    relationships: List['Relationship'] = field(default_factory=list)
    entity_graph: Optional[nx.Graph] = None
    
    # Level 3
    community_reports: List['CommunityReport'] = field(default_factory=list)
    
    # Level 4
    claims: List['Claim'] = field(default_factory=list)


class ProgressiveIndexManager:
    """Progressive インデックス管理"""
    
    def __init__(
        self,
        storage: 'StorageProtocol',
        chunker: TextChunkerProtocol,
        embedder: EmbedderProtocol,
        nlp_extractor: NLPExtractorProtocol,
        entity_extractor: Optional[EntityExtractorProtocol] = None,
        community_detector: Optional[CommunityDetector] = None,
        llm_client: Optional['ChatModelProtocol'] = None
    ):
        self.storage = storage
        self.chunker = chunker
        self.embedder = embedder
        self.nlp_extractor = nlp_extractor
        self.entity_extractor = entity_extractor
        self.community_detector = community_detector or CommunityDetector()
        self.llm_client = llm_client
    
    async def build_level(
        self,
        index: ProgressiveIndex,
        target_level: IndexLevel,
        papers: Optional[List[AcademicPaper]] = None
    ) -> ProgressiveIndex:
        """
        指定レベルまでインデックスを構築
        
        Args:
            index: 既存インデックス（または新規）
            target_level: 目標レベル
            papers: 新規追加論文（Level 0構築時）
        
        Returns:
            更新されたインデックス
        """
        current = index.current_level
        
        # 下位レベルから順に構築
        for level in range(current + 1, target_level + 1):
            level_enum = IndexLevel(level)
            
            if level_enum == IndexLevel.RAW:
                index = await self._build_level_0(index, papers)
            elif level_enum == IndexLevel.LAZY:
                index = await self._build_level_1(index)
            elif level_enum == IndexLevel.PARTIAL:
                index = await self._build_level_2(index)
            elif level_enum == IndexLevel.FULL:
                index = await self._build_level_3(index)
            elif level_enum == IndexLevel.ENHANCED:
                index = await self._build_level_4(index)
            
            # ステータス更新
            from datetime import datetime
            index.level_status[level_enum] = LevelStatus(
                level=level_enum,
                is_built=True,
                built_at=datetime.utcnow().isoformat(),
                document_count=len(index.documents),
                chunk_count=len(index.chunks)
            )
            index.current_level = level_enum
            
            # 永続化
            await self.storage.save_index(index)
        
        return index
    
    async def _build_level_0(
        self,
        index: ProgressiveIndex,
        papers: List[AcademicPaper]
    ) -> ProgressiveIndex:
        """
        Level 0: Baseline RAG
        - ドキュメント格納
        - チャンク分割
        - エンベディング生成
        """
        index.documents = papers
        
        # チャンク分割
        all_chunks = []
        for paper in papers:
            chunks = self.chunker.chunk_paper(paper)
            all_chunks.extend(chunks)
        
        index.chunks = all_chunks
        
        # エンベディング
        embeddings = await self.embedder.embed_chunks(all_chunks)
        index.embeddings = {
            e.id: e.vector for e in embeddings
        }
        
        return index
    
    async def _build_level_1(
        self,
        index: ProgressiveIndex
    ) -> ProgressiveIndex:
        """
        Level 1: LazyGraphRAG
        - NLP特徴抽出
        - 名詞句グラフ構築
        - NLPベースコミュニティ検出
        """
        # NLP抽出
        features = self.nlp_extractor.extract_batch(index.chunks)
        
        # 名詞句グラフ構築
        index.noun_graph = self.nlp_extractor.build_noun_graph(features)
        
        # コミュニティ検出
        index.nlp_communities = self.community_detector.detect(
            index.noun_graph.graph,
            hierarchical=True
        )
        
        return index
    
    async def _build_level_2(
        self,
        index: ProgressiveIndex
    ) -> ProgressiveIndex:
        """
        Level 2: GraphRAG with Entities
        - LLMエンティティ抽出
        - 関係抽出
        - エンティティグラフ構築
        """
        if not self.entity_extractor:
            raise ValueError("EntityExtractor required for Level 2")
        
        # エンティティ抽出
        index.entities = await self.entity_extractor.extract_batch(
            index.chunks
        )
        
        # 関係抽出（別コンポーネント）
        # index.relationships = await self.relationship_extractor.extract(...)
        
        # エンティティグラフ構築
        index.entity_graph = self._build_entity_graph(
            index.entities,
            index.relationships
        )
        
        return index
    
    async def _build_level_3(
        self,
        index: ProgressiveIndex
    ) -> ProgressiveIndex:
        """
        Level 3: GraphRAG with Community Reports
        - エンティティグラフのコミュニティ検出
        - コミュニティレポート生成
        """
        if not self.llm_client:
            raise ValueError("LLM client required for Level 3")
        
        # エンティティグラフのコミュニティ検出
        entity_communities = self.community_detector.detect(
            index.entity_graph,
            hierarchical=True
        )
        
        # コミュニティレポート生成
        for community in entity_communities:
            report = await self._generate_community_report(
                community,
                index.entities
            )
            index.community_reports.append(report)
        
        return index
    
    async def _build_level_4(
        self,
        index: ProgressiveIndex
    ) -> ProgressiveIndex:
        """
        Level 4: Pre-extracted Claims
        - 全チャンクからクレーム事前抽出
        """
        if not self.llm_client:
            raise ValueError("LLM client required for Level 4")
        
        # クレーム抽出（LazySearch の ClaimExtractor を使用）
        # index.claims = await claim_extractor.extract_all(index.chunks)
        
        return index
```

---

## 4. Queryコンポーネント群

### 4.1 VectorSearch

ベクトル検索コンポーネント（Baseline RAG）。

#### 4.1.1 インターフェース定義

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SearchHit:
    """検索ヒット"""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 学術論文固有
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    section_type: Optional[str] = None


@dataclass
class VectorSearchConfig:
    """ベクトル検索設定"""
    top_k: int = 10
    min_score: float = 0.0
    include_metadata: bool = True
    rerank: bool = False
    rerank_model: Optional[str] = None


class VectorSearchProtocol(ABC):
    """ベクトル検索プロトコル"""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ベクトル検索"""
        ...
    
    @abstractmethod
    async def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ベクトルで直接検索"""
        ...
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ハイブリッド検索（ベクトル + キーワード）"""
        ...
```

#### 4.1.2 LanceDB 実装

```python
class LanceDBVectorSearch(VectorSearchProtocol):
    """LanceDB ベクトル検索（ローカル環境）"""
    
    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedder: EmbedderProtocol,
        config: VectorSearchConfig = None
    ):
        import lancedb
        
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embedder = embedder
        self.config = config or VectorSearchConfig()
        self._table = None
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ベクトル検索"""
        query_vector = await self.embedder.embed(query)
        return await self.search_by_vector(query_vector.tolist(), top_k, filter)
    
    async def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ベクトルで検索"""
        import asyncio
        
        def _search():
            table = self.db.open_table(self.table_name)
            query_builder = table.search(vector).limit(top_k)
            if filter:
                filter_expr = self._build_filter(filter)
                if filter_expr:
                    query_builder = query_builder.where(filter_expr)
            return query_builder.to_list()
        
        results = await asyncio.to_thread(_search)
        
        return [
            SearchHit(
                chunk_id=r["id"],
                score=1.0 - r.get("_distance", 0),
                content=r["content"],
                metadata=r.get("metadata", {}),
                paper_id=r.get("paper_id"),
                paper_title=r.get("paper_title"),
                section_type=r.get("section_type")
            )
            for r in results
        ]
```

#### 4.1.3 Azure AI Search 実装

```python
class AzureAISearchVectorSearch(VectorSearchProtocol):
    """Azure AI Search ベクトル検索（本番環境）"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        embedder: EmbedderProtocol
    ):
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
        self.embedder = embedder
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ベクトル検索"""
        import asyncio
        from azure.search.documents.models import VectorizedQuery
        
        query_vector = await self.embedder.embed(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector.tolist(),
            k_nearest_neighbors=top_k,
            fields="contentVector"
        )
        
        def _search():
            return list(self.client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=self._build_odata_filter(filter) if filter else None,
                top=top_k
            ))
        
        results = await asyncio.to_thread(_search)
        
        return [
            SearchHit(
                chunk_id=r["id"],
                score=r["@search.score"],
                content=r["content"],
                metadata=r.get("metadata", {})
            )
            for r in results
        ]
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """ハイブリッド検索（Semantic Ranker）"""
        import asyncio
        from azure.search.documents.models import (
            VectorizedQuery,
            QueryType
        )
        
        query_vector = await self.embedder.embed(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector.tolist(),
            k_nearest_neighbors=top_k,
            fields="contentVector"
        )
        
        def _hybrid_search():
            return list(self.client.search(
                search_text=query,
                vector_queries=[vector_query],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default",
                top=top_k
            ))
        
        results = await asyncio.to_thread(_hybrid_search)
        
        return [
            SearchHit(
                chunk_id=r["id"],
                score=r.get("@search.reranker_score", r["@search.score"]),
                content=r["content"],
                metadata=r.get("metadata", {})
            )
            for r in results
        ]
```

---

### 4.2 LazySearch

LazyGraphRAG 検索コンポーネント。

#### 4.2.1 データモデル

```python
@dataclass
class LazySearchConfig:
    """LazySearch設定"""
    rate_query: int = 1
    rate_relevancy: int = 32
    rate_cluster: int = 2
    query_expansion_limit: int = 8
    max_context_tokens: int = 8000
    max_iterations: int = 3
    min_relevance_score: float = 0.5


@dataclass
class Claim:
    """クレーム"""
    id: str
    text: str
    confidence: float
    source_chunk_id: str
    source_paper_id: Optional[str] = None


@dataclass
class LazySearchResult:
    """LazySearch結果"""
    query: str
    answer: str
    relevant_chunks: List[SearchHit]
    claims: List[Claim]
    subqueries: List[str]
    iterations: int
    total_tokens_used: int
    search_time_ms: float
```

#### 4.2.2 実装

```python
class LazySearch:
    """LazyGraphRAG 検索"""
    
    def __init__(
        self,
        index: ProgressiveIndex,
        vector_search: VectorSearchProtocol,
        llm_client: 'ChatModelProtocol',
        config: LazySearchConfig = None
    ):
        self.index = index
        self.vector_search = vector_search
        self.llm_client = llm_client
        self.config = config or LazySearchConfig()
        
        self.query_expander = QueryExpander(llm_client)
        self.relevance_tester = RelevanceTester(llm_client)
        self.claim_extractor = ClaimExtractor(llm_client)
    
    async def search(
        self,
        query: str,
        config: Optional[LazySearchConfig] = None
    ) -> LazySearchResult:
        """
        LazySearch 実行
        
        フロー:
        1. クエリ拡張（サブクエリ生成）
        2. ベクトル検索
        3. 関連性テスト
        4. クレーム抽出
        5. 反復深化
        6. 回答生成
        """
        import time
        
        config = config or self.config
        start_time = time.time()
        
        # 1. クエリ拡張
        subqueries = await self.query_expander.expand(
            query, limit=config.query_expansion_limit
        )
        
        # 2. ベクトル検索
        all_hits = []
        for subquery in [query] + subqueries:
            hits = await self.vector_search.search(
                subquery, top_k=config.rate_relevancy
            )
            all_hits.extend(hits)
        
        # 重複除去
        unique_hits = self._deduplicate_hits(all_hits)
        
        # 3. 関連性テスト
        relevant_hits = await self.relevance_tester.test_batch(
            query, unique_hits, min_score=config.min_relevance_score
        )
        
        # 4. クレーム抽出
        claims = await self.claim_extractor.extract_batch(query, relevant_hits)
        
        # 5. コンテキスト構築 & 回答生成
        context = self._build_context(claims, config.max_context_tokens)
        answer = await self._generate_answer(query, context)
        
        search_time = (time.time() - start_time) * 1000
        
        return LazySearchResult(
            query=query,
            answer=answer,
            relevant_chunks=relevant_hits,
            claims=claims,
            subqueries=subqueries,
            iterations=1,
            total_tokens_used=self._count_tokens(context),
            search_time_ms=search_time
        )
    
    def _deduplicate_hits(self, hits: List[SearchHit]) -> List[SearchHit]:
        """重複除去"""
        seen = set()
        unique = []
        for hit in hits:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                unique.append(hit)
        return unique
    
    def _build_context(self, claims: List[Claim], max_tokens: int) -> str:
        """コンテキスト構築"""
        context_parts = []
        total_tokens = 0
        
        for claim in sorted(claims, key=lambda c: c.confidence, reverse=True):
            claim_tokens = len(claim.text.split()) * 1.3  # 概算
            if total_tokens + claim_tokens > max_tokens:
                break
            context_parts.append(f"- {claim.text}")
            total_tokens += claim_tokens
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """回答生成"""
        prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {query}

Answer:"""
        
        return await self.llm_client.chat(prompt)


class QueryExpander:
    """クエリ拡張"""
    
    PROMPT = '''Generate {limit} search subqueries for: {query}
Output as JSON array: ["subquery1", "subquery2", ...]'''
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def expand(self, query: str, limit: int = 8) -> List[str]:
        import json
        import re
        
        prompt = self.PROMPT.format(query=query, limit=limit)
        response = await self.llm_client.chat(prompt)
        
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return []


class RelevanceTester:
    """関連性テスト"""
    
    PROMPT = '''Rate relevance (0-10):
Query: {query}
Text: {text}
Output only the number:'''
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def test_batch(
        self,
        query: str,
        hits: List[SearchHit],
        min_score: float = 0.5
    ) -> List[SearchHit]:
        import asyncio
        
        async def test_one(hit):
            prompt = self.PROMPT.format(query=query, text=hit.content[:500])
            response = await self.llm_client.chat(prompt)
            try:
                score = float(response.strip()) / 10.0
                hit.score = min(max(score, 0.0), 1.0)
            except ValueError:
                hit.score = 0.5
            return hit
        
        tasks = [test_one(h) for h in hits]
        scored = await asyncio.gather(*tasks)
        return [h for h in scored if h.score >= min_score]


class ClaimExtractor:
    """クレーム抽出"""
    
    PROMPT = '''Extract key claims relevant to: {query}
Text: {text}
Output JSON: [{{"claim": "...", "confidence": 0.9}}, ...]'''
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def extract_batch(
        self,
        query: str,
        hits: List[SearchHit]
    ) -> List[Claim]:
        import asyncio
        import json
        import re
        import uuid
        
        async def extract_one(hit):
            prompt = self.PROMPT.format(query=query, text=hit.content)
            response = await self.llm_client.chat(prompt)
            claims = []
            try:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    data = json.loads(json_match.group())
                    for item in data:
                        claims.append(Claim(
                            id=str(uuid.uuid4()),
                            text=item["claim"],
                            confidence=item.get("confidence", 0.8),
                            source_chunk_id=hit.chunk_id
                        ))
            except json.JSONDecodeError:
                pass
            return claims
        
        tasks = [extract_one(h) for h in hits]
        results = await asyncio.gather(*tasks)
        return [c for claims in results for c in claims]
```

---

### 4.3 QueryRouter

クエリルーティング（Unified GraphRAG）。

#### 4.3.1 データモデル

```python
from enum import Enum


class SearchMode(Enum):
    """検索モード"""
    AUTO = "auto"
    VECTOR = "vector"
    LAZY = "lazy"
    GRAPHRAG = "graphrag"
    HYBRID = "hybrid"


class QueryType(Enum):
    """クエリタイプ"""
    SURVEY = "survey"
    EXPLORATION = "exploration"
    COMPARISON = "comparison"
    FACTOID = "factoid"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """ルーティング決定"""
    mode: SearchMode
    query_type: QueryType
    confidence: float
    reasoning: str
    params: Dict[str, Any] = field(default_factory=dict)
```

#### 4.3.2 実装

```python
class QueryRouter:
    """クエリルーター"""
    
    SURVEY_PATTERNS = [
        r'(?i)(研究動向|トレンド|trend|survey|overview|概要)',
        r'(?i)(全体的|主要な|main\s+themes?)',
    ]
    
    COMPARISON_PATTERNS = [
        r'(?i)(比較|compare|versus|vs\.?|違い|difference)',
        r'(?i)(優れ|better|worse|advantage)',
    ]
    
    FACTOID_PATTERNS = [
        r'(?i)(いくつ|何個|how\s+many|数値|value)',
        r'(?i)(どこに|where|何ページ)',
    ]
    
    def __init__(
        self,
        llm_client: Optional['ChatModelProtocol'] = None,
        use_llm_classification: bool = False
    ):
        self.llm_client = llm_client
        self.use_llm = use_llm_classification and llm_client is not None
    
    async def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """クエリをルーティング"""
        import re
        
        # ルールベース判定
        query_type, confidence = self._rule_based_classify(query)
        
        # LLM分類（オプション）
        if self.use_llm and confidence < 0.7:
            query_type, confidence = await self._llm_classify(query)
        
        # モード決定
        mode = self._determine_mode(query_type, context)
        
        return RoutingDecision(
            mode=mode,
            query_type=query_type,
            confidence=confidence,
            reasoning=f"{query_type.value} query -> {mode.value} search",
            params=self._determine_params(mode, query_type)
        )
    
    def _rule_based_classify(self, query: str) -> tuple[QueryType, float]:
        import re
        
        for pattern in self.SURVEY_PATTERNS:
            if re.search(pattern, query):
                return QueryType.SURVEY, 0.8
        
        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query):
                return QueryType.COMPARISON, 0.8
        
        for pattern in self.FACTOID_PATTERNS:
            if re.search(pattern, query):
                return QueryType.FACTOID, 0.8
        
        return QueryType.EXPLORATION, 0.6
    
    def _determine_mode(
        self,
        query_type: QueryType,
        context: Optional[Dict[str, Any]]
    ) -> SearchMode:
        index_level = context.get("index_level", 1) if context else 1
        
        mode_mapping = {
            QueryType.SURVEY: SearchMode.GRAPHRAG if index_level >= 3 else SearchMode.LAZY,
            QueryType.EXPLORATION: SearchMode.LAZY,
            QueryType.COMPARISON: SearchMode.HYBRID,
            QueryType.FACTOID: SearchMode.VECTOR,
            QueryType.UNKNOWN: SearchMode.LAZY,
        }
        
        return mode_mapping.get(query_type, SearchMode.LAZY)
    
    def _determine_params(
        self,
        mode: SearchMode,
        query_type: QueryType
    ) -> Dict[str, Any]:
        params = {}
        
        if mode == SearchMode.HYBRID:
            params["methods"] = ["vector", "lazy"]
            params["fusion"] = "rrf"
        
        if mode == SearchMode.GRAPHRAG:
            params["search_type"] = "global" if query_type == QueryType.SURVEY else "local"
        
        return params
```

---

### 4.4 HybridSearch

ハイブリッド検索（RRF融合）。

```python
@dataclass
class HybridSearchConfig:
    """ハイブリッド検索設定"""
    methods: List[str] = field(default_factory=lambda: ["vector", "lazy"])
    fusion: str = "rrf"
    rrf_k: int = 60


class HybridSearch:
    """ハイブリッド検索"""
    
    def __init__(
        self,
        vector_search: VectorSearchProtocol,
        lazy_search: LazySearch,
        config: HybridSearchConfig = None
    ):
        self.vector_search = vector_search
        self.lazy_search = lazy_search
        self.config = config or HybridSearchConfig()
    
    async def search(
        self,
        query: str,
        methods: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[SearchHit]:
        """ハイブリッド検索"""
        methods = methods or self.config.methods
        
        results_by_method = {}
        
        if "vector" in methods:
            results_by_method["vector"] = await self.vector_search.search(query)
        
        if "lazy" in methods:
            lazy_result = await self.lazy_search.search(query)
            results_by_method["lazy"] = lazy_result.relevant_chunks
        
        return self._rrf_fusion(results_by_method, top_k)
    
    def _rrf_fusion(
        self,
        results: Dict[str, List[SearchHit]],
        top_k: int
    ) -> List[SearchHit]:
        """Reciprocal Rank Fusion"""
        k = self.config.rrf_k
        rrf_scores = {}
        hit_map = {}
        
        for method, hits in results.items():
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit.chunk_id
                
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    hit_map[chunk_id] = hit
                
                rrf_scores[chunk_id] += 1.0 / (k + rank)
        
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        result = []
        for chunk_id in sorted_ids[:top_k]:
            hit = hit_map[chunk_id]
            hit.score = rrf_scores[chunk_id]
            result.append(hit)
        
        return result
```

---

## 5. 引用ネットワークコンポーネント

### 5.1 CitationNetworkBuilder

引用ネットワーク構築。

```python
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class CitationEdge:
    """引用エッジ"""
    citing_paper_id: str
    cited_paper_id: str
    context: Optional[str] = None  # 引用コンテキスト
    section: Optional[str] = None  # 引用セクション


class CitationNetworkBuilder:
    """引用ネットワーク構築"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build(self, papers: List[AcademicPaper]) -> nx.DiGraph:
        """引用ネットワークを構築"""
        
        # ノード追加（論文）
        for paper in papers:
            self.graph.add_node(
                paper.id,
                title=paper.title,
                year=paper.publication_date.year if paper.publication_date else None,
                doi=paper.doi
            )
        
        # エッジ追加（引用関係）
        paper_ids = {p.id for p in papers}
        
        for paper in papers:
            for ref in paper.references:
                # 参照先がインデックス内にあるか確認
                cited_id = self._resolve_reference(ref, papers)
                if cited_id and cited_id in paper_ids:
                    self.graph.add_edge(
                        paper.id,
                        cited_id,
                        ref_id=ref.ref_id
                    )
        
        return self.graph
    
    def _resolve_reference(
        self,
        ref: Reference,
        papers: List[AcademicPaper]
    ) -> Optional[str]:
        """参照を解決"""
        # DOIマッチ
        if ref.doi:
            for paper in papers:
                if paper.doi == ref.doi:
                    return paper.id
        
        # arXiv IDマッチ
        if ref.arxiv_id:
            for paper in papers:
                if paper.arxiv_id == ref.arxiv_id:
                    return paper.id
        
        # タイトル類似度マッチ（フォールバック）
        if ref.title:
            for paper in papers:
                if self._similar_title(ref.title, paper.title):
                    return paper.id
        
        return None
    
    def _similar_title(self, title1: str, title2: str) -> bool:
        """タイトル類似度判定"""
        # 簡易実装: 正規化して比較
        def normalize(s):
            return ''.join(c.lower() for c in s if c.isalnum())
        
        return normalize(title1) == normalize(title2)
    
    def get_citations(self, paper_id: str) -> List[str]:
        """被引用論文を取得"""
        return list(self.graph.predecessors(paper_id))
    
    def get_references(self, paper_id: str) -> List[str]:
        """参照論文を取得"""
        return list(self.graph.successors(paper_id))


class CoCitationAnalyzer:
    """共引用分析"""
    
    def __init__(self, citation_graph: nx.DiGraph):
        self.graph = citation_graph
    
    def get_co_citations(
        self,
        paper_id: str,
        min_count: int = 2
    ) -> List[Tuple[str, int]]:
        """
        共引用論文を取得
        （同じ論文に引用される他の論文）
        """
        # この論文を引用している論文
        citing_papers = list(self.graph.predecessors(paper_id))
        
        # 共引用カウント
        co_citation_count = {}
        
        for citing in citing_papers:
            # citing が引用している他の論文
            for other in self.graph.successors(citing):
                if other != paper_id:
                    co_citation_count[other] = co_citation_count.get(other, 0) + 1
        
        # フィルタ & ソート
        results = [
            (paper, count) 
            for paper, count in co_citation_count.items()
            if count >= min_count
        ]
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_bibliographic_coupling(
        self,
        paper_id: str,
        min_count: int = 2
    ) -> List[Tuple[str, int]]:
        """
        書誌結合論文を取得
        （同じ論文を引用する他の論文）
        """
        # この論文が引用している論文
        referenced_papers = list(self.graph.successors(paper_id))
        
        # 書誌結合カウント
        coupling_count = {}
        
        for ref in referenced_papers:
            # ref を引用している他の論文
            for other in self.graph.predecessors(ref):
                if other != paper_id:
                    coupling_count[other] = coupling_count.get(other, 0) + 1
        
        results = [
            (paper, count)
            for paper, count in coupling_count.items()
            if count >= min_count
        ]
        
        return sorted(results, key=lambda x: x[1], reverse=True)
```

---

## 6. Storageコンポーネント

### 6.1 FileStorage

Parquetベースのファイルストレージ。

```python
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class StorageProtocol(ABC):
    """ストレージプロトコル"""
    
    @abstractmethod
    async def save_index(self, index: ProgressiveIndex) -> None:
        """インデックスを保存"""
        ...
    
    @abstractmethod
    async def load_index(self, index_id: str) -> ProgressiveIndex:
        """インデックスを読み込み"""
        ...
    
    @abstractmethod
    async def save_chunks(self, chunks: List[TextChunk]) -> None:
        """チャンクを保存"""
        ...
    
    @abstractmethod
    async def load_chunks(self, index_id: str) -> List[TextChunk]:
        """チャンクを読み込み"""
        ...


class FileStorage(StorageProtocol):
    """ファイルベースストレージ"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_index(self, index: ProgressiveIndex) -> None:
        """インデックスを保存"""
        import json
        import asyncio
        
        index_dir = self.base_path / index.id
        index_dir.mkdir(exist_ok=True)
        
        # メタデータ
        metadata = {
            "id": index.id,
            "name": index.name,
            "current_level": index.current_level.value,
            "level_status": {
                str(k.value): {
                    "is_built": v.is_built,
                    "built_at": v.built_at,
                    "document_count": v.document_count,
                    "chunk_count": v.chunk_count,
                }
                for k, v in index.level_status.items()
            }
        }
        
        def _save():
            with open(index_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        await asyncio.to_thread(_save)
        
        # チャンク
        if index.chunks:
            await self.save_chunks(index.chunks, index.id)
    
    async def save_chunks(
        self,
        chunks: List[TextChunk],
        index_id: str = "default"
    ) -> None:
        """チャンクをParquetで保存"""
        import asyncio
        from dataclasses import asdict
        
        index_dir = self.base_path / index_id
        index_dir.mkdir(exist_ok=True)
        
        def _save():
            df = pd.DataFrame([asdict(c) for c in chunks])
            df.to_parquet(index_dir / "chunks.parquet")
        
        await asyncio.to_thread(_save)
    
    async def load_chunks(self, index_id: str) -> List[TextChunk]:
        """チャンクを読み込み"""
        import asyncio
        
        chunks_path = self.base_path / index_id / "chunks.parquet"
        
        def _load():
            if not chunks_path.exists():
                return []
            df = pd.read_parquet(chunks_path)
            return [TextChunk(**row) for row in df.to_dict('records')]
        
        return await asyncio.to_thread(_load)
```

### 6.2 CacheManager

キャッシュ管理（Redis / ローカル）。

```python
from abc import ABC, abstractmethod
from typing import Optional, Any
import json


class CacheProtocol(ABC):
    """キャッシュプロトコル"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        ...
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """キャッシュ設定"""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """キャッシュ削除"""
        ...


class LocalCache(CacheProtocol):
    """ローカルメモリキャッシュ"""
    
    def __init__(self, max_size: int = 1000):
        from collections import OrderedDict
        self._cache = OrderedDict()
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value
    
    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)


class RedisCache(CacheProtocol):
    """Redis キャッシュ（本番環境）"""
    
    def __init__(
        self,
        host: str,
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0
    ):
        import redis.asyncio as redis
        
        self.client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
    
    async def get(self, key: str) -> Optional[Any]:
        value = await self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        serialized = json.dumps(value)
        if ttl:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)
    
    async def delete(self, key: str) -> None:
        await self.client.delete(key)
```

---

## 7. LLMクライアント

### 7.1 ChatModelProtocol

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict


class ChatModelProtocol(ABC):
    """チャットモデルプロトコル"""
    
    @abstractmethod
    async def chat(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """チャット応答生成"""
        ...
    
    @abstractmethod
    async def chat_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """ストリーミング応答"""
        ...
```

### 7.2 AzureOpenAIClient

```python
class AzureOpenAIClient(ChatModelProtocol):
    """Azure OpenAI クライアント"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-02-15-preview"
    ):
        from openai import AsyncAzureOpenAI
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment = deployment
    
    async def chat(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        messages = history or []
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        
        return response.choices[0].message.content
    
    async def chat_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        messages = history or []
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### 7.3 OllamaClient

```python
class OllamaClient(ChatModelProtocol):
    """Ollama クライアント（ローカル環境）"""
    
    def __init__(
        self,
        host: str = "http://192.168.224.1:11434",
        model: str = "llama3.2"
    ):
        import httpx
        
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)
    
    async def chat(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        messages = history or []
        messages.append({"role": "user", "content": prompt})
        
        response = await self._client.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0)
                }
            }
        )
        response.raise_for_status()
        
        return response.json()["message"]["content"]
    
    async def chat_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        messages = history or []
        messages.append({"role": "user", "content": prompt})
        
        async with self._client.stream(
            "POST",
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
```

### 7.4 LLMFactory

```python
class LLMFactory:
    """LLMファクトリー"""
    
    @staticmethod
    def create(provider: str, config: dict) -> ChatModelProtocol:
        if provider == "azure_openai":
            return AzureOpenAIClient(
                endpoint=config["endpoint"],
                api_key=config["api_key"],
                deployment=config["deployment"]
            )
        elif provider == "ollama":
            return OllamaClient(
                host=config.get("host", "http://192.168.224.1:11434"),
                model=config.get("model", "llama3.2")
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    @staticmethod
    def create_from_environment() -> ChatModelProtocol:
        import os
        
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            return LLMFactory.create("azure_openai", {
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            })
        else:
            return LLMFactory.create("ollama", {
                "host": os.getenv("OLLAMA_HOST", "http://192.168.224.1:11434"),
                "model": os.getenv("OLLAMA_MODEL", "llama3.2")
            })
```

---

## 8. MCP Server

### 8.1 MCPServer

MCP (Model Context Protocol) サーバー実装。

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class MCPTool:
    """MCPツール定義"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPServer:
    """MCP Server"""
    
    def __init__(
        self,
        facade: 'MONJYUFacade',
        transport: str = "stdio"
    ):
        self.facade = facade
        self.transport = transport
        
        self.tools = [
            MCPTool(
                name="monjyu_search",
                description="Search academic papers using query",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "mode": {"type": "string", "enum": ["auto", "vector", "lazy", "hybrid"]},
                        "top_k": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="monjyu_index",
                description="Index academic papers",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paths": {"type": "array", "items": {"type": "string"}},
                        "level": {"type": "integer", "default": 1}
                    },
                    "required": ["paths"]
                }
            ),
            MCPTool(
                name="monjyu_get_paper",
                description="Get paper details by ID",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"}
                    },
                    "required": ["paper_id"]
                }
            ),
            MCPTool(
                name="monjyu_citations",
                description="Get citation network for a paper",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "depth": {"type": "integer", "default": 1}
                    },
                    "required": ["paper_id"]
                }
            ),
            MCPTool(
                name="monjyu_summarize",
                description="Summarize a paper or section",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "section": {"type": "string"}
                    },
                    "required": ["paper_id"]
                }
            ),
            MCPTool(
                name="monjyu_compare",
                description="Compare multiple papers or methods",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paper_ids": {"type": "array", "items": {"type": "string"}},
                        "aspect": {"type": "string"}
                    },
                    "required": ["paper_ids"]
                }
            ),
            MCPTool(
                name="monjyu_survey",
                description="Generate a survey on a topic",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "max_papers": {"type": "integer", "default": 10}
                    },
                    "required": ["topic"]
                }
            )
        ]
    
    async def handle_request(self, request: dict) -> dict:
        """リクエスト処理"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema
                    }
                    for t in self.tools
                ]
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            return await self._call_tool(tool_name, arguments)
        
        return {"error": f"Unknown method: {method}"}
    
    async def _call_tool(
        self,
        name: str,
        arguments: dict
    ) -> dict:
        """ツール呼び出し"""
        try:
            if name == "monjyu_search":
                result = await self.facade.search(
                    query=arguments["query"],
                    mode=arguments.get("mode", "auto"),
                    top_k=arguments.get("top_k", 10)
                )
                return {"content": [{"type": "text", "text": result.answer}]}
            
            elif name == "monjyu_index":
                result = await self.facade.index(
                    paths=arguments["paths"],
                    level=arguments.get("level", 1)
                )
                return {"content": [{"type": "text", "text": f"Indexed {result.document_count} papers"}]}
            
            elif name == "monjyu_get_paper":
                paper = await self.facade.get_paper(arguments["paper_id"])
                return {"content": [{"type": "text", "text": json.dumps(paper, default=str)}]}
            
            elif name == "monjyu_citations":
                citations = await self.facade.get_citations(
                    arguments["paper_id"],
                    depth=arguments.get("depth", 1)
                )
                return {"content": [{"type": "text", "text": json.dumps(citations)}]}
            
            elif name == "monjyu_summarize":
                summary = await self.facade.summarize(
                    arguments["paper_id"],
                    section=arguments.get("section")
                )
                return {"content": [{"type": "text", "text": summary}]}
            
            elif name == "monjyu_compare":
                comparison = await self.facade.compare(
                    arguments["paper_ids"],
                    aspect=arguments.get("aspect")
                )
                return {"content": [{"type": "text", "text": comparison}]}
            
            elif name == "monjyu_survey":
                survey = await self.facade.survey(
                    arguments["topic"],
                    max_papers=arguments.get("max_papers", 10)
                )
                return {"content": [{"type": "text", "text": survey}]}
            
            else:
                return {"error": f"Unknown tool: {name}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    async def run_stdio(self):
        """stdio トランスポートで実行"""
        import sys
        import asyncio
        
        while True:
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line)
                response = await self.handle_request(request)
                
                print(json.dumps(response), flush=True)
            
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
```

---

## 9. CLI

### 9.1 CLI実装

```python
import typer
from pathlib import Path
from typing import Optional, List
from enum import Enum

app = typer.Typer(
    name="monjyu",
    help="MONJYU - Academic Paper RAG System"
)


class SearchModeOption(str, Enum):
    auto = "auto"
    vector = "vector"
    lazy = "lazy"
    hybrid = "hybrid"


@app.command()
def index(
    input_path: Path = typer.Argument(..., help="Input directory or file"),
    output_path: Path = typer.Option("./output", "--output", "-o"),
    level: int = typer.Option(1, "--level", "-l", help="Index level (0-4)"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Build index from academic papers."""
    import asyncio
    
    async def _index():
        facade = create_facade()
        result = await facade.index(
            paths=[str(input_path)],
            output_path=str(output_path),
            level=level,
            recursive=recursive
        )
        
        typer.echo(f"✅ Indexed {result.document_count} papers")
        typer.echo(f"   Chunks: {result.chunk_count}")
        typer.echo(f"   Level: {result.level}")
        typer.echo(f"   Output: {output_path}")
    
    asyncio.run(_index())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    index_path: Path = typer.Option("./output", "--index", "-i"),
    mode: SearchModeOption = typer.Option(SearchModeOption.auto, "--mode", "-m"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    stream: bool = typer.Option(False, "--stream", "-s")
):
    """Search academic papers."""
    import asyncio
    
    async def _search():
        facade = create_facade(index_path=str(index_path))
        
        if stream:
            async for chunk in facade.search_stream(query, mode=mode.value):
                typer.echo(chunk, nl=False)
            typer.echo()
        else:
            result = await facade.search(query, mode=mode.value, top_k=top_k)
            
            typer.echo(f"\n📝 Answer:\n{result.answer}\n")
            typer.echo(f"📚 Sources ({len(result.relevant_chunks)} chunks):")
            for hit in result.relevant_chunks[:5]:
                typer.echo(f"  - [{hit.score:.2f}] {hit.paper_title or hit.chunk_id}")
    
    asyncio.run(_search())


@app.command()
def upgrade(
    index_path: Path = typer.Option("./output", "--index", "-i"),
    target_level: int = typer.Argument(..., help="Target level (1-4)")
):
    """Upgrade index to higher level."""
    import asyncio
    
    async def _upgrade():
        facade = create_facade(index_path=str(index_path))
        result = await facade.upgrade_index(target_level)
        
        typer.echo(f"✅ Upgraded to Level {result.level}")
    
    asyncio.run(_upgrade())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    transport: str = typer.Option("stdio", "--transport", "-t")
):
    """Start MCP server."""
    import asyncio
    
    async def _serve():
        facade = create_facade()
        server = MCPServer(facade, transport=transport)
        
        if transport == "stdio":
            typer.echo("Starting MCP server (stdio)...", err=True)
            await server.run_stdio()
        else:
            typer.echo(f"Starting MCP server at {host}:{port}...", err=True)
            # HTTP/SSE transport implementation
    
    asyncio.run(_serve())


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current config"),
    init: bool = typer.Option(False, "--init", help="Initialize config")
):
    """Manage configuration."""
    if show:
        # Show current config
        pass
    elif init:
        # Initialize config file
        pass


def create_facade(index_path: str = "./output") -> 'MONJYUFacade':
    """Facade生成"""
    # 環境に応じたコンポーネント生成
    from monjyu.facade import MONJYUFacade
    return MONJYUFacade(index_path=index_path)


if __name__ == "__main__":
    app()
```

---

## 10. 設定・エラーハンドリング

### 10.1 環境別設定

```yaml
# config/local.yaml
environment: local

llm:
  provider: ollama
  host: "http://192.168.224.1:11434"
  model: llama3.2
  embedding_model: nomic-embed-text

storage:
  type: file
  base_path: ./output
  vector_store: lancedb

pdf_processor:
  type: unstructured
  strategy: hi_res
  languages: [eng, jpn]

index:
  chunk_size: 300
  chunk_overlap: 100
  default_level: 1

query:
  default_mode: auto
  max_context_tokens: 8000
```

```yaml
# config/production.yaml
environment: production

llm:
  provider: azure_openai
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  deployment: gpt-4o
  embedding_deployment: text-embedding-3-large

storage:
  type: azure
  blob_connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  vector_store: azure_ai_search
  search_endpoint: ${AZURE_SEARCH_ENDPOINT}
  search_api_key: ${AZURE_SEARCH_API_KEY}
  
  cache:
    type: redis
    host: ${REDIS_HOST}
    password: ${REDIS_PASSWORD}

pdf_processor:
  type: azure_document_intelligence
  endpoint: ${AZURE_DI_ENDPOINT}
  api_key: ${AZURE_DI_KEY}

index:
  chunk_size: 300
  chunk_overlap: 100
  default_level: 1

query:
  default_mode: auto
  max_context_tokens: 8000
```

### 10.2 例外階層

```python
class MONJYUError(Exception):
    """MONJYU基底例外"""
    pass


class ConfigurationError(MONJYUError):
    """設定エラー"""
    pass


class IndexError(MONJYUError):
    """インデックスエラー"""
    pass


class StorageError(MONJYUError):
    """ストレージエラー"""
    pass


class LLMError(MONJYUError):
    """LLMエラー"""
    pass


class QueryError(MONJYUError):
    """クエリエラー"""
    pass


class PDFProcessError(MONJYUError):
    """PDF処理エラー"""
    pass


class ExternalAPIError(MONJYUError):
    """外部APIエラー"""
    pass
```

---

## 11. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-01-06 | 初版（LazyGraphRAGベース） |
| 3.0.0 | 2025-12-24 | v3.0 要件対応、学術論文特化、MCP Server追加 |
| 3.1.0 | 2025-12-27 | 実装ステータス追加 (28/32 = 87.5%)、アーキテクチャv3.1準拠 |
| 3.2.0 | 2026-01-07 | 全コンポーネント実装完了 (32/32 = 100%)、要件v3.1/アーキテクチャv3.2準拠、テスト状況更新（2,417テスト/83%カバレッジ） |

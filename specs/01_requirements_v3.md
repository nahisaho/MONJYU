# MONJYU 要件定義書 v3.1

**文書番号**: MONJYU-REQ-003  
**バージョン**: 3.1.0  
**作成日**: 2025-12-24  
**最終更新日**: 2026-01-07  
**ステータス**: ✅ Approved (実装完了)  
**準拠**: MUSUBI SDD Constitutional Article IV (EARS Format)

**実装状況**: ✅ 2,417 tests passed | 83% coverage

---

## 1. 概要

### 1.1 プロジェクトビジョン

**MONJYU** (文殊) - 「三人寄れば文殊の知恵」

LazyGraphRAGをベースとした次世代RAGシステム。**Unified GraphRAG** と **Progressive GraphRAG** という2つの新しいアーキテクチャパターンを実装し、コスト効率と品質の最適なバランスを実現する。

**ターゲットドメイン: 学術論文 (AI for Science)**

本システムは学術論文の検索・分析を主要ターゲットとする。arXiv、PubMed、IEEE Xplore等の学術論文データベースから取得した論文PDFを処理し、研究者の文献調査・先行研究分析・手法比較を支援する。

**学術論文特有の課題**:
- 複雑なレイアウト（2カラム、図表、数式）
- 引用ネットワークの構造化
- 専門用語・略語の理解
- 多言語（英語・日本語・中国語等）対応
- 継続的な新規論文の追加

### 1.2 対象アーキテクチャパターン

本要件定義は以下の6つのアーキテクチャパターンをサポートする：

| アーキテクチャ | 説明 | 実装優先度 |
|---------------|------|-----------|
| **Baseline RAG** | チャンク検索 + 生成 | P0 (必須) |
| **GraphRAG** | グラフ構築 + 検索 | P1 (高) |
| **LazyGraphRAG** | 遅延グラフ + 動的抽出 | P0 (必須) |
| **Hybrid GraphRAG** | 複数エンジン並列実行 + マージ | P1 (高) |
| **Unified GraphRAG** | Query Router による動的選択 | P0 (必須) 💡 |
| **Progressive GraphRAG** | 段階的インデックス + 予算制御 | P0 (必須) 💡 |

> 💡 **Unified GraphRAG** と **Progressive GraphRAG** は本プロジェクトで提案する新規アーキテクチャ

### 1.3 システム範囲

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MONJYU System                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   Index Layer   │  │   Query Layer   │  │   Architecture      │ │
│  │                 │  │                 │  │   Controller        │ │
│  │  - Document     │  │  - Global Search│  │                     │ │
│  │  - TextUnit     │  │  - Local Search │  │  - Unified Router   │ │
│  │  - Entity       │  │  - Lazy Search  │  │  - Progressive      │ │
│  │  - Relationship │  │  - Vector Search│  │    Budget Manager   │ │
│  │  - Community    │  │  - Hybrid Merge │  │  - Hybrid Merger    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Storage Layer  │  │   API/CLI       │  │   LLM/Embedding     │ │
│  │                 │  │                 │  │   Providers         │ │
│  │  - Parquet      │  │  - REST API     │  │                     │ │
│  │  - Vector DB    │  │  - CLI          │  │  - OpenAI/Azure     │ │
│  │  - Graph DB     │  │  - Streaming    │  │  - Local Models     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 機能要件（EARS形式）

### 2.1 インデックス機能 (Index Domain)

#### REQ-IDX-001: ドキュメントローダー

**EARS Pattern**: Ubiquitous

> The system SHALL load documents from specified input directories supporting txt, md, pdf, html, csv, json, docx, xlsx formats.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-001 |
| 優先度 | P0 (必須) |
| 入力 | ディレクトリパス、ファイルパターン |
| 出力 | Document[] (id, title, text, metadata) |
| 受入基準 | 100MB以上のドキュメントセットを処理可能 |

**対応ファイルフォーマット**:

| フォーマット | 処理方式 | ライブラリ |
|-------------|---------|-----------|
| `.txt`, `.json` | テキスト抽出 | unstructured.partition.text |
| `.md` | Markdown解析 | unstructured.partition.md |
| `.html`, `.htm` | HTML解析 | unstructured.partition.html |
| `.csv` | CSV解析 | unstructured.partition.csv |
| `.xml` | XML解析 | unstructured.partition.xml |
| `.docx` | Word解析 | unstructured.partition.docx |
| `.doc` | Word (旧形式) | unstructured.partition.doc |
| `.pptx` | PowerPoint解析 | unstructured.partition.pptx |
| `.xlsx` | Excel解析 | unstructured.partition.xlsx |
| `.eml`, `.msg` | メール解析 | unstructured.partition.email/msg |
| `.pdf` | PDF解析 | Azure Document Intelligence / unstructured |

---

#### REQ-IDX-001a: PDF前処理（学術論文対応）

**EARS Pattern**: Ubiquitous

> The system SHALL process PDF documents including academic papers with complex layouts, tables, figures, and mathematical expressions.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-001a |
| 優先度 | P0 (必須) |
| 入力 | PDF files |
| 出力 | ProcessedDocument (text, tables, figures, metadata) |
| 受入基準 | 2カラム論文、図表、数式を正確に抽出 |

**PDF処理戦略**:

| 方式 | 説明 | 用途 |
|------|------|------|
| **Azure Document Intelligence** | Azure AI による高精度レイアウト解析 | エンタープライズ、複雑な論文 |
| **unstructured** | ローカル解析（PyMuPDF/pdfminer） | コスト重視、シンプルなPDF |
| **OCR** | 画像ベースPDFの文字認識 | スキャン文書 |

**Azure Document Intelligence 設定**:
```yaml
pdf_processing:
  provider: azure_document_intelligence  # azure_document_intelligence / unstructured
  model: prebuilt-layout                 # prebuilt-layout / prebuilt-document
  api_version: "2024-02-29-preview"
  features:
    - tables                             # テーブル抽出
    - figures                            # 図表抽出
    - key_value_pairs                    # キーバリューペア
    - formulas                           # 数式抽出（学術論文向け）
```

**学術論文特有の処理**:

| 要素 | 処理方法 | 出力 |
|------|---------|------|
| **タイトル・著者** | メタデータ抽出 | title, authors[], affiliations[] |
| **アブストラクト** | セクション識別 | abstract (text) |
| **本文** | 2カラム→1カラム変換 | body_text (text) |
| **図表** | キャプション付き抽出 | figures[], tables[] with captions |
| **数式** | LaTeX/MathML変換 | equations[] |
| **参考文献** | 構造化抽出 | references[] (title, authors, year, doi) |
| **引用関係** | インライン引用検出 | citations[] (position, ref_id) |

**IMRaD構造認識** (Introduction, Methods, Results, and Discussion):

| セクション | 識別パターン | 用途 |
|-----------|-------------|------|
| **Introduction** | 背景・目的・貢献 | 研究動機の理解 |
| **Related Work** | 先行研究・比較 | 文献調査支援 |
| **Methods** | 手法・アルゴリズム | 手法比較・再現 |
| **Experiments/Results** | 実験設定・結果 | ベンチマーク調査 |
| **Discussion** | 考察・限界 | 批判的分析 |
| **Conclusion** | 結論・今後の課題 | サマリー生成 |

**学術メタデータ抽出**:

| メタデータ | ソース | 形式 |
|-----------|--------|------|
| **DOI** | 論文PDF/CrossRef API | 10.xxxx/xxxxx |
| **arXiv ID** | arXiv URL/PDF | arXiv:YYMM.NNNNN |
| **出版年** | 論文ヘッダー | YYYY |
| **ジャーナル/会議** | 論文ヘッダー | 文字列 |
| **キーワード** | 論文内/API | リスト |
| **引用数** | Semantic Scholar API | 整数 |
| **被引用論文** | Semantic Scholar API | DOIリスト |

**数式・化学式処理**:

| 種別 | 入力形式 | 出力形式 | 用途 |
|------|---------|---------|------|
| **数式** | 画像/LaTeX | LaTeX/テキスト記述 | 数学・物理論文 |
| **化学式** | ChemDraw/画像 | SMILES/InChI | 化学・生物論文 |
| **反応式** | 画像 | RXNSMILES | 合成化学論文 |

---

#### REQ-IDX-001b: Word/PowerPoint前処理

**EARS Pattern**: Ubiquitous

> The system SHALL process Word and PowerPoint documents preserving structure, tables, and embedded content.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-001b |
| 優先度 | P1 (高) |
| 入力 | .docx, .doc, .pptx, .ppt files |
| 出力 | ProcessedDocument (text, tables, metadata) |
| 受入基準 | 見出し構造、テーブル、画像テキストを保持 |

**Word処理機能**:
- 見出し階層（Heading 1-6）の構造化
- テーブルのHTML形式変換（構造保持）
- 画像のOCR処理（オプション）
- コメント・変更履歴の除外（設定可能）

**PowerPoint処理機能**:
- スライド単位のチャンキング
- ノート（発表者メモ）の抽出
- 図表・グラフのテキスト抽出

---

#### REQ-IDX-001c: 前処理パイプライン

**EARS Pattern**: Ubiquitous

> The system SHALL provide a configurable preprocessing pipeline that processes documents through multiple stages.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-001c |
| 優先度 | P0 (必須) |
| 入力 | Raw files |
| 出力 | ProcessedDocument[] |
| 受入基準 | パイプライン各ステージの設定可能 |

**パイプライン構成**:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Document Preprocessing Pipeline                │
├──────────────────────────────────────────────────────────────────┤
│  1. ファイル検出     │ 拡張子判定、MIME タイプ検出              │
│  2. フォーマット変換 │ PDF/Word/PPT → 構造化テキスト           │
│  3. 要素分類        │ Title, Section, Paragraph, Table, Figure │
│  4. メタデータ抽出   │ タイトル、著者、日付、ページ番号          │
│  5. テーブル変換     │ HTML/Markdown形式に変換（構造保持）      │
│  6. 言語検出        │ 多言語対応（日本語/英語/その他）         │
│  7. クリーニング     │ ヘッダー/フッター除去、正規化            │
│  8. 出力            │ ProcessedDocument (text, metadata)       │
└──────────────────────────────────────────────────────────────────┘
```

**出力データ構造**:

```python
@dataclass
class AcademicPaperDocument:
    """学術論文ドキュメント（学術論文特化）"""
    # === 基本情報 ===
    file_name: str
    file_type: str
    title: str
    
    # === 著者情報 ===
    authors: list[Author]
    
    # === 識別子 ===
    doi: str | None
    arxiv_id: str | None
    pmid: str | None  # PubMed ID
    
    # === 出版情報 ===
    publication_year: int | None
    venue: str | None  # ジャーナル or 会議名
    venue_type: str  # "journal" | "conference" | "preprint"
    
    # === 構造化コンテンツ ===
    abstract: str
    sections: list[AcademicSection]  # IMRaD構造
    tables: list[Table]
    figures: list[Figure]
    equations: list[Equation]
    
    # === 引用関係 ===
    references: list[Reference]
    inline_citations: list[InlineCitation]
    
    # === メタデータ ===
    keywords: list[str]
    citation_count: int | None
    page_count: int
    language: str
    processing_timestamp: datetime

@dataclass
class Author:
    """著者"""
    name: str
    affiliation: str | None
    email: str | None
    orcid: str | None  # ORCID識別子

@dataclass
class AcademicSection:
    """学術論文セクション（IMRaD対応）"""
    heading: str
    level: int  # 1-6
    section_type: str  # "introduction" | "methods" | "results" | "discussion" | "other"
    content: str
    page_numbers: list[int]
    subsections: list['AcademicSection']

@dataclass
class Table:
    """テーブル"""
    table_id: str
    caption: str | None
    content: str  # HTML or Markdown format
    page_number: int

@dataclass
class Figure:
    """図"""
    figure_id: str
    caption: str | None
    image_path: str | None
    page_number: int

@dataclass
class Equation:
    """数式"""
    equation_id: str
    latex: str
    description: str | None  # テキスト化した説明
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
class InlineCitation:
    """本文中の引用"""
    position: int  # 文字位置
    ref_ids: list[str]  # 参照先（複数可）
    context: str  # 引用周辺のテキスト
```

---

#### REQ-IDX-002: テキストユニット分割

**EARS Pattern**: Ubiquitous

> The system SHALL split documents into configurable TextUnits with chunk_size and chunk_overlap parameters.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-002 |
| 優先度 | P0 (必須) |
| 入力 | Document[], ChunkConfig |
| 出力 | TextUnit[] (id, text, n_tokens, document_ids) |
| 受入基準 | デフォルト300トークン、オーバーラップ100トークン |

---

#### REQ-IDX-003: ベクトルエンベディング

**EARS Pattern**: Ubiquitous

> The system SHALL generate vector embeddings for TextUnits using configurable embedding models.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-003 |
| 優先度 | P0 (必須) |
| 入力 | TextUnit[] |
| 出力 | Embedding[] (id, vector, dimensions) |
| 受入基準 | OpenAI/Azure OpenAI/ローカルモデル対応 |
| 対応アーキテクチャ | 全アーキテクチャ共通 |

---

#### REQ-IDX-004: NLPベース軽量インデックス

**EARS Pattern**: Ubiquitous

> The system SHALL extract keywords and named entities using NLP techniques without LLM calls for cost-efficient indexing.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-004 |
| 優先度 | P0 (必須) |
| 入力 | TextUnit[] |
| 出力 | NLPFeatures[] (keywords, entities, noun_phrases) |
| 受入基準 | LLMコスト0、spaCy/NLTK使用 |
| 対応アーキテクチャ | LazyGraphRAG, Progressive (Level 0-1) |

---

#### REQ-IDX-005: エンティティ抽出

**EARS Pattern**: Conditional

> IF full GraphRAG mode is enabled, the system SHALL extract entities (person, organization, geo, event, concept) from TextUnits using LLM.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-005 |
| 優先度 | P1 (高) |
| 入力 | TextUnit[] |
| 出力 | Entity[] (title, type, description, text_unit_ids) |
| 受入基準 | precision > 80%, recall > 70% |
| 対応アーキテクチャ | GraphRAG, Progressive (Level 2+) |

**学術論文向けエンティティタイプ**:

| カテゴリ | エンティティタイプ | 例 |
|----------|------------------|-----|
| **人物** | RESEARCHER | "Geoffrey Hinton", "岡野原大輔" |
| **組織** | ORGANIZATION | "Google DeepMind", "東京大学" |
| **手法** | METHOD | "Transformer", "Attention Mechanism" |
| **モデル** | MODEL | "GPT-4", "BERT", "ResNet" |
| **データセット** | DATASET | "ImageNet", "COCO", "SQuAD" |
| **評価指標** | METRIC | "F1-score", "BLEU", "Accuracy" |
| **タスク** | TASK | "Image Classification", "Question Answering" |
| **概念** | CONCEPT | "Self-Attention", "Knowledge Distillation" |
| **ツール** | TOOL | "PyTorch", "TensorFlow", "Hugging Face" |
| **論文** | PAPER | "Attention Is All You Need" |

---

#### REQ-IDX-005a: 引用ネットワーク構築

**EARS Pattern**: Conditional

> IF citation network is enabled, the system SHALL build a citation graph from paper references and inline citations.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-005a |
| 優先度 | P0 (必須) |
| 入力 | AcademicPaperDocument[] |
| 出力 | CitationGraph (nodes: Paper[], edges: CitationEdge[]) |
| 受入基準 | DOIマッチング率 > 80% |
| 対応アーキテクチャ | 全アーキテクチャ |

**引用ネットワーク構造**:

```
                    引用ネットワーク

        【被引用論文】           【引用論文】
    ╔═══════════╗       ╔═══════════╗
    ║ Vaswani+17 ║───────║ BERT 2018 ║
    ║ Attention  ║       ║           ║
    ╙═══════════╜       ╙═══════════╜
           │                     │
           └─────────────────────┤
                                 ▼
                         ╔═══════════╗
                         ║ GPT-3 2020 ║
                         ╙═══════════╜
```

**引用エッジ種別**:

| エッジ種別 | 説明 | 検索活用 |
|---------|------|----------|
| **cites** | AがBを引用 | 先行研究追跡 |
| **cited_by** | AがBに引用される | 後続研究発見 |
| **co_citation** | AとBが同じ論文に引用される | 関連論文探索 |
| **bibliographic_coupling** | AとBが同じ論文を引用 | 類似研究探索 |

**外部API連携（オプション）**:

| API | 取得データ | 用途 |
|-----|---------|------|
| **Semantic Scholar** | 引用数、被引用論文 | 引用ネットワーク拡張 |
| **CrossRef** | DOI解決、メタデータ | 論文名寄せ |
| **OpenAlex** | 著者、機関、分野 | エンティティ拡充 |
| **CORE** | OA論文フルテキスト、メタデータ | オープンアクセス論文取得 |
| **Unpaywall** | OA版URL、OAステータス | 無料アクセス可能版の発見 |

---

#### REQ-IDX-006: リレーションシップ抽出

**EARS Pattern**: Conditional

> IF full GraphRAG mode is enabled, the system SHALL extract relationships between entities from TextUnits using LLM.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-006 |
| 優先度 | P1 (高) |
| 入力 | TextUnit[], Entity[] |
| 出力 | Relationship[] (source, target, description, weight) |
| 受入基準 | エンティティ間の有意な関係を抽出 |
| 対応アーキテクチャ | GraphRAG, Progressive (Level 2+) |

---

#### REQ-IDX-007: コミュニティ検出

**EARS Pattern**: Conditional

> IF community detection is enabled, the system SHALL detect hierarchical communities from the entity-relationship graph using the Leiden algorithm.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-007 |
| 優先度 | P1 (高) |
| 入力 | Entity[], Relationship[] |
| 出力 | Community[] (id, level, entity_ids, relationship_ids) |
| 受入基準 | 階層的クラスタリング、複数レベル |
| 対応アーキテクチャ | GraphRAG, LazyGraphRAG, Progressive (Level 1+) |

---

#### REQ-IDX-008: コミュニティレポート生成

**EARS Pattern**: Conditional

> IF full GraphRAG mode is enabled, the system SHALL generate summary reports for each community.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-008 |
| 優先度 | P1 (高) |
| 入力 | Community[], Entity[], Relationship[] |
| 出力 | CommunityReport[] (community_id, title, summary, findings) |
| 受入基準 | 各コミュニティのエグゼクティブサマリー生成 |
| 対応アーキテクチャ | GraphRAG, Progressive (Level 3+) |

---

#### REQ-IDX-009: Progressive インデックス管理

**EARS Pattern**: Ubiquitous

> The system SHALL manage a 5-level progressive index that can be built incrementally based on usage patterns.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-009 |
| 優先度 | P0 (必須) |
| 入力 | Document[], TargetLevel |
| 出力 | ProgressiveIndex with levels 0-4 |
| 受入基準 | 各レベルの独立構築、段階的拡張可能 |
| 対応アーキテクチャ | Progressive GraphRAG |

**インデックスレベル定義**:

| Level | 名称 | 内容 | 構築コスト | 使用技術 |
|-------|------|------|-----------|---------|
| 0 | Raw | チャンク + 埋め込み | 💰 | Embedding |
| 1 | Lazy | 名詞句グラフ + コミュニティ | 💰 | NLP |
| 2 | Partial | エンティティ + 関係性 | 💰💰💰 | LLM |
| 3 | Full | コミュニティサマリー | 💰💰💰💰 | LLM |
| 4 | Enhanced | 事前抽出クレーム | 💰💰💰💰💰 | LLM |

---

### 2.2 クエリ機能 (Query Domain)

#### REQ-QRY-001: Vector Search (Baseline RAG)

**EARS Pattern**: Event-driven

> WHEN a user submits a query, the system SHALL perform vector similarity search on TextUnits.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-001 |
| 優先度 | P0 (必須) |
| 入力 | query: str, top_k: int |
| 出力 | SearchResult (response, source_chunks, scores) |
| 受入基準 | レイテンシ < 1秒 |
| 対応アーキテクチャ | Baseline RAG, Hybrid |

---

#### REQ-QRY-002: Global Search

**EARS Pattern**: Event-driven

> WHEN a user submits a global query, the system SHALL generate a response using community reports in a map-reduce pattern.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-002 |
| 優先度 | P1 (高) |
| 入力 | query: str, community_level: int |
| 出力 | SearchResult (response, citations, context_data) |
| 受入基準 | データセット全体の要約質問に回答可能 |
| 対応アーキテクチャ | GraphRAG, Hybrid, Unified |

---

#### REQ-QRY-003: Local Search

**EARS Pattern**: Event-driven

> WHEN a user submits a local query, the system SHALL generate a response by combining entity information, relationships, and source text.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-003 |
| 優先度 | P1 (高) |
| 入力 | query: str, top_k_entities: int |
| 出力 | SearchResult (response, citations, entities, relationships) |
| 受入基準 | 特定エンティティに関する詳細質問に回答可能 |
| 対応アーキテクチャ | GraphRAG, Hybrid, Unified |

---

#### REQ-QRY-004: Lazy Search

**EARS Pattern**: Ubiquitous

> The system SHALL provide budget-controlled lazy search with iterative deepening, query expansion, relevance testing, and claim extraction.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-004 |
| 優先度 | P0 (必須) |
| 入力 | query: str, budget: int, context_tokens: int |
| 出力 | SearchResult with claims and context |
| 受入基準 | GraphRAGの1/100コストで同等品質 |
| 対応アーキテクチャ | LazyGraphRAG, Hybrid, Unified |

**サブコンポーネント**:
- REQ-QRY-004a: QueryExpander - クエリ拡張
- REQ-QRY-004b: RelevanceTester - 関連性評価
- REQ-QRY-004c: ClaimExtractor - クレーム抽出
- REQ-QRY-004d: IterativeDeepener - 反復深化

---

#### REQ-QRY-005: Hybrid Search with RRF

**EARS Pattern**: Event-driven

> WHEN hybrid mode is selected, the system SHALL execute multiple search engines in parallel and merge results using Reciprocal Rank Fusion (RRF).

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-005 |
| 優先度 | P1 (高) |
| 入力 | query: str, engines: List[SearchEngine] |
| 出力 | SearchResult with merged context |
| 受入基準 | 複数エンジンの結果を統合、網羅性最大化 |
| 対応アーキテクチャ | Hybrid GraphRAG |

**RRF アルゴリズム**:
```
score(d) = Σ 1/(k + rank_i(d))
```
- k: ランキング定数（デフォルト60）
- rank_i(d): エンジンiでのドキュメントdの順位

---

#### REQ-QRY-006: Query Router (Unified)

**EARS Pattern**: Ubiquitous

> The system SHALL classify incoming queries and route them to the optimal search engine based on query characteristics.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-006 |
| 優先度 | P0 (必須) |
| 入力 | query: str |
| 出力 | SearchMode (LAZY / GRAPHRAG / HYBRID / VECTOR) |
| 受入基準 | 分類精度 > 85% |
| 対応アーキテクチャ | Unified GraphRAG |

**分類基準（学術論文向け）**:

| クエリタイプ | 例 | 選択モード | 理由 |
|-------------|-----|-----------|------|
| **サーベイ・傾向分析** | 「この分野の研究動向は？」「主要なアプローチは？」 | GRAPHRAG | 広範な文脈が必要 |
| **手法調査** | 「Transformerを使った手法は？」「○○の実装方法は？」 | LAZY | 探索的、特定トピック |
| **手法比較** | 「BERTとGPTの違いは？」「精度比較は？」 | HYBRID | 複数視点の網羅性 |
| **具体的事実** | 「○○のSOTA精度は？」「データセットサイズは？」 | VECTOR | ピンポイント検索 |
| **先行研究調査** | 「○○を最初に提案したのは？」「この手法の元論文は？」 | LAZY | 引用ネットワーク活用 |
| **ベンチマーク調査** | 「ImageNetでの性能一覧は？」「○○タスクの評価指標は？」 | HYBRID | 表形式データの網羅 |

**実装アプローチ**:
- Level 1: ルールベース（キーワード・パターンマッチング）
- Level 2: ML分類器（軽量モデル）
- Level 3: LLM分類（高精度・高コスト）

---

#### REQ-QRY-007: Budget-Controlled Search (Progressive)

**EARS Pattern**: Ubiquitous

> The system SHALL execute search within the specified cost budget, progressively deepening through index levels.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-007 |
| 優先度 | P0 (必須) |
| 入力 | query: str, budget: CostBudget |
| 出力 | SearchResult with cost_used |
| 受入基準 | 予算内での最適品質を保証 |
| 対応アーキテクチャ | Progressive GraphRAG |

**コスト予算レベル**:

| Budget | 使用レベル | ユースケース |
|--------|-----------|-------------|
| MINIMAL | Level 0-1 | 探索的・ワンオフ検索 |
| STANDARD | Level 0-2 | 一般的なクエリ |
| PREMIUM | Level 0-3 | 高品質が必要な場合 |
| UNLIMITED | Level 0-4 | 最高品質 |

---

### 2.3 アーキテクチャ制御機能 (Architecture Controller)

#### REQ-ARC-001: Unified GraphRAG Controller

**EARS Pattern**: Ubiquitous

> The system SHALL provide a unified interface that dynamically selects the optimal search strategy based on query analysis.

| 項目 | 内容 |
|------|------|
| ID | REQ-ARC-001 |
| 優先度 | P0 (必須) |
| 入力 | query: str, mode: SearchMode = AUTO |
| 出力 | SearchResult |
| 受入基準 | 自動モード選択でHybrid比30%コスト削減 |

```python
class UnifiedGraphRAG:
    def search(self, query: str, mode: SearchMode = SearchMode.AUTO):
        if mode == SearchMode.AUTO:
            mode = self.router.classify(query)
        
        match mode:
            case SearchMode.LAZY:
                return await self.lazy_search.search(query)
            case SearchMode.GRAPHRAG:
                return await self.graphrag.search(query)
            case SearchMode.HYBRID:
                return await self.hybrid_search(query)
            case SearchMode.VECTOR:
                return await self.vector_search.search(query)
```

---

#### REQ-ARC-002: Progressive GraphRAG Controller

**EARS Pattern**: Ubiquitous

> The system SHALL provide a progressive search interface that builds indexes on-demand and searches within budget constraints.

| 項目 | 内容 |
|------|------|
| ID | REQ-ARC-002 |
| 優先度 | P0 (必須) |
| 入力 | query: str, budget: CostBudget = STANDARD |
| 出力 | SearchResult with levels_used |
| 受入基準 | 初期コストゼロで運用開始可能 |

```python
class ProgressiveGraphRAG:
    def search(self, query: str, budget: CostBudget = CostBudget.STANDARD):
        max_level = self._budget_to_level(budget)
        results = []
        
        for level in range(max_level + 1):
            if not self.index.has_level(level):
                await self.index.build_level(level, query)
            results.append(await self._search_at_level(query, level))
        
        return self._merge_progressive(results)
```

---

#### REQ-ARC-003: Hybrid GraphRAG Controller

**EARS Pattern**: Event-driven

> WHEN hybrid mode is requested, the system SHALL execute all configured search engines in parallel and merge results.

| 項目 | 内容 |
|------|------|
| ID | REQ-ARC-003 |
| 優先度 | P1 (高) |
| 入力 | query: str |
| 出力 | SearchResult with merged context |
| 受入基準 | 並列実行、RRFマージ |

---

### 2.4 API/CLI 機能

#### REQ-API-001: Python API

**EARS Pattern**: Ubiquitous

> The system SHALL provide a Python API for programmatic access to all search and indexing functions.

| 項目 | 内容 |
|------|------|
| ID | REQ-API-001 |
| 優先度 | P0 (必須) |
| 入力 | Python method calls |
| 出力 | SearchResult, IndexResult |
| 受入基準 | async対応、型ヒント完備 |

```python
# 使用例
from monjyu import MONJYU

# Unified モード（推奨）
monjyu = MONJYU(mode="unified")
result = await monjyu.search("全体の傾向は？")

# Progressive モード
monjyu = MONJYU(mode="progressive")
result = await monjyu.search("詳細を教えて", budget="premium")

# 明示的モード指定
result = await monjyu.search("AとBの違いは？", mode="hybrid")
```

---

#### REQ-API-002: CLI

**EARS Pattern**: Ubiquitous

> The system SHALL provide a CLI for index creation and query execution.

| 項目 | 内容 |
|------|------|
| ID | REQ-API-002 |
| 優先度 | P0 (必須) |
| 入力 | CLI commands |
| 出力 | Console output, files |
| 受入基準 | monjyu init/index/query コマンド |

```bash
# インデックス作成
monjyu index --input ./docs --mode progressive --level 1

# クエリ実行
monjyu query "全体の傾向は？" --mode unified
monjyu query "詳細を教えて" --mode progressive --budget premium
```

---

#### REQ-API-003: ストリーミング出力

**EARS Pattern**: Optional Feature

> WHERE streaming is enabled, the system SHALL stream response tokens as they are generated.

| 項目 | 内容 |
|------|------|
| ID | REQ-API-003 |
| 優先度 | P1 (高) |
| 入力 | query: str, stream: bool = True |
| 出力 | AsyncGenerator[str, None] |
| 受入基準 | リアルタイムトークン出力 |

---

#### REQ-API-004: MCP Server

**EARS Pattern**: Ubiquitous

> The system SHALL provide a Model Context Protocol (MCP) server for integration with AI assistants such as Claude, GitHub Copilot, and other MCP-compatible clients.

| 項目 | 内容 |
|------|------|
| ID | REQ-API-004 |
| 優先度 | P0 (必須) |
| 入力 | MCP tool calls (JSON-RPC) |
| 出力 | MCP tool responses |
| 受入基準 | MCP仕様準拠、stdio/SSE転送対応 |

**提供ツール一覧**:

| ツール名 | 説明 | パラメータ |
|---------|------|-----------|
| `monjyu_search` | 論文検索（統合インターフェース） | query, mode?, budget? |
| `monjyu_index` | 論文インデックス構築 | input_path, level? |
| `monjyu_get_paper` | 特定論文の詳細取得 | doi?, arxiv_id?, title? |
| `monjyu_citations` | 引用ネットワーク取得 | paper_id, depth? |
| `monjyu_summarize` | 論文/トピックのサマリー生成 | query, scope? |
| `monjyu_compare` | 複数手法の比較分析 | methods[], criteria[]? |
| `monjyu_survey` | 文献サーベイ実行 | topic, year_range?, limit? |

**MCP Server 設定例**:

```json
{
  "mcpServers": {
    "monjyu": {
      "command": "python",
      "args": ["-m", "monjyu.mcp_server"],
      "env": {
        "MONJYU_INDEX_PATH": "./index",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

**ツール詳細**:

##### monjyu_search
```json
{
  "name": "monjyu_search",
  "description": "学術論文を検索し、クエリに関連する情報を返します。Unified/Lazy/GraphRAG/Hybridモードを自動または手動で選択できます。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "検索クエリ（自然言語）"
      },
      "mode": {
        "type": "string",
        "enum": ["auto", "lazy", "graphrag", "hybrid", "vector"],
        "default": "auto",
        "description": "検索モード"
      },
      "budget": {
        "type": "string",
        "enum": ["minimal", "standard", "premium", "unlimited"],
        "default": "standard",
        "description": "コスト予算（Progressiveモード時）"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "description": "返却する結果数"
      }
    },
    "required": ["query"]
  }
}
```

##### monjyu_get_paper
```json
{
  "name": "monjyu_get_paper",
  "description": "DOIまたはarXiv IDから論文の詳細情報を取得します。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "doi": {
        "type": "string",
        "description": "論文のDOI (例: 10.1234/example)"
      },
      "arxiv_id": {
        "type": "string",
        "description": "arXiv ID (例: 2301.00001)"
      },
      "include_citations": {
        "type": "boolean",
        "default": false,
        "description": "引用・被引用情報を含めるか"
      }
    }
  }
}
```

##### monjyu_survey
```json
{
  "name": "monjyu_survey",
  "description": "指定トピックの文献サーベイを実行し、主要な研究動向・手法をまとめます。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "サーベイ対象トピック"
      },
      "year_range": {
        "type": "object",
        "properties": {
          "start": { "type": "integer" },
          "end": { "type": "integer" }
        },
        "description": "対象年範囲"
      },
      "focus": {
        "type": "string",
        "enum": ["methods", "datasets", "applications", "all"],
        "default": "all",
        "description": "フォーカス領域"
      }
    },
    "required": ["topic"]
  }
}
```

**使用例（Claude Desktop）**:

```
User: Transformerの注意機構について最新の研究動向を教えて

Claude: monjyu_surveyツールを使って調査します...

[monjyu_survey: topic="Transformer attention mechanism", year_range={start:2022, end:2025}]

調査結果に基づくと、以下の3つの主要な研究方向があります：
1. 効率的注意機構（Linear Attention, Flash Attention）
2. 長文脈対応（ALiBi, RoPE）
3. マルチモーダル注意機構...
```

---

### 2.5 ストレージ機能

#### REQ-STG-001: インデックス永続化

**EARS Pattern**: Ubiquitous

> The system SHALL persist all index data to configurable storage backends.

| 項目 | 内容 |
|------|------|
| ID | REQ-STG-001 |
| 優先度 | P0 (必須) |
| 入力 | Index data |
| 出力 | Persisted files/database |
| 受入基準 | Parquet形式、ファイルシステム/クラウドストレージ対応 |

---

#### REQ-STG-002: ベクトルストレージ

**EARS Pattern**: Ubiquitous

> The system SHALL store and query vector embeddings using configurable vector database backends.

| 項目 | 内容 |
|------|------|
| ID | REQ-STG-002 |
| 優先度 | P0 (必須) |
| 入力 | Embedding[] |
| 出力 | Similarity search results |
| 受入基準 | LanceDB/FAISS/Azure AI Search対応 |

---

#### REQ-STG-003: Progressive インデックスキャッシュ

**EARS Pattern**: Conditional

> IF Progressive mode is enabled, the system SHALL cache dynamically built index levels for reuse.

| 項目 | 内容 |
|------|------|
| ID | REQ-STG-003 |
| 優先度 | P0 (必須) |
| 入力 | Index level data |
| 出力 | Cached index |
| 受入基準 | レベル別キャッシュ、LRU eviction |
| 対応アーキテクチャ | Progressive GraphRAG |

---

## 3. 非機能要件

### 3.1 パフォーマンス要件

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-PERF-001 | Vector Search レイテンシ | < 500ms |
| NFR-PERF-002 | Lazy Search レイテンシ | < 5s |
| NFR-PERF-003 | Hybrid Search レイテンシ | < 10s |
| NFR-PERF-004 | インデックス構築スループット | > 100 docs/min |
| NFR-PERF-005 | 同時クエリ処理 | > 100 concurrent (本番) |

### 3.1.1 スケーラビリティ要件 (本番環境)

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-SCALE-001 | MCP Server インスタンス数 | 1-20 (オートスケール) |
| NFR-SCALE-002 | 同時接続ユーザー数 | > 100 |
| NFR-SCALE-003 | スケールアウト時間 | < 60s |
| NFR-SCALE-004 | ゼロダウンタイムデプロイ | 対応 |
| NFR-SCALE-005 | リージョン分散 | Japan East + Japan West |

### 3.2 コスト要件

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-COST-001 | Level 0-1 インデックス | LLMコスト $0 |
| NFR-COST-002 | Lazy Search vs GraphRAG | 1/100 コスト |
| NFR-COST-003 | Unified vs Hybrid | 30% コスト削減 |

### 3.3 品質要件

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-QUAL-001 | 回答の正確性（RAGAS） | > 0.8 |
| NFR-QUAL-002 | Query Router 分類精度 | > 85% |
| NFR-QUAL-003 | テストカバレッジ | > 80% |

### 3.4 学術論文固有の品質要件

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-ACAD-001 | PDFレイアウト解析精度 | > 95%（テキスト抽出） |
| NFR-ACAD-002 | 図表キャプション対応率 | > 90% |
| NFR-ACAD-003 | 参考文献抽出精度 | > 85% |
| NFR-ACAD-004 | DOIマッチング率 | > 80% |
| NFR-ACAD-005 | IMRaDセクション識別精度 | > 80% |
| NFR-ACAD-006 | 数式LaTeX変換精度 | > 90% |
| NFR-ACAD-007 | 引用ネットワーク網羅率 | > 70%（コーパス内） |

### 3.5 拡張性要件

| ID | 要件 | 説明 |
|----|------|------|
| NFR-EXT-001 | LLMプロバイダー | OpenAI, Azure OpenAI, Ollama (Local) |
| NFR-EXT-002 | ストレージバックエンド | File, S3, Azure Blob |
| NFR-EXT-003 | ベクトルDB | LanceDB, FAISS, Azure AI Search |
| NFR-EXT-004 | 学術API連携 | Semantic Scholar, CrossRef, OpenAlex, CORE, Unpaywall |
| NFR-EXT-005 | PDF処理バックエンド | Azure Document Intelligence, unstructured |

### 3.6 環境別構成

**ローカル開発環境 (Windows)**:

| コンポーネント | プロバイダー | 説明 |
|----------------|-------------|------|
| **LLM** | Ollama | ローカルLLM (llama3, mistral, phi3等) |
| **Embedding** | Ollama | nomic-embed-text, mxbai-embed-large等 |
| **PDF処理** | unstructured | ローカルPDF解析 (PyMuPDF) |
| **ベクトルDB** | LanceDB | ローカルファイルベース |
| **ストレージ** | Local File | Parquetファイル |

```yaml
# config/local.yaml
llm:
  provider: ollama
  model: llama3.2
  base_url: http://192.168.224.1:11434

embedding:
  provider: ollama
  model: nomic-embed-text
  base_url: http://192.168.224.1:11434

pdf_processing:
  provider: unstructured
  strategy: fast  # fast / hi_res / ocr_only

vector_store:
  provider: lancedb
  path: ./storage/lancedb

storage:
  provider: file
  base_path: ./storage/index
```

**本番環境 (Azure) - スケールアウト対応**:

| コンポーネント | プロバイダー | 説明 |
|----------------|-------------|------|
| **LLM** | Azure OpenAI | GPT-4o, GPT-4o-mini |
| **Embedding** | Azure OpenAI | text-embedding-3-large |
| **PDF処理** | Azure Document Intelligence | 高精度レイアウト解析 |
| **ベクトルDB** | Azure AI Search | マネージドベクトル検索 |
| **ストレージ** | Azure Blob Storage | スケーラブルストレージ |
| **MCP Server** | Azure Container Apps | スケールアウト対応 |
| **API Gateway** | Azure API Management | 認証・レート制限・ログ |
| **キャッシュ** | Azure Cache for Redis | セッション・結果キャッシュ |
| **モニタリング** | Azure Monitor + App Insights | ログ・メトリクス・トレース |

**スケールアウトアーキテクチャ**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Azure 本番環境                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   【クライアント】                                                          │
│   Claude Desktop / GitHub Copilot / Custom Apps                        │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Azure API Management (Gateway)                                  │  │
│   │  - 認証 (Entra ID / API Key)                                      │  │
│   │  - レート制限 (100 req/min/user)                                  │  │
│   │  - ログ・メトリクス                                              │  │
│   └─────────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Azure Container Apps (スケールアウト)                           │  │
│   │                                                                 │  │
│   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   │  │
│   │  │ MCP Server  │ │ MCP Server  │ │ MCP Server  │   │  │
│   │  │  Replica 1  │ │  Replica 2  │ │  Replica N  │   │  │
│   │  └───────────────┘ └───────────────┘ └───────────────┘   │  │
│   │       │               │               │                   │  │
│   │       └───────────────┼───────────────┘                   │  │
│   │                       ▼                                       │  │
│   │              Auto-scaling (1-20 replicas)                      │  │
│   │              - CPU > 70% → Scale Out                          │  │
│   │              - CPU < 30% → Scale In                           │  │
│   └─────────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Azure Cache for Redis (セッション & 結果キャッシュ)              │  │
│   └─────────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  バックエンドサービス (マネージド・自動スケール)               │  │
│   │                                                                 │  │
│   │  ┌─────────────┐  ┌───────────────┐  ┌────────────────┐  │  │
│   │  │Azure OpenAI│  │Azure AI Search│  │Azure Document │  │  │
│   │  │  (LLM)     │  │ (ベクトルDB) │  │ Intelligence  │  │  │
│   │  └─────────────┘  └───────────────┘  └────────────────┘  │  │
│   │                                                                 │  │
│   │  ┌─────────────────────────────────────────────────┐  │  │
│   │  │  Azure Blob Storage (インデックス・論文データ)      │  │  │
│   │  └─────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```yaml
# config/production.yaml
llm:
  provider: azure_openai
  deployment: gpt-4o
  api_version: "2024-08-01-preview"
  # AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY は環境変数

embedding:
  provider: azure_openai
  deployment: text-embedding-3-large
  api_version: "2024-08-01-preview"

pdf_processing:
  provider: azure_document_intelligence
  model: prebuilt-layout
  api_version: "2024-02-29-preview"
  features:
    - tables
    - figures
    - formulas
  # AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY は環境変数

vector_store:
  provider: azure_ai_search
  index_name: monjyu-papers
  # AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY は環境変数

storage:
  provider: azure_blob
  container: monjyu-index
  # AZURE_STORAGE_CONNECTION_STRING は環境変数

# スケールアウト設定
scaling:
  min_replicas: 1
  max_replicas: 20
  target_cpu_percent: 70

# キャッシュ設定
cache:
  provider: redis
  # AZURE_REDIS_CONNECTION_STRING は環境変数
  ttl_seconds: 3600  # 結果キャッシュ1時間

# API管理
api_management:
  rate_limit: 100  # requests/min/user
  auth_provider: entra_id
```

**Azure Container Apps デプロイ設定**:

```yaml
# infra/container-app.yaml
properties:
  configuration:
    ingress:
      external: true
      targetPort: 8000
      transport: http
    secrets:
      - name: azure-openai-key
        keyVaultUrl: https://monjyu-kv.vault.azure.net/secrets/openai-key
  template:
    containers:
      - name: monjyu-mcp-server
        image: monjyu.azurecr.io/mcp-server:latest
        resources:
          cpu: 1.0
          memory: 2Gi
        env:
          - name: MONJYU_ENV
            value: production
          - name: AZURE_OPENAI_API_KEY
            secretRef: azure-openai-key
    scale:
      minReplicas: 1
      maxReplicas: 20
      rules:
        - name: cpu-scaling
          custom:
            type: cpu
            metadata:
              type: Utilization
              value: "70"
        - name: http-scaling
          http:
            metadata:
              concurrentRequests: "50"
```

**環境切り替え**:

```bash
# ローカル開発
MONJYU_ENV=local monjyu index --input ./papers

# 本番環境
MONJYU_ENV=production monjyu index --input ./papers
```

```python
# Python API
from monjyu import MONJYU

# ローカル
monjyu = MONJYU(env="local")  # Ollama 使用

# 本番
monjyu = MONJYU(env="production")  # Azure 使用

# 明示的指定
monjyu = MONJYU(
    llm_provider="ollama",
    llm_model="llama3.2",
    pdf_provider="unstructured"
)
```

### 3.7 セキュリティ要件

| ID | 要件 | 説明 |
|----|------|------|
| NFR-SEC-001 | 認証 | Azure Entra ID / API Key認証 |
| NFR-SEC-002 | 認可 | RBAC（管理者/研究者/閲覧者） |
| NFR-SEC-003 | 通信暗号化 | TLS 1.3必須（本番環境） |
| NFR-SEC-004 | 保存データ暗号化 | Azure Storage暗号化（AES-256） |
| NFR-SEC-005 | APIキー管理 | Azure Key Vault使用 |
| NFR-SEC-006 | 監査ログ | 全API呼び出しのログ記録 |
| NFR-SEC-007 | レート制限 | 100 req/min/user（DoS対策） |

**ロール定義**:

| ロール | 権限 |
|--------|------|
| **admin** | 全操作（インデックス構築・削除・設定変更） |
| **researcher** | 検索・閲覧・エクスポート |
| **viewer** | 検索・閲覧のみ |

### 3.8 可用性要件

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-AVL-001 | サービス稼働率（SLA） | 99.9%（本番環境） |
| NFR-AVL-002 | 計画メンテナンス | 月1回、深夜帯 |
| NFR-AVL-003 | RTO（目標復旧時間） | < 1時間 |
| NFR-AVL-004 | RPO（目標復旧時点） | < 24時間 |
| NFR-AVL-005 | リージョン冗長 | Japan East + Japan West |
| NFR-AVL-006 | ヘルスチェック | 30秒間隔 |

### 3.9 運用要件

| ID | 要件 | 説明 |
|----|------|------|
| NFR-OPS-001 | バックアップ | 日次自動バックアップ（30日保持） |
| NFR-OPS-002 | リストア | バックアップからの復旧手順整備 |
| NFR-OPS-003 | 監視アラート | CPU/メモリ/エラー率の閾値アラート |
| NFR-OPS-004 | ログ集約 | Azure Monitor / Application Insights |
| NFR-OPS-005 | メトリクス | レイテンシ・スループット・エラー率 |
| NFR-OPS-006 | トレース | 分散トレーシング（リクエスト追跡） |
| NFR-OPS-007 | ダッシュボード | 運用監視ダッシュボード |

**アラート閾値**:

| メトリクス | Warning | Critical |
|-----------|---------|----------|
| CPU使用率 | > 70% | > 90% |
| メモリ使用率 | > 70% | > 90% |
| エラー率 | > 1% | > 5% |
| レイテンシ (p95) | > 5s | > 10s |
| ディスク使用率 | > 70% | > 90% |

### 3.10 外部連携要件

| ID | 要件 | 説明 |
|----|------|------|
| NFR-INT-001 | Semantic Scholar API | 引用数・被引用論文取得 |
| NFR-INT-002 | CrossRef API | DOI解決・メタデータ取得 |
| NFR-INT-003 | OpenAlex API | 著者・機関・分野情報 |
| NFR-INT-004 | CORE API | OA論文フルテキスト取得 |
| NFR-INT-005 | Unpaywall API | OA版URL検索 |
| NFR-INT-006 | API呼び出し制限対応 | レート制限・リトライ・キャッシュ |
| NFR-INT-007 | APIフォールバック | API障害時の代替処理 |

**API呼び出し設定**:

```yaml
external_apis:
  semantic_scholar:
    base_url: https://api.semanticscholar.org/graph/v1
    rate_limit: 100/5min
    timeout: 30s
    retry:
      max_attempts: 3
      backoff: exponential
    cache_ttl: 24h
  
  crossref:
    base_url: https://api.crossref.org
    rate_limit: 50/sec (polite pool)
    timeout: 30s
    cache_ttl: 7d
  
  core:
    base_url: https://api.core.ac.uk/v3
    rate_limit: 10/sec
    timeout: 60s
    cache_ttl: 24h
```

### 3.11 エラー処理要件

| ID | 要件 | 説明 |
|----|------|------|
| NFR-ERR-001 | エラー分類 | Transient / Permanent / User Error |
| NFR-ERR-002 | 自動リトライ | Transientエラーは自動リトライ |
| NFR-ERR-003 | リトライ戦略 | Exponential backoff（最大3回） |
| NFR-ERR-004 | サーキットブレーカー | 連続失敗時のサービス保護 |
| NFR-ERR-005 | グレースフルデグラデーション | 一部機能障害時の縮退運転 |
| NFR-ERR-006 | エラーメッセージ | ユーザーフレンドリーなメッセージ |
| NFR-ERR-007 | エラーログ | 詳細なスタックトレース記録 |

**エラー分類**:

| 分類 | 例 | 対応 |
|------|-----|------|
| **Transient** | ネットワークタイムアウト、API一時障害 | 自動リトライ |
| **Permanent** | 不正なDOI、存在しない論文 | エラー返却 |
| **User Error** | 不正なクエリ形式、認証エラー | 400系エラー返却 |
| **System Error** | 内部エラー、リソース不足 | 500系エラー + アラート |

### 3.12 テスト要件

| ID | 要件 | 説明 | 現状 |
|----|------|------|------|
| NFR-TST-001 | 単体テストカバレッジ | > 80% | ✅ 83% |
| NFR-TST-002 | 統合テスト | 主要フロー網羅 | ✅ 165 tests |
| NFR-TST-003 | E2Eテスト | ユーザーシナリオベース | ✅ 24 tests |
| NFR-TST-004 | パフォーマンステスト | 負荷テスト・ストレステスト | ⏳ 計画中 |
| NFR-TST-005 | 回帰テスト | CI/CDパイプラインで自動実行 | ✅ 実装済み |
| NFR-TST-006 | セキュリティテスト | 脆弱性スキャン | ⏳ 計画中 |

**現在のテスト状況** (2026-01-07):
- **総テスト数**: 2,417 tests (80+ files)
- **Unit Tests**: 2,200+ tests
- **Integration Tests**: 165 tests
- **E2E Tests**: 24 tests
- **カバレッジ**: 83%

**テスト戦略**:

```
┌────────────────────────────────────────────────────────┐
│                    テストピラミッド                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│                    ┌─────────┐                         │
│                    │  E2E    │  10%                    │
│                    │  Tests  │  (主要シナリオ)         │
│                   ┌┴─────────┴┐                        │
│                   │Integration│  20%                   │
│                   │   Tests   │  (API・DB連携)         │
│                  ┌┴───────────┴┐                       │
│                  │    Unit     │  70%                  │
│                  │    Tests    │  (ロジック検証)        │
│                  └─────────────┘                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 3.13 ドキュメント要件

| ID | 要件 | 説明 |
|----|------|------|
| NFR-DOC-001 | API仕様書 | OpenAPI 3.0形式 |
| NFR-DOC-002 | MCPツール仕様 | JSON Schema定義 |
| NFR-DOC-003 | ユーザーガイド | 利用手順・FAQ |
| NFR-DOC-004 | 運用マニュアル | デプロイ・監視・障害対応 |
| NFR-DOC-005 | 開発者ガイド | アーキテクチャ・コーディング規約 |
| NFR-DOC-006 | CHANGELOG | バージョン毎の変更履歴 |

---

## 4. アーキテクチャ選択ガイド

### 4.1 選択フローチャート

```
                    スタート
                       │
                       ▼
              ┌────────────────┐
              │ 網羅性が最優先？ │
              └────────────────┘
                   │
          Yes ─────┴───── No
           │                │
           ▼                ▼
    ┌──────────┐    ┌────────────────┐
    │  Hybrid  │    │ 事前にフルIndex │
    └──────────┘    │ を構築できる？  │
                    └────────────────┘
                           │
                  Yes ─────┴───── No
                   │                │
                   ▼                ▼
            ┌──────────┐    ┌─────────────┐
            │ Unified  │    │ Progressive │
            └──────────┘    └─────────────┘
```

### 4.2 ユースケース別推奨（学術論文ドメイン）

| ユースケース | 推奨アーキテクチャ | 理由 |
|-------------|------------------|------|
| **文献サーベイ** | Unified | クエリに応じた最適モード自動選択 |
| **先行研究調査** | LAZY | 引用ネットワークを活用した効率的探索 |
| **手法比較分析** | Hybrid | 複数論文からの網羅的情報収集 |
| **ベンチマーク調査** | Hybrid | 表・数値データの網羅性 |
| **新規論文の継続追加** | Progressive | 段階的インデックス構築、低初期コスト |
| **特定トピック深掘り** | GRAPHRAG | コミュニティ構造を活用した全体像把握 |
| **クイック検索（日常利用）** | Unified | コストと品質の自動バランス |

---

## 5. 実装優先度

### 5.1 Progressive GraphRAG レベル依存関係

Progressive GraphRAGは「メタアーキテクチャ」であり、各レベルで異なる実装を使用します。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Progressive GraphRAG レベル構造                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 4: Enhanced    ┌─────────────────────────────────────────────┐  │
│  (LLM: 💰💰💰💰💰)     │ GraphRAG + 事前クレーム抽出                │  │
│                       └─────────────────────────────────────────────┘  │
│                                      ▲                                  │
│  Level 3: Full        ┌─────────────────────────────────────────────┐  │
│  (LLM: 💰💰💰💰)       │ GraphRAG（コミュニティサマリー）            │  │
│                       └─────────────────────────────────────────────┘  │
│                                      ▲                                  │
│  Level 2: Partial     ┌─────────────────────────────────────────────┐  │
│  (LLM: 💰💰💰)         │ GraphRAG（エンティティ・関係性抽出）        │  │
│                       └─────────────────────────────────────────────┘  │
│                                      ▲                                  │
│  ════════════════════════════════════╪══════════════════════════════   │
│                          【LLMコスト境界線】                            │
│  ════════════════════════════════════╪══════════════════════════════   │
│                                      ▲                                  │
│  Level 1: Lazy        ┌─────────────────────────────────────────────┐  │
│  (LLM: $0)            │ LazyGraphRAG（NLPベース名詞句グラフ）       │  │
│                       └─────────────────────────────────────────────┘  │
│                                      ▲                                  │
│  Level 0: Raw         ┌─────────────────────────────────────────────┐  │
│  (Embedding only)     │ Baseline RAG（チャンク + ベクトル埋め込み） │  │
│                       └─────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**実装依存関係**:

| レベル | 必要な実装 | 前提レベル | インデックスコスト |
|--------|-----------|-----------|------------------|
| Level 0 | Baseline RAG (Vector Search) | なし | Embedding のみ |
| Level 1 | LazyGraphRAG (NLP + Community) | Level 0 | LLM $0 |
| Level 2 | GraphRAG (Entity/Relationship) | Level 0 | LLM 使用 |
| Level 3 | GraphRAG (Community Reports) | Level 2 | LLM 使用 |
| Level 4 | GraphRAG (Pre-extracted Claims) | Level 3 | LLM 使用 |

> **重要**: Level 0-1 のみで MVP は完結可能。GraphRAG (Level 2+) は後続フェーズで実装。

### 5.2 フェーズ別実装計画

#### Phase 1: MVP (v0.1.0) - Level 0-1 対応

**目標**: LLMコストゼロでの基本機能提供

```
┌────────────────────────────────────────────────────────────┐
│  Phase 1 スコープ                                          │
│                                                            │
│  ✅ Level 0: Baseline RAG                                  │
│  ✅ Level 1: LazyGraphRAG                                  │
│  ❌ Level 2-4: GraphRAG (Phase 2以降)                      │
└────────────────────────────────────────────────────────────┘
```

| 要件ID | 要件名 | 説明 |
|--------|--------|------|
| REQ-IDX-001 | ドキュメントローダー | PDF学術論文の前処理 |
| REQ-IDX-001a | PDF前処理 | Azure Document Intelligence対応 |
| REQ-IDX-002 | テキストユニット分割 | チャンキング |
| REQ-IDX-003 | ベクトルエンベディング | 埋め込み生成 |
| REQ-IDX-004 | NLPベース軽量インデックス | spaCy/名詞句抽出 |
| REQ-IDX-005a | 引用ネットワーク構築 | 論文間引用関係 |
| REQ-QRY-001 | Vector Search | Baseline RAG検索 |
| REQ-QRY-004 | Lazy Search | LazyGraphRAG検索 |
| REQ-API-001 | Python API | 基本API |
| REQ-API-002 | CLI | 基本CLI |
| REQ-API-004 | MCP Server | 基本ツール (search, get_paper) |
| REQ-STG-001 | インデックス永続化 | Parquet保存 |
| REQ-STG-002 | ベクトルストレージ | LanceDB |

#### Phase 2: Unified + GraphRAG (v0.2.0) - Level 2-3 対応

**目標**: Query RouterとGraphRAGの実装

```
┌────────────────────────────────────────────────────────────┐
│  Phase 2 スコープ                                          │
│                                                            │
│  ✅ Level 2: GraphRAG (Entity/Relationship)               │
│  ✅ Level 3: GraphRAG (Community Reports)                 │
│  ✅ Query Router (Unified)                                │
└────────────────────────────────────────────────────────────┘
```

| 要件ID | 要件名 | 説明 |
|--------|--------|------|
| REQ-IDX-005 | エンティティ抽出 | LLMによる学術エンティティ抽出 |
| REQ-IDX-006 | リレーションシップ抽出 | 関係性抽出 |
| REQ-IDX-007 | コミュニティ検出 | Leiden algorithm |
| REQ-IDX-008 | コミュニティレポート生成 | サマリー生成 |
| REQ-QRY-002 | Global Search | GraphRAG Global |
| REQ-QRY-003 | Local Search | GraphRAG Local |
| REQ-QRY-006 | Query Router | クエリ分類・ルーティング |
| REQ-ARC-001 | Unified Controller | 統合インターフェース |

#### Phase 3: Progressive Controller (v0.3.0)

**目標**: 予算制御と段階的インデックス管理

| 要件ID | 要件名 | 説明 |
|--------|--------|------|
| REQ-IDX-009 | Progressive インデックス管理 | レベル別管理 |
| REQ-QRY-007 | Budget-Controlled Search | 予算制御検索 |
| REQ-ARC-002 | Progressive Controller | 段階的検索制御 |
| REQ-STG-003 | Progressive キャッシュ | レベル別キャッシュ |

#### Phase 4: Hybrid & Polish (v1.0.0)

**目標**: 全機能完成、品質向上

| 要件ID | 要件名 | 説明 |
|--------|--------|------|
| REQ-QRY-005 | Hybrid Search with RRF | 並列実行+マージ |
| REQ-ARC-003 | Hybrid Controller | Hybridモード制御 |
| REQ-API-003 | ストリーミング出力 | リアルタイム出力 |
| REQ-API-004 | MCP Server (Full) | 全ツール実装 |
| NFR-* | 全非機能要件達成 | パフォーマンス・品質 |

### 5.3 実装順序の根拠

```
なぜ Phase 1 で Level 0-1 なのか？

1. 【コスト効率】 LLMコスト $0 で MVP 提供可能
2. 【早期価値提供】 LazyGraphRAG は GraphRAG の 1/100 コストで同等品質
3. 【段階的投資】 利用パターンを見てから Level 2+ への投資判断可能
4. 【学術論文特化】 引用ネットワーク（REQ-IDX-005a）は Level 1 で構築可能
```

---

## 6. 用語集

### 6.1 アーキテクチャ用語

| 用語 | 定義 |
|------|------|
| **Baseline RAG** | チャンク分割 + ベクトル検索のシンプルなRAG |
| **GraphRAG** | エンティティ・関係性抽出によるナレッジグラフベースRAG |
| **LazyGraphRAG** | クエリ時に動的に情報抽出する遅延型RAG |
| **Hybrid GraphRAG** | 複数エンジン並列実行 + RRFマージ |
| **Unified GraphRAG** | Query Routerによる動的エンジン選択 |
| **Progressive GraphRAG** | 段階的インデックス + 予算制御 |
| **RRF** | Reciprocal Rank Fusion、複数検索結果のマージ手法 |
| **Query Router** | クエリを分類し最適なエンジンに振り分けるコンポーネント |
| **CostBudget** | 検索コスト上限を指定するパラメータ |

### 6.2 学術論文用語

| 用語 | 定義 |
|------|------|
| **IMRaD** | Introduction, Methods, Results, and Discussion。学術論文の標準的な構成 |
| **DOI** | Digital Object Identifier。論文の一意識別子 |
| **arXiv ID** | arXivプレプリントサーバーの論文ID |
| **Citation Network** | 論文間の引用関係を表すグラフ |
| **Co-citation** | 2つの論文が同じ論文に引用される関係 |
| **Bibliographic Coupling** | 2つの論文が同じ論文を引用する関係 |
| **ORCID** | Open Researcher and Contributor ID。研究者の一意識別子 |
| **Semantic Scholar** | Allen AIの学術論文検索エンジン、引用API提供 |
| **CrossRef** | DOI登録機関、メタデータAPI提供 |
| **OpenAlex** | オープンアクセスの学術データベース |

---

## 7. 参考文献

- [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost-in-local-graph-rag-methods/)
- [Azure AI Search Hybrid Search with RRF](https://learn.microsoft.com/azure/search/hybrid-search-ranking)

---

**文書履歴**:

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-12-24 | 初版（LazyGraphRAGベース） |
| 2.0.0 | 2025-12-24 | EARS形式対応 |
| 3.0.0 | 2025-12-24 | Unified/Progressive GraphRAG追加、学術論文ターゲット、MCP Server、スケールアウト対応、MECE完全化 |
| 3.0.0 | 2025-12-24 | ✅ 承認 |
| **3.1.0** | **2026-01-07** | **実装完了ステータス追加、テスト状況反映（2,417 tests / 83% coverage）** |

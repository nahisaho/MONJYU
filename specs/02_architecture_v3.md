# MONJYU アーキテクチャ設計書 v3.1

**文書番号**: MONJYU-ARCH-003  
**バージョン**: 3.1.0  
**作成日**: 2025-12-27  
**ステータス**: Approved  
**準拠要件**: [01_requirements_v3.md](01_requirements_v3.md) v3.0.0 (Approved)

---

## 1. システム概要

### 1.1 C4 Model - システムコンテキスト図

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              System Context                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│                           【外部システム】                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Semantic     │  │ CrossRef    │  │  OpenAlex   │  │  CORE / Unpaywall   │  │
│  │Scholar API  │  │    API      │  │    API      │  │       API           │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                │                     │             │
│         └────────────────┴────────────────┴─────────────────────┘             │
│                                   │                                            │
│                                   ▼                                            │
│    ┌─────────────┐       ┌───────────────────────────────┐      ┌──────────┐ │
│    │  研究者     │       │           MONJYU              │      │  LLM     │ │
│    │ (User)     │◀─────▶│   学術論文検索・分析システム    │◀────▶│Provider  │ │
│    └─────────────┘       │                               │      │          │ │
│          │               │  - Unified GraphRAG           │      │- Azure   │ │
│          │               │  - Progressive GraphRAG       │      │  OpenAI  │ │
│          ▼               │  - LazyGraphRAG               │      │- Ollama  │ │
│    ┌─────────────┐       └───────────────────────────────┘      └──────────┘ │
│    │   Claude    │                     │                                      │
│    │   Desktop   │                     │                                      │
│    │   /Copilot  │                     ▼                                      │
│    │ (MCP Client)│       ┌───────────────────────────────┐                   │
│    └─────────────┘       │       Storage Layer           │                   │
│                          │  - Azure Blob / Local File    │                   │
│                          │  - Azure AI Search / LanceDB  │                   │
│                          │  - Azure Cache for Redis      │                   │
│                          └───────────────────────────────┘                   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 C4 Model - コンテナ図

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                 MONJYU System                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  【Presentation Layer】                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CLI       │  │ Python API  │  │ MCP Server  │  │  REST API (Future)  │  │
│  │  (Typer)    │  │   (async)   │  │  (stdio/SSE)│  │                     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                │                     │             │
│         └────────────────┴────────────────┴─────────────────────┘             │
│                                   │                                            │
│                                   ▼                                            │
│  【Application Layer】                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         MONJYU Facade                                    │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │  │
│  │  │ UnifiedGraphRAG │  │ProgressiveGraph │  │    HybridGraphRAG       │  │  │
│  │  │   Controller    │  │  RAG Controller │  │      Controller         │  │  │
│  │  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │  │
│  │           │                    │                        │               │  │
│  │           └────────────────────┼────────────────────────┘               │  │
│  │                                │                                         │  │
│  │                    ┌───────────┴───────────┐                            │  │
│  │                    │     Query Router      │                            │  │
│  │                    │ (AUTO/LAZY/GRAPH/VEC) │                            │  │
│  │                    └───────────────────────┘                            │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                   │                                            │
│                                   ▼                                            │
│  【Domain Layer】                                                               │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐ │
│  │   Index Domain    │  │   Query Domain    │  │    Citation Domain        │ │
│  │                   │  │                   │  │                           │ │
│  │  - DocLoader      │  │  - VectorSearch   │  │  - CitationNetwork        │ │
│  │  - PDFProcessor   │  │  - LazySearch     │  │  - Co-citation            │ │
│  │  - Chunker        │  │  - GlobalSearch   │  │  - BibliographicCoupling  │ │
│  │  - Embedder       │  │  - LocalSearch    │  │                           │ │
│  │  - EntityExtract  │  │  - HybridMerge    │  │                           │ │
│  │  - CommunityDetect│  │                   │  │                           │ │
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘ │
│                                   │                                            │
│                                   ▼                                            │
│  【Infrastructure Layer】                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LLM Client  │  │  Embedding  │  │ PDF Process │  │   External APIs     │  │
│  │             │  │   Client    │  │   Client    │  │                     │  │
│  │- AzureOpenAI│  │- AzureOpenAI│  │- AzureDocInt│  │- SemanticScholar    │  │
│  │- Ollama     │  │- Ollama     │  │- Unstructured│ │- CrossRef/OpenAlex  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                                                │
│  【Storage Layer】                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Index Storage│  │Vector Store │  │   Cache     │  │   Citation Graph    │  │
│  │             │  │             │  │             │  │                     │  │
│  │- Parquet    │  │- AzureAISrch│  │- Redis      │  │- NetworkX/Neo4j     │  │
│  │- AzureBlob  │  │- LanceDB    │  │- Local      │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Progressive GraphRAG レベルアーキテクチャ

### 2.1 レベル構造

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Progressive GraphRAG - Index Levels                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Level 4: Enhanced     ┌─────────────────────────────────────────────────────┐ │
│  (LLM: 💰💰💰💰💰)      │  Full GraphRAG + Pre-extracted Claims              │ │
│                        │  Components: Level 3 + claim_store                  │ │
│                        └─────────────────────────────────────────────────────┘ │
│                                           ▲                                    │
│  Level 3: Full         ┌─────────────────────────────────────────────────────┐ │
│  (LLM: 💰💰💰💰)        │  GraphRAG with Community Reports                    │ │
│                        │  Components: Level 2 + community_reports[]          │ │
│                        └─────────────────────────────────────────────────────┘ │
│                                           ▲                                    │
│  Level 2: Partial      ┌─────────────────────────────────────────────────────┐ │
│  (LLM: 💰💰💰)          │  GraphRAG with Entities & Relationships             │ │
│                        │  Components: Level 1 + entities[] + relationships[] │ │
│                        └─────────────────────────────────────────────────────┘ │
│                                           ▲                                    │
│  ═══════════════════════════════════════════════════════════════════════════  │
│                              【LLM Cost Boundary】                              │
│  ═══════════════════════════════════════════════════════════════════════════  │
│                                           ▲                                    │
│  Level 1: Lazy         ┌─────────────────────────────────────────────────────┐ │
│  (LLM: $0)             │  LazyGraphRAG (NLP-based)                           │ │
│                        │  Components: Level 0 + noun_graph + communities[]   │ │
│                        └─────────────────────────────────────────────────────┘ │
│                                           ▲                                    │
│  Level 0: Raw          ┌─────────────────────────────────────────────────────┐ │
│  (Embedding only)      │  Baseline RAG                                       │ │
│                        │  Components: documents[] + text_units[] + vectors[] │ │
│                        └─────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 レベル別データモデル

```python
# Level 0: Raw (Baseline RAG)
@dataclass
class Level0Index:
    documents: list[Document]
    text_units: list[TextUnit]
    embeddings: np.ndarray  # shape: (n_units, embedding_dim)
    metadata: IndexMetadata

# Level 1: Lazy (LazyGraphRAG)
@dataclass
class Level1Index(Level0Index):
    noun_phrases: list[NounPhrase]
    noun_graph: nx.Graph  # 名詞句の共起グラフ
    communities: list[Community]  # Leiden algorithm
    citation_network: CitationGraph  # 論文間引用

# Level 2: Partial (GraphRAG - Entities)
@dataclass
class Level2Index(Level1Index):
    entities: list[Entity]  # LLM extracted
    relationships: list[Relationship]  # LLM extracted
    entity_graph: nx.Graph

# Level 3: Full (GraphRAG - Community Reports)
@dataclass
class Level3Index(Level2Index):
    community_reports: list[CommunityReport]  # LLM generated

# Level 4: Enhanced (Pre-extracted Claims)
@dataclass
class Level4Index(Level3Index):
    claims: list[Claim]  # Pre-extracted for faster query
```

---

## 3. ドメインモデル（学術論文特化）

### 3.1 コアエンティティ

```python
@dataclass
class AcademicPaper:
    """学術論文ドキュメント"""
    id: str
    file_path: str
    
    # === 基本情報 ===
    title: str
    authors: list[Author]
    abstract: str
    
    # === 識別子 ===
    doi: str | None
    arxiv_id: str | None
    pmid: str | None
    
    # === 出版情報 ===
    publication_year: int | None
    venue: str | None
    venue_type: Literal["journal", "conference", "preprint"]
    
    # === 構造化コンテンツ ===
    sections: list[Section]  # IMRaD構造
    tables: list[Table]
    figures: list[Figure]
    equations: list[Equation]
    
    # === 引用 ===
    references: list[Reference]
    inline_citations: list[InlineCitation]
    
    # === メタデータ ===
    keywords: list[str]
    citation_count: int | None
    language: str

@dataclass
class Author:
    name: str
    affiliation: str | None
    orcid: str | None

@dataclass
class Section:
    heading: str
    level: int
    section_type: Literal["introduction", "related_work", "methods", 
                          "experiments", "results", "discussion", 
                          "conclusion", "other"]
    content: str
    subsections: list['Section']

@dataclass
class Reference:
    ref_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
```

### 3.2 学術エンティティタイプ

```python
class AcademicEntityType(Enum):
    """学術論文向けエンティティタイプ"""
    RESEARCHER = "researcher"      # 研究者
    ORGANIZATION = "organization"  # 機関・組織
    METHOD = "method"              # 手法・アルゴリズム
    MODEL = "model"                # モデル (GPT-4, BERT等)
    DATASET = "dataset"            # データセット
    METRIC = "metric"              # 評価指標
    TASK = "task"                  # タスク
    CONCEPT = "concept"            # 概念
    TOOL = "tool"                  # ツール・フレームワーク
    PAPER = "paper"                # 論文

@dataclass
class AcademicEntity:
    id: str
    name: str
    type: AcademicEntityType
    description: str
    aliases: list[str]
    source_text_units: list[str]
    
    # 学術特有
    first_mentioned_year: int | None
    citation_count: int | None  # この概念の引用頻度
```

### 3.3 引用ネットワーク

```python
@dataclass
class CitationGraph:
    """論文間引用ネットワーク"""
    papers: dict[str, AcademicPaper]  # paper_id -> Paper
    
    # エッジタイプ
    cites: list[tuple[str, str]]  # (citing_paper, cited_paper)
    
    def get_cited_by(self, paper_id: str) -> list[str]:
        """被引用論文を取得"""
        pass
    
    def get_co_citations(self, paper_id: str) -> list[tuple[str, float]]:
        """共引用論文を取得（同じ論文に引用される）"""
        pass
    
    def get_bibliographic_coupling(self, paper_id: str) -> list[tuple[str, float]]:
        """書誌結合論文を取得（同じ論文を引用する）"""
        pass
    
    def get_citation_path(self, from_id: str, to_id: str) -> list[str]:
        """引用パスを取得"""
        pass
```

---

## 4. クエリルーティングアーキテクチャ

### 4.1 Unified GraphRAG Query Router

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Unified GraphRAG Query Router                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input Query: "Transformerの注意機構について最新の研究動向を教えて"              │
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Query Classifier                                  │   │
│  │                                                                          │   │
│  │   Level 1: Rule-based (Keywords + Patterns)                             │   │
│  │   Level 2: ML Classifier (Lightweight)                                  │   │
│  │   Level 3: LLM Classifier (High accuracy)                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                   │                                             │
│                                   ▼                                             │
│  ┌───────────────┬───────────────┬───────────────┬───────────────────────┐    │
│  │   GRAPHRAG    │     LAZY      │    HYBRID     │       VECTOR          │    │
│  │               │               │               │                       │    │
│  │ "全体の傾向"  │ "〇〇について"│ "AとBの違い" │ "〇〇の値は？"        │    │
│  │ "主要テーマ"  │ "教えて"     │ "比較して"   │ "どこに書いてある？"   │    │
│  │               │               │               │                       │    │
│  │  → Global/    │  → Lazy      │  → Parallel   │  → Vector             │    │
│  │    Local      │    Search    │    + RRF      │    Search             │    │
│  │    Search     │              │               │                       │    │
│  └───────────────┴───────────────┴───────────────┴───────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 学術論文向けクエリパターン

```python
class AcademicQueryPattern(Enum):
    """学術論文向けクエリパターン"""
    
    # Survey系 → GRAPHRAG
    SURVEY_TREND = "survey_trend"        # 研究動向
    SURVEY_OVERVIEW = "survey_overview"  # 分野概要
    
    # 探索系 → LAZY
    METHOD_SEARCH = "method_search"      # 手法調査
    PRIOR_WORK = "prior_work"            # 先行研究
    IMPLEMENTATION = "implementation"    # 実装方法
    
    # 比較系 → HYBRID
    METHOD_COMPARISON = "method_comparison"  # 手法比較
    BENCHMARK_COMPARE = "benchmark_compare"  # ベンチマーク比較
    
    # ファクト系 → VECTOR
    SPECIFIC_VALUE = "specific_value"    # 具体的な値
    DATASET_INFO = "dataset_info"        # データセット情報
    CITATION_INFO = "citation_info"      # 引用情報

# ルーティングルール
ROUTING_RULES = {
    AcademicQueryPattern.SURVEY_TREND: SearchMode.GRAPHRAG,
    AcademicQueryPattern.SURVEY_OVERVIEW: SearchMode.GRAPHRAG,
    AcademicQueryPattern.METHOD_SEARCH: SearchMode.LAZY,
    AcademicQueryPattern.PRIOR_WORK: SearchMode.LAZY,
    AcademicQueryPattern.IMPLEMENTATION: SearchMode.LAZY,
    AcademicQueryPattern.METHOD_COMPARISON: SearchMode.HYBRID,
    AcademicQueryPattern.BENCHMARK_COMPARE: SearchMode.HYBRID,
    AcademicQueryPattern.SPECIFIC_VALUE: SearchMode.VECTOR,
    AcademicQueryPattern.DATASET_INFO: SearchMode.VECTOR,
    AcademicQueryPattern.CITATION_INFO: SearchMode.VECTOR,
}
```

---

## 5. シーケンス図

### 5.1 Unified GraphRAG クエリフロー

```
┌──────┐     ┌─────────┐     ┌────────┐     ┌──────────┐     ┌─────────┐
│ User │     │Unified  │     │ Query  │     │ Search   │     │   LLM   │
│      │     │Controller│    │ Router │     │ Engine   │     │         │
└──┬───┘     └────┬────┘     └───┬────┘     └────┬─────┘     └────┬────┘
   │              │              │               │                │
   │ search(query)│              │               │                │
   │─────────────▶│              │               │                │
   │              │ classify()   │               │                │
   │              │─────────────▶│               │                │
   │              │              │               │                │
   │              │   mode=LAZY  │               │                │
   │              │◀─────────────│               │                │
   │              │              │               │                │
   │              │     lazy_search(query)       │                │
   │              │─────────────────────────────▶│                │
   │              │              │               │                │
   │              │              │               │ expand_query() │
   │              │              │               │───────────────▶│
   │              │              │               │   subqueries   │
   │              │              │               │◀───────────────│
   │              │              │               │                │
   │              │              │               │ test_relevance()
   │              │              │               │───────────────▶│
   │              │              │               │    scores      │
   │              │              │               │◀───────────────│
   │              │              │               │                │
   │              │              │               │ extract_claims()
   │              │              │               │───────────────▶│
   │              │              │               │    claims      │
   │              │              │               │◀───────────────│
   │              │              │               │                │
   │              │     SearchResult             │                │
   │              │◀─────────────────────────────│                │
   │              │              │               │                │
   │              │ generate_response()          │                │
   │              │─────────────────────────────────────────────▶│
   │              │              │               │     response   │
   │              │◀─────────────────────────────────────────────│
   │              │              │               │                │
   │ SearchResult │              │               │                │
   │◀─────────────│              │               │                │
```

### 5.2 Progressive インデックス構築フロー

```
┌──────┐     ┌───────────┐     ┌────────┐     ┌─────────┐     ┌─────────┐
│ User │     │Progressive│     │ Index  │     │   NLP   │     │   LLM   │
│      │     │Controller │     │ Manager│     │ Engine  │     │         │
└──┬───┘     └─────┬─────┘     └───┬────┘     └────┬────┘     └────┬────┘
   │               │               │               │                │
   │ index(docs,   │               │               │                │
   │  level=1)     │               │               │                │
   │──────────────▶│               │               │                │
   │               │               │               │                │
   │               │ build_level_0()               │                │
   │               │──────────────▶│               │                │
   │               │               │ chunk()       │                │
   │               │               │──────────────▶│                │
   │               │               │ embed()       │                │
   │               │               │──────────────▶│                │
   │               │  Level0Index  │               │                │
   │               │◀──────────────│               │                │
   │               │               │               │                │
   │               │ build_level_1()               │                │
   │               │──────────────▶│               │                │
   │               │               │ extract_nouns │                │
   │               │               │──────────────▶│                │
   │               │               │ build_graph() │                │
   │               │               │──────────────▶│                │
   │               │               │ detect_communities()           │
   │               │               │──────────────▶│                │
   │               │  Level1Index  │               │                │
   │               │◀──────────────│               │                │
   │               │               │               │                │
   │ IndexResult   │               │               │                │
   │◀──────────────│               │               │                │
   │               │               │               │                │
   │ [Later: upgrade to level 2]   │               │                │
   │               │               │               │                │
   │ upgrade(level=2)              │               │                │
   │──────────────▶│               │               │                │
   │               │ build_level_2()               │                │
   │               │──────────────▶│               │                │
   │               │               │ extract_entities()             │
   │               │               │─────────────────────────────▶│
   │               │               │    entities[]                 │
   │               │               │◀─────────────────────────────│
   │               │  Level2Index  │               │                │
   │               │◀──────────────│               │                │
```

---

## 6. 本番環境デプロイアーキテクチャ

### 6.1 Azure スケールアウト構成

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Azure Production Architecture                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Azure Front Door (Global Load Balancer)               │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Azure API Management                                  │   │
│  │    - Authentication (Entra ID / API Key)                                │   │
│  │    - Rate Limiting (100 req/min/user)                                   │   │
│  │    - Request/Response Logging                                           │   │
│  │    - API Versioning                                                     │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                             │
│              ┌────────────────────┼────────────────────┐                       │
│              │                    │                    │                       │
│              ▼                    ▼                    ▼                       │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │               Azure Container Apps (Auto-scaling 1-20)                 │    │
│  │                                                                        │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │   │ MCP Server  │  │ MCP Server  │  │ MCP Server  │  │ MCP Server  │ │    │
│  │   │  Replica 1  │  │  Replica 2  │  │  Replica 3  │  │  Replica N  │ │    │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │          │                │                │                │        │    │
│  │          └────────────────┴────────────────┴────────────────┘        │    │
│  │                                    │                                  │    │
│  │                          ┌─────────┴─────────┐                       │    │
│  │                          │  Shared Services  │                       │    │
│  │                          └───────────────────┘                       │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                   │                                             │
│         ┌─────────────────────────┼─────────────────────────┐                  │
│         │                         │                         │                  │
│         ▼                         ▼                         ▼                  │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │ Azure Cache     │  │  Azure OpenAI       │  │  Azure Document         │   │
│  │ for Redis       │  │                     │  │  Intelligence           │   │
│  │                 │  │  - GPT-4o           │  │                         │   │
│  │ - Session       │  │  - text-embedding-  │  │  - PDF Layout           │   │
│  │ - Query Cache   │  │    3-large          │  │  - Table/Figure         │   │
│  │ - Result Cache  │  │                     │  │  - Formula              │   │
│  └─────────────────┘  └─────────────────────┘  └─────────────────────────┘   │
│                                                                                 │
│         ┌─────────────────────────┬─────────────────────────┐                  │
│         │                         │                         │                  │
│         ▼                         ▼                         ▼                  │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │ Azure AI Search │  │  Azure Blob Storage │  │  Azure Monitor          │   │
│  │                 │  │                     │  │                         │   │
│  │ - Vector Index  │  │  - Paper PDFs       │  │  - Application Insights │   │
│  │ - Hybrid Search │  │  - Index Data       │  │  - Log Analytics        │   │
│  │ - Semantic      │  │  - Parquet Files    │  │  - Alerts               │   │
│  │   Ranking       │  │                     │  │  - Dashboard            │   │
│  └─────────────────┘  └─────────────────────┘  └─────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 ローカル開発環境

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Local Development Architecture                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          Developer Machine                               │   │
│  │                                                                          │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐│   │
│  │   │    CLI      │  │ Python API  │  │      VS Code + Claude          ││   │
│  │   │   (monjyu)  │  │   (pytest)  │  │      (MCP Client)              ││   │
│  │   └──────┬──────┘  └──────┬──────┘  └────────────────┬────────────────┘│   │
│  │          │                │                          │                 │   │
│  │          └────────────────┴──────────────────────────┘                 │   │
│  │                                   │                                     │   │
│  │                                   ▼                                     │   │
│  │   ┌───────────────────────────────────────────────────────────────┐    │   │
│  │   │                    MONJYU (Local)                              │    │   │
│  │   │                                                                │    │   │
│  │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │   │
│  │   │   │   Indexer   │  │   Search    │  │    MCP Server       │  │    │   │
│  │   │   │             │  │   Engine    │  │    (stdio)          │  │    │   │
│  │   │   └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │    │   │
│  │   └──────────┼────────────────┼───────────────────┼──────────────┘    │   │
│  │              │                │                    │                   │   │
│  │              ▼                ▼                    ▼                   │   │
│  │   ┌─────────────────────────────────────────────────────────────┐     │   │
│  │   │                    Infrastructure                            │     │   │
│  │   │                                                              │     │   │
│  │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│     │   │
│  │   │   │   LanceDB   │  │  Local File │  │    unstructured     ││     │   │
│  │   │   │  (Vector)   │  │  (Parquet)  │  │    (PDF Parse)      ││     │   │
│  │   │   └─────────────┘  └─────────────┘  └─────────────────────┘│     │   │
│  │   └──────────────────────────────────────────────────────────────┘     │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Windows Host (192.168.224.1)                          │   │
│  │                                                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                        Ollama                                    │   │   │
│  │   │                                                                  │   │   │
│  │   │   - llama3.2 (LLM)                                              │   │   │
│  │   │   - nomic-embed-text (Embedding)                                │   │   │
│  │   │   - Port: 11434                                                 │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. ディレクトリ構造（v3.1 - 実装反映）

```
MONJYU/
├── __init__.py
├── AGENTS.md                       # MUSUBI SDD エージェント設定
├── README.md
├── pyproject.toml
│
├── monjyu/                         # 🔵 メインパッケージ
│   ├── __init__.py
│   │
│   ├── api/                        # 🎯 MONJYU Facade API
│   │   ├── __init__.py
│   │   ├── base.py                 # ベース定義
│   │   ├── config.py               # API設定
│   │   ├── factory.py              # コンポーネントファクトリー
│   │   ├── monjyu.py               # メインFacade (REQ-API-001)
│   │   └── state.py                # 状態管理
│   │
│   ├── cli/                        # 🖥️ CLI (REQ-API-002)
│   │   ├── __init__.py
│   │   ├── main.py                 # Typer app
│   │   └── commands/               # サブコマンド
│   │       ├── config_cmd.py
│   │       ├── document.py
│   │       └── citation.py
│   │
│   ├── controller/                 # 🎮 Controller Layer
│   │   ├── __init__.py
│   │   ├── budget.py               # BudgetController
│   │   ├── unified/                # Unified GraphRAG (REQ-ARC-001)
│   │   │   └── controller.py
│   │   ├── progressive/            # Progressive GraphRAG (REQ-ARC-002)
│   │   │   ├── controller.py
│   │   │   └── types.py
│   │   └── hybrid/                 # Hybrid GraphRAG (REQ-ARC-003)
│   │       ├── controller.py
│   │       └── types.py
│   │
│   ├── document/                   # 📄 ドキュメント処理
│   │   ├── __init__.py
│   │   ├── loader.py               # FileLoader (REQ-IDX-001)
│   │   ├── parser.py               # DocumentParser
│   │   ├── chunker.py              # TextChunker (REQ-IDX-002)
│   │   ├── pipeline.py             # PreprocessingPipeline (REQ-IDX-001c)
│   │   ├── models.py               # AcademicPaperDocument, TextUnit
│   │   └── pdf/                    # PDF処理 (REQ-IDX-001a)
│   │       └── processor.py
│   │
│   ├── embedding/                  # 🔢 エンベディング (REQ-IDX-003)
│   │   ├── __init__.py
│   │   ├── base.py                 # Protocol定義
│   │   ├── azure_openai.py         # Azure OpenAI
│   │   └── ollama.py               # Ollama
│   │
│   ├── index/                      # 📊 インデックス構築
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── manager.py
│   │   ├── lancedb.py              # LanceDB Vector Index
│   │   ├── azure_search.py         # Azure AI Search
│   │   ├── level0/                 # Level 0: Raw (REQ-IDX-009)
│   │   │   └── builder.py
│   │   ├── level1/                 # Level 1: Lazy
│   │   │   └── builder.py
│   │   ├── entity_extractor/       # エンティティ抽出 (REQ-IDX-005)
│   │   │   ├── protocol.py
│   │   │   └── llm_extractor.py
│   │   ├── relationship_extractor/ # 関係性抽出 (REQ-IDX-006)
│   │   │   ├── protocol.py
│   │   │   └── llm_extractor.py
│   │   ├── community_detector/     # コミュニティ検出 (REQ-IDX-007)
│   │   │   └── leiden.py
│   │   └── community_report_generator/ # レポート生成 (REQ-IDX-008)
│   │       └── generator.py
│   │
│   ├── query/                      # 🔍 クエリ処理
│   │   ├── vector_search/          # Vector Search (REQ-QRY-001)
│   │   │   └── in_memory.py
│   │   ├── global_search/          # Global Search (REQ-QRY-002)
│   │   │   └── search.py
│   │   ├── local_search/           # Local Search (REQ-QRY-003)
│   │   │   └── search.py
│   │   └── router/                 # Query Router (REQ-QRY-006)
│   │       └── router.py
│   │
│   ├── lazy/                       # 🦥 LazySearch (REQ-QRY-004)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── engine.py
│   │   ├── claim_extractor.py
│   │   └── community_searcher.py
│   │
│   ├── citation/                   # 📚 引用ネットワーク (REQ-IDX-005a)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── builder.py
│   │   ├── analyzer.py
│   │   ├── resolver.py
│   │   ├── manager.py
│   │   └── metrics.py
│   │
│   ├── graph/                      # 🕸️ グラフ操作
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── noun_phrase_graph.py
│   │   └── community_detector.py
│   │
│   ├── nlp/                        # 🔤 NLP処理 (REQ-IDX-004)
│   │   ├── __init__.py
│   │   └── rake_extractor.py
│   │
│   ├── storage/                    # 💾 ストレージ
│   │   ├── __init__.py
│   │   ├── parquet.py              # Parquet Storage (REQ-STG-001)
│   │   └── cache.py                # Cache Manager (REQ-STG-003)
│   │
│   ├── mcp_server/                 # 🤖 MCP Server (REQ-API-004)
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── handlers.py
│   │   └── tools.py
│   │
│   ├── errors/                     # ⚠️ エラーハンドリング
│   │   └── __init__.py
│   │
│   ├── observability/              # 📈 可観測性
│   │   └── __init__.py
│   │
│   └── search/                     # 🔎 検索抽象化
│       └── __init__.py
│
├── lazy_search/                    # 📦 Legacy LazySearch (移行中)
│   ├── __init__.py
│   ├── search.py
│   ├── query_expander.py
│   ├── relevance_tester.py
│   ├── claim_extractor.py
│   ├── iterative_deepener.py
│   ├── context.py
│   ├── state.py
│   └── core/
│
├── config/                         # ⚙️ 設定
│   └── lazy_search_config.py
│
├── prompts/                        # 💬 プロンプト
│   ├── __init__.py
│   └── lazy_search_system_prompt.py
│
├── tests/                          # 🧪 テスト (1268 tests)
│   ├── __init__.py
│   ├── conftest.py
│   ├── mock_provider.py
│   ├── unit/                       # ユニットテスト (1086)
│   ├── integration/                # 統合テスト (165)
│   ├── e2e/                        # E2Eテスト (17)
│   └── benchmarks/
│
├── specs/                          # 📋 仕様書
│   ├── 01_requirements_v3.md       # ✅ Approved
│   ├── 02_architecture_v3.md       # ✅ Updated
│   ├── 03_components_v3.md
│   └── 04_api_v3.md
│
├── steering/                       # 🧭 Steering Files (MUSUBI)
│   ├── product.ja.md
│   ├── structure.ja.md
│   ├── tech.ja.md
│   ├── project.yml
│   └── rules/
│
├── docs/                           # 📖 ドキュメント
│   └── *.md
│
├── storage/                        # 📁 データ出力
│   └── ...
│
└── output/                         # 📤 実行出力
    └── monjyu_state.json
```

---

## 8. 技術選定 (ADR)

### ADR-001: ベクトルストア選択

| 観点 | LanceDB | Azure AI Search | FAISS |
|------|---------|-----------------|-------|
| ローカル開発 | ◎ | △ | ○ |
| 本番スケール | ○ | ◎ | △ |
| ハイブリッド検索 | ○ | ◎ | △ |
| コスト | ◎ | △ | ◎ |

**決定**: 
- ローカル: **LanceDB**
- 本番: **Azure AI Search**

### ADR-002: PDF処理選択

| 観点 | Azure Document Intelligence | unstructured |
|------|---------------------------|--------------|
| 精度 | ◎ | ○ |
| 学術論文対応 | ◎ (数式・表) | ○ |
| コスト | △ | ◎ |
| ローカル実行 | × | ◎ |

**決定**:
- ローカル: **unstructured**
- 本番: **Azure Document Intelligence**

### ADR-003: LLMプロバイダー選択

| 観点 | Azure OpenAI | Ollama |
|------|-------------|--------|
| 品質 | ◎ | ○ |
| コスト | △ | ◎ |
| レイテンシ | ○ | ○ |
| ローカル実行 | × | ◎ |

**決定**:
- ローカル: **Ollama** (192.168.224.1:11434)
- 本番: **Azure OpenAI**

### ADR-004: グラフDB選択

| 観点 | NetworkX | Neo4j |
|------|----------|-------|
| 学習コスト | ◎ | △ |
| スケール | △ | ◎ |
| クエリ性能 | ○ | ◎ |
| 依存 | なし | あり |

**決定**: Phase 1-2 は **NetworkX**、スケール必要時に Neo4j 検討

---

## 9. 次のステップ

1. **Level 2 (Eager) 実装** - Entity/Relationship Extraction 自動化
2. **MCP Server 完成** - REQ-API-004 対応
3. **Hybrid Search** - REQ-QRY-005 RRF 実装

---

**文書履歴**:

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-12-24 | 初版（LazyGraphRAGベース） |
| 3.0.0 | 2025-12-24 | 要件v3.0対応、学術論文特化、スケールアウト構成追加 |
| 3.1.0 | 2025-12-27 | 実装反映 - ディレクトリ構造を実際のコード構成に更新、ステータスApproved |

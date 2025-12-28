# FEAT-004: Vector Search (Baseline RAG)

**フィーチャーID**: FEAT-004  
**名称**: ベクトル検索 (Baseline RAG Query)  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

Level 0 インデックスを使用したベクトル類似検索。LazyGraphRAGの最初のレベルとして、高速な検索結果を提供する。

### 1.1 スコープ

```
Query → Embedding → Vector Search → Top-K TextUnits → LLM Synthesis → Answer
```

- **入力**: 自然言語クエリ
- **処理**: クエリ埋め込み、ベクトル検索、LLM回答生成
- **出力**: 回答テキスト + ソース引用
- **レイテンシ目標**: < 5秒

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-QRY-VEC-001 | ベクトル類似検索 | P0 |
| FR-QRY-VEC-002 | Top-K結果取得 | P0 |
| FR-QRY-VEC-003 | 類似度閾値フィルタリング | P0 |
| FR-QRY-VEC-004 | ハイブリッド検索（キーワード+ベクトル） | P1 |
| FR-QRY-RES-001 | LLM回答生成 | P0 |
| FR-QRY-RES-002 | ソース引用 | P0 |

### 1.3 依存関係

- **依存**: FEAT-002 (Index Level 0)
- **被依存**: FEAT-007 (Python API), FEAT-008 (CLI), FEAT-009 (MCP Server)

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Vector Search Engine                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐│
│  │ QueryEncoder│    │ VectorSearcher  │    │ AnswerSynthesizer    ││
│  │             │    │                 │    │                      ││
│  │ - encode()  │───▶│ - search()      │───▶│ - synthesize()       ││
│  └─────────────┘    │ - hybrid()      │    │ - with_citations()   ││
│                     └─────────────────┘    └──────────────────────┘│
│         │                   │                        │              │
│         ▼                   ▼                        ▼              │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐│
│  │ Embedding   │    │ SearchResults   │    │ SearchResponse       ││
│  │ Client      │    │ (Top-K)         │    │ (Answer+Citations)   ││
│  └─────────────┘    └─────────────────┘    └──────────────────────┘│
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Vector Index Backend                      │   │
│  │  ┌─────────────────────┐    ┌─────────────────────────────┐ │   │
│  │  │ LanceDB (Local)     │    │ Azure AI Search (Production)│ │   │
│  │  └─────────────────────┘    └─────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 シーケンス図

```
User        QueryEncoder    VectorSearcher    AnswerSynthesizer    LLM
  │              │                │                   │              │
  │──query──────▶│                │                   │              │
  │              │──embed()──────▶│                   │              │
  │              │◀──vector───────│                   │              │
  │              │                │                   │              │
  │              │────search(vector)────▶│            │              │
  │              │                │◀─TextUnits[]─────│              │
  │              │                │                   │              │
  │              │                │───synthesize()───▶│──prompt────▶│
  │              │                │                   │◀──answer────│
  │              │                │                   │              │
  │◀──SearchResponse──────────────│───────────────────│              │
```

### 2.3 クラス図

```python
from dataclasses import dataclass, field
from typing import Protocol
from enum import Enum

# === Enums ===

class SearchMode(Enum):
    """検索モード"""
    VECTOR = "vector"           # ベクトルのみ
    KEYWORD = "keyword"         # キーワードのみ
    HYBRID = "hybrid"           # ハイブリッド

# === Protocols ===

class QueryEncoderProtocol(Protocol):
    """クエリエンコーダープロトコル"""
    def encode(self, query: str) -> list[float]: ...
    def encode_batch(self, queries: list[str]) -> list[list[float]]: ...

class VectorSearcherProtocol(Protocol):
    """ベクトル検索プロトコル"""
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> "SearchResults": ...
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5  # vector weight
    ) -> "SearchResults": ...

class AnswerSynthesizerProtocol(Protocol):
    """回答合成プロトコル"""
    def synthesize(
        self,
        query: str,
        context: list["TextUnit"],
        system_prompt: str | None = None
    ) -> "SynthesizedAnswer": ...

# === Data Classes ===

@dataclass
class SearchHit:
    """検索ヒット"""
    text_unit_id: str
    document_id: str
    text: str
    score: float  # 類似度スコア (0-1)
    
    # メタデータ
    chunk_index: int = 0
    document_title: str = ""
    
    # ハイブリッド検索用
    vector_score: float = 0.0
    keyword_score: float = 0.0

@dataclass
class SearchResults:
    """検索結果"""
    hits: list[SearchHit]
    total_count: int
    query_vector: list[float] = field(default_factory=list)
    search_time_ms: float = 0.0
    
    @property
    def texts(self) -> list[str]:
        return [h.text for h in self.hits]
    
    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

@dataclass
class Citation:
    """引用"""
    text_unit_id: str
    document_id: str
    document_title: str
    text_snippet: str  # 引用箇所
    relevance_score: float

@dataclass
class SynthesizedAnswer:
    """合成された回答"""
    answer: str
    citations: list[Citation]
    confidence: float = 0.0
    
    # メタデータ
    tokens_used: int = 0
    model: str = ""

@dataclass
class SearchResponse:
    """検索レスポンス"""
    query: str
    answer: SynthesizedAnswer
    search_results: SearchResults
    
    # パフォーマンス
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0
    
    # 検索パラメータ
    mode: SearchMode = SearchMode.VECTOR
    top_k: int = 10
```

---

## 3. 詳細設計

### 3.1 QueryEncoder

```python
from typing import Protocol

class QueryEncoder:
    """クエリエンコーダー"""
    
    def __init__(self, embedding_client: "EmbeddingClientProtocol"):
        self.embedding_client = embedding_client
    
    def encode(self, query: str) -> list[float]:
        """クエリを埋め込みベクトルに変換"""
        return self.embedding_client.embed(query)
    
    def encode_batch(self, queries: list[str]) -> list[list[float]]:
        """複数クエリを一括変換"""
        return self.embedding_client.embed_batch(queries)


class QueryExpander:
    """クエリ拡張"""
    
    def __init__(self, llm_client: "LLMClientProtocol"):
        self.llm_client = llm_client
    
    def expand(self, query: str, num_expansions: int = 3) -> list[str]:
        """クエリを複数の関連クエリに拡張"""
        prompt = f"""
        以下のクエリに対して、{num_expansions}個の異なる言い回しや関連クエリを生成してください。
        各クエリは改行で区切ってください。
        
        クエリ: {query}
        """
        
        response = self.llm_client.generate(prompt)
        expansions = response.strip().split("\n")
        
        return [query] + expansions[:num_expansions]
```

### 3.2 VectorSearcher

```python
import lancedb
from typing import Optional

class LanceDBVectorSearcher:
    """LanceDB ベクトル検索"""
    
    def __init__(
        self,
        db_path: str = "./output/index/level_0/vector_index",
        table_name: str = "embeddings"
    ):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> SearchResults:
        """ベクトル検索を実行"""
        import time
        start = time.time()
        
        # LanceDB検索
        results = (
            self.table
            .search(query_vector)
            .limit(top_k)
            .to_list()
        )
        
        # SearchHitに変換
        hits = []
        for r in results:
            score = 1.0 - r.get("_distance", 0.0)  # cosine distance → similarity
            
            if score >= threshold:
                hits.append(SearchHit(
                    text_unit_id=r["text_unit_id"],
                    document_id=r["document_id"],
                    text=r["text"],
                    score=score,
                    chunk_index=r.get("chunk_index", 0),
                    document_title=r.get("document_title", ""),
                    vector_score=score
                ))
        
        elapsed = (time.time() - start) * 1000
        
        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed
        )
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5
    ) -> SearchResults:
        """ハイブリッド検索（ベクトル + BM25）"""
        import time
        start = time.time()
        
        # ベクトル検索
        vector_results = (
            self.table
            .search(query_vector)
            .limit(top_k * 2)
            .to_list()
        )
        
        # FTS検索（LanceDBのフルテキスト検索）
        fts_results = (
            self.table
            .search(query_text, query_type="fts")
            .limit(top_k * 2)
            .to_list()
        )
        
        # スコア統合
        hit_map = {}
        
        for r in vector_results:
            text_unit_id = r["text_unit_id"]
            vector_score = 1.0 - r.get("_distance", 0.0)
            
            if text_unit_id not in hit_map:
                hit_map[text_unit_id] = {
                    "data": r,
                    "vector_score": vector_score,
                    "keyword_score": 0.0
                }
            else:
                hit_map[text_unit_id]["vector_score"] = vector_score
        
        for r in fts_results:
            text_unit_id = r["text_unit_id"]
            keyword_score = r.get("_score", 0.0)
            
            if text_unit_id not in hit_map:
                hit_map[text_unit_id] = {
                    "data": r,
                    "vector_score": 0.0,
                    "keyword_score": keyword_score
                }
            else:
                hit_map[text_unit_id]["keyword_score"] = keyword_score
        
        # Reciprocal Rank Fusion (RRF) スコア計算
        hits = []
        for text_unit_id, data in hit_map.items():
            rrf_score = (
                alpha * data["vector_score"] +
                (1 - alpha) * data["keyword_score"]
            )
            
            r = data["data"]
            hits.append(SearchHit(
                text_unit_id=text_unit_id,
                document_id=r["document_id"],
                text=r["text"],
                score=rrf_score,
                chunk_index=r.get("chunk_index", 0),
                document_title=r.get("document_title", ""),
                vector_score=data["vector_score"],
                keyword_score=data["keyword_score"]
            ))
        
        # スコアでソートしてTop-K
        hits.sort(key=lambda x: x.score, reverse=True)
        hits = hits[:top_k]
        
        elapsed = (time.time() - start) * 1000
        
        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed
        )


class AzureAISearchVectorSearcher:
    """Azure AI Search ベクトル検索"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str
    ):
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> SearchResults:
        """ベクトル検索を実行"""
        import time
        from azure.search.documents.models import VectorizedQuery
        
        start = time.time()
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )
        
        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["text_unit_id", "document_id", "text", "document_title", "chunk_index"]
        )
        
        hits = []
        for r in results:
            score = r["@search.score"]
            if score >= threshold:
                hits.append(SearchHit(
                    text_unit_id=r["text_unit_id"],
                    document_id=r["document_id"],
                    text=r["text"],
                    score=score,
                    chunk_index=r.get("chunk_index", 0),
                    document_title=r.get("document_title", ""),
                    vector_score=score
                ))
        
        elapsed = (time.time() - start) * 1000
        
        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed
        )
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5
    ) -> SearchResults:
        """ハイブリッド検索（Semantic Ranker使用）"""
        import time
        from azure.search.documents.models import VectorizedQuery
        
        start = time.time()
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields="embedding"
        )
        
        results = self.client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            select=["text_unit_id", "document_id", "text", "document_title", "chunk_index"],
            query_type="semantic",
            semantic_configuration_name="monjyu-semantic-config"
        )
        
        hits = []
        for r in results:
            score = r.get("@search.reranker_score", r["@search.score"])
            hits.append(SearchHit(
                text_unit_id=r["text_unit_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=score,
                chunk_index=r.get("chunk_index", 0),
                document_title=r.get("document_title", "")
            ))
        
        hits = hits[:top_k]
        elapsed = (time.time() - start) * 1000
        
        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed
        )
```

### 3.3 AnswerSynthesizer

```python
class AnswerSynthesizer:
    """回答合成"""
    
    DEFAULT_SYSTEM_PROMPT = """
あなたは学術論文の専門家です。
与えられたコンテキスト情報に基づいて、ユーザーの質問に正確かつ簡潔に回答してください。

ルール:
1. コンテキストに含まれる情報のみを使用して回答してください
2. 情報が不十分な場合は、その旨を明示してください
3. 回答には必ず引用元（Citation）を含めてください
4. 学術的な正確性を最優先してください
"""
    
    def __init__(self, llm_client: "LLMClientProtocol"):
        self.llm_client = llm_client
    
    def synthesize(
        self,
        query: str,
        context: list[SearchHit],
        system_prompt: str | None = None
    ) -> SynthesizedAnswer:
        """コンテキストから回答を合成"""
        
        # コンテキスト構築
        context_text = self._build_context(context)
        
        # プロンプト構築
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        user_prompt = f"""
## コンテキスト
{context_text}

## 質問
{query}

## 回答形式
回答を記述した後、使用した引用元を [1], [2] のように示してください。
"""
        
        # LLM呼び出し
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system
        )
        
        # 引用抽出
        answer_text, citations = self._extract_citations(response, context)
        
        return SynthesizedAnswer(
            answer=answer_text,
            citations=citations,
            model=self.llm_client.model_name
        )
    
    def _build_context(self, hits: list[SearchHit]) -> str:
        """コンテキストテキストを構築"""
        parts = []
        for i, hit in enumerate(hits):
            parts.append(f"[{i+1}] {hit.document_title}")
            parts.append(f"Score: {hit.score:.3f}")
            parts.append(hit.text)
            parts.append("")
        return "\n".join(parts)
    
    def _extract_citations(
        self,
        response: str,
        context: list[SearchHit]
    ) -> tuple[str, list[Citation]]:
        """回答から引用を抽出"""
        import re
        
        # [1], [2] などのパターンを検出
        citation_pattern = r'\[(\d+)\]'
        cited_indices = set(int(m) for m in re.findall(citation_pattern, response))
        
        citations = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(context):
                hit = context[idx - 1]
                citations.append(Citation(
                    text_unit_id=hit.text_unit_id,
                    document_id=hit.document_id,
                    document_title=hit.document_title,
                    text_snippet=hit.text[:200] + "...",
                    relevance_score=hit.score
                ))
        
        return response, citations
```

### 3.4 VectorSearchEngine (Facade)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VectorSearchConfig:
    """検索設定"""
    top_k: int = 10
    threshold: float = 0.0
    mode: SearchMode = SearchMode.VECTOR
    hybrid_alpha: float = 0.5
    expand_query: bool = False
    num_expansions: int = 3


class VectorSearchEngine:
    """ベクトル検索エンジン"""
    
    def __init__(
        self,
        embedding_client: "EmbeddingClientProtocol",
        vector_searcher: VectorSearcherProtocol,
        llm_client: "LLMClientProtocol",
        config: VectorSearchConfig | None = None
    ):
        self.encoder = QueryEncoder(embedding_client)
        self.searcher = vector_searcher
        self.synthesizer = AnswerSynthesizer(llm_client)
        self.expander = QueryExpander(llm_client)
        self.config = config or VectorSearchConfig()
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: SearchMode | None = None,
        synthesize: bool = True
    ) -> SearchResponse:
        """検索を実行"""
        import time
        total_start = time.time()
        
        top_k = top_k or self.config.top_k
        mode = mode or self.config.mode
        
        # 1. クエリエンコード
        query_vector = self.encoder.encode(query)
        
        # 2. 検索実行
        search_start = time.time()
        
        if mode == SearchMode.HYBRID:
            search_results = self.searcher.hybrid_search(
                query_text=query,
                query_vector=query_vector,
                top_k=top_k,
                alpha=self.config.hybrid_alpha
            )
        else:
            search_results = self.searcher.search(
                query_vector=query_vector,
                top_k=top_k,
                threshold=self.config.threshold
            )
        
        search_time = (time.time() - search_start) * 1000
        
        # 3. 回答合成（オプション）
        synthesis_start = time.time()
        
        if synthesize:
            answer = self.synthesizer.synthesize(query, search_results.hits)
        else:
            answer = SynthesizedAnswer(answer="", citations=[])
        
        synthesis_time = (time.time() - synthesis_start) * 1000
        total_time = (time.time() - total_start) * 1000
        
        return SearchResponse(
            query=query,
            answer=answer,
            search_results=search_results,
            total_time_ms=total_time,
            search_time_ms=search_time,
            synthesis_time_ms=synthesis_time,
            mode=mode,
            top_k=top_k
        )
    
    def search_expanded(
        self,
        query: str,
        num_expansions: int = 3,
        top_k: int | None = None
    ) -> SearchResponse:
        """クエリ拡張して検索"""
        top_k = top_k or self.config.top_k
        
        # クエリ拡張
        expanded_queries = self.expander.expand(query, num_expansions)
        
        # 各クエリで検索
        all_hits = []
        for q in expanded_queries:
            query_vector = self.encoder.encode(q)
            results = self.searcher.search(
                query_vector=query_vector,
                top_k=top_k
            )
            all_hits.extend(results.hits)
        
        # 重複除去 & リランキング
        seen = set()
        unique_hits = []
        for hit in sorted(all_hits, key=lambda x: x.score, reverse=True):
            if hit.text_unit_id not in seen:
                seen.add(hit.text_unit_id)
                unique_hits.append(hit)
        
        unique_hits = unique_hits[:top_k]
        
        # 回答合成
        answer = self.synthesizer.synthesize(query, unique_hits)
        
        return SearchResponse(
            query=query,
            answer=answer,
            search_results=SearchResults(
                hits=unique_hits,
                total_count=len(unique_hits)
            ),
            mode=SearchMode.VECTOR,
            top_k=top_k
        )
```

---

## 4. 設定

```yaml
# config/vector_search.yaml

vector_search:
  # 検索設定
  search:
    top_k: 10
    threshold: 0.0
    mode: hybrid  # vector | keyword | hybrid
    hybrid_alpha: 0.5  # 0=keyword only, 1=vector only
  
  # クエリ拡張
  query_expansion:
    enabled: false
    num_expansions: 3
  
  # ローカル環境（LanceDB）
  local:
    db_path: ./output/index/level_0/vector_index
    table_name: embeddings
  
  # 本番環境（Azure AI Search）
  azure:
    endpoint: ${AZURE_SEARCH_ENDPOINT}
    api_key: ${AZURE_SEARCH_API_KEY}
    index_name: monjyu-index
    semantic_config: monjyu-semantic-config

# LLM設定（回答合成用）
llm:
  local:
    base_url: http://192.168.224.1:11434
    model: llama3:8b-instruct-q4_K_M
    temperature: 0.3
  
  azure:
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}
    deployment: gpt-4o-mini
    temperature: 0.3
```

---

## 5. テスト計画

### 5.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_query_encode | QueryEncoder.encode | 768次元ベクトルを返す |
| test_vector_search | VectorSearcher.search | Top-K結果を返す |
| test_hybrid_search | VectorSearcher.hybrid_search | 統合スコアで返す |
| test_synthesize | AnswerSynthesizer.synthesize | 回答と引用を返す |
| test_citation_extraction | AnswerSynthesizer._extract_citations | 引用を正しく抽出 |

### 5.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_e2e_search | VectorSearchEngine.search | 完全なSearchResponseを返す |
| test_expanded_search | VectorSearchEngine.search_expanded | 拡張クエリで検索 |
| test_lancedb_backend | LanceDBVectorSearcher | ローカル検索成功 |
| test_azure_backend | AzureAISearchVectorSearcher | Azure検索成功 |

### 5.3 パフォーマンステスト

| テストケース | 条件 | 期待結果 |
|-------------|------|---------|
| test_search_latency | 単一クエリ | < 500ms (検索のみ) |
| test_e2e_latency | 検索+合成 | < 5sec |
| test_throughput | 並列10クエリ | > 2 queries/sec |

---

## 6. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-004-01 | QueryEncoder 実装 | 1h | - |
| TASK-004-02 | LanceDBVectorSearcher 実装 | 3h | TASK-004-01 |
| TASK-004-03 | AzureAISearchVectorSearcher 実装 | 3h | TASK-004-01 |
| TASK-004-04 | AnswerSynthesizer 実装 | 2h | - |
| TASK-004-05 | QueryExpander 実装 | 1h | - |
| TASK-004-06 | VectorSearchEngine 実装 | 2h | TASK-004-01~05 |
| TASK-004-07 | 単体テスト作成 | 2h | TASK-004-01~06 |
| TASK-004-08 | 統合テスト作成 | 2h | TASK-004-07 |
| **合計** | | **16h** | |

---

## 7. 受入基準

- [ ] ベクトル検索でTop-K結果を取得できる
- [ ] ハイブリッド検索（ベクトル+キーワード）が動作する
- [ ] 類似度閾値でフィルタリングできる
- [ ] LLMで回答を合成し、引用を付与できる
- [ ] ローカル（LanceDB）と本番（Azure AI Search）の両方で動作する
- [ ] 検索レイテンシ < 500ms（合成なし）
- [ ] E2Eレイテンシ < 5秒（合成あり）

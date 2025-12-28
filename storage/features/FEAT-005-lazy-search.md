# FEAT-005: Lazy Search (LazyGraphRAG Query)

**フィーチャーID**: FEAT-005  
**名称**: Lazy Search (LazyGraphRAG Query)  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

LazyGraphRAGの中核となるクエリエンジン。Level 0/1インデックスを使用したコスト効率の良い検索を提供し、必要に応じてLevel 2以上に動的に深化する。

### 1.1 スコープ

```
Query → Relevance Test → Vector Search → (NLP Graph Search) → Iterative Deepening → Answer
```

- **入力**: 自然言語クエリ、検索深度設定
- **処理**: 
  - Level 0: ベクトル検索（FEAT-004）
  - Level 1: NLPグラフ＋コミュニティ検索
  - 動的深化: 関連性テストに基づく段階的展開
- **出力**: 回答テキスト + ソース引用 + 検索コンテキスト
- **特徴**: **遅延評価**による最小LLMコスト

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-QRY-LZY-001 | Lazy Search基本クエリ | P0 |
| FR-QRY-LZY-002 | 関連性テスト | P0 |
| FR-QRY-LZY-003 | クレーム抽出 | P0 |
| FR-QRY-LZY-004 | 動的深化 | P0 |
| FR-QRY-LZY-005 | Best-First検索 | P0 |
| FR-QRY-LZY-006 | コミュニティ検索 | P0 |
| FR-QRY-LZY-007 | DRIFT的な会話コンテキスト | P1 |

### 1.3 依存関係

- **依存**: FEAT-002 (Index Level 0), FEAT-003 (Index Level 1), FEAT-004 (Vector Search)
- **被依存**: FEAT-007 (Python API), FEAT-008 (CLI), FEAT-009 (MCP Server)

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Lazy Search Engine                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐  │
│  │ RelevanceTester │    │ ClaimExtractor │    │ IterativeDeepen │  │
│  │                 │    │                │    │                 │  │
│  │ - test_batch()  │    │ - extract()    │    │ - deepen()      │  │
│  │ - score()       │    │ - merge()      │    │ - expand_best() │  │
│  └────────┬────────┘    └───────┬────────┘    └────────┬────────┘  │
│           │                     │                      │            │
│           ▼                     ▼                      ▼            │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                     LazySearchState                             ││
│  │  - query, context, claims, visited, priority_queue              ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Search Backends                            │   │
│  │  ┌───────────────┐  ┌──────────────────┐  ┌───────────────┐ │   │
│  │  │ VectorSearch  │  │ CommunitySearch  │  │ GraphSearch   │ │   │
│  │  │ (Level 0)     │  │ (Level 1)        │  │ (Level 1)     │ │   │
│  │  └───────────────┘  └──────────────────┘  └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 検索フロー図

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Lazy Search Flow                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Query ─────┬────────────────────────────────────────────┐          │
│             │                                            │          │
│             ▼                                            │          │
│  ┌──────────────────┐                                   │          │
│  │ Level 0          │  Vector Search (初期候補)          │          │
│  │ Initial Fetch    │  ────────────────────────────────▶│          │
│  └────────┬─────────┘                                   │          │
│           │                                              │          │
│           ▼                                              ▼          │
│  ┌──────────────────┐                          ┌─────────────────┐ │
│  │ Relevance Test   │  関連性テスト              │ Priority Queue  │ │
│  │ (LLM Call)       │◀────────────────────────▶│ (Best-First)    │ │
│  └────────┬─────────┘                          └─────────────────┘ │
│           │                                              │          │
│           ▼                                              │          │
│  ┌──────────────────┐                                   │          │
│  │ Claim Extraction │  クレーム抽出（関連チャンクのみ）    │          │
│  │ (LLM Call)       │  ─────────────────────────────────│          │
│  └────────┬─────────┘                                   │          │
│           │                                              │          │
│           ▼                                              │          │
│  ┌──────────────────┐                                   │          │
│  │ Sufficient?      │  十分な情報か?                     │          │
│  │ (LLM判定)        │                                   │          │
│  └────────┬─────────┘                                   │          │
│           │                                              │          │
│     No    │     Yes                                      │          │
│     ▼     └─────────────────────────────────────────────┘          │
│  ┌──────────────────┐                                              │
│  │ Level 1          │  コミュニティ/グラフ展開                       │
│  │ Expand           │  ────────────────────────────────────────▶   │
│  └────────┬─────────┘                          (Priority Queue)    │
│           │                                                         │
│           └───────────────▶ (ループ)                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 クラス図

```python
from dataclasses import dataclass, field
from typing import Protocol, Callable
from enum import Enum
import heapq

# === Enums ===

class SearchLevel(Enum):
    """検索レベル"""
    LEVEL_0 = 0  # Vector Search (Baseline)
    LEVEL_1 = 1  # NLP Graph + Community
    LEVEL_2 = 2  # LLM Entity Extraction (Future)
    LEVEL_3 = 3  # Full Graph (Future)
    LEVEL_4 = 4  # Graph + Community Summary (Future)

class RelevanceScore(Enum):
    """関連性スコア"""
    HIGH = 2      # 直接的に関連
    MEDIUM = 1    # 部分的に関連
    LOW = 0       # 関連なし

# === Protocols ===

class RelevanceTesterProtocol(Protocol):
    """関連性テストプロトコル"""
    def test(self, query: str, text: str) -> RelevanceScore: ...
    def test_batch(self, query: str, texts: list[str]) -> list[RelevanceScore]: ...

class ClaimExtractorProtocol(Protocol):
    """クレーム抽出プロトコル"""
    def extract(self, query: str, text: str) -> list["Claim"]: ...
    def extract_batch(self, query: str, texts: list[str]) -> list["Claim"]: ...

class IterativeDeepenerProtocol(Protocol):
    """動的深化プロトコル"""
    def should_deepen(self, state: "LazySearchState") -> bool: ...
    def get_next_candidates(self, state: "LazySearchState") -> list["SearchCandidate"]: ...

# === Data Classes ===

@dataclass
class Claim:
    """抽出されたクレーム"""
    text: str
    source_text_unit_id: str
    source_document_id: str
    confidence: float = 1.0
    
    # メタデータ
    extracted_at: str = ""
    relevance_score: RelevanceScore = RelevanceScore.HIGH

@dataclass
class SearchCandidate:
    """検索候補"""
    id: str  # text_unit_id or community_id
    source: str  # "vector" | "community" | "graph"
    priority: float  # 優先度（高いほど優先）
    level: SearchLevel
    
    # メタデータ
    text: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __lt__(self, other):
        # heapq用（優先度の高い順）
        return self.priority > other.priority

@dataclass
class LazySearchState:
    """Lazy Search の状態"""
    query: str
    
    # 収集した情報
    context: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    
    # 訪問済み
    visited_text_units: set[str] = field(default_factory=set)
    visited_communities: set[str] = field(default_factory=set)
    
    # 優先度キュー
    priority_queue: list[SearchCandidate] = field(default_factory=list)
    
    # 統計
    llm_calls: int = 0
    tokens_used: int = 0
    current_level: SearchLevel = SearchLevel.LEVEL_0
    
    def add_candidate(self, candidate: SearchCandidate):
        """候補を追加"""
        heapq.heappush(self.priority_queue, candidate)
    
    def pop_candidate(self) -> SearchCandidate | None:
        """最優先の候補を取得"""
        if self.priority_queue:
            return heapq.heappop(self.priority_queue)
        return None
    
    def mark_visited(self, candidate: SearchCandidate):
        """訪問済みにマーク"""
        if candidate.source in ["vector", "graph"]:
            self.visited_text_units.add(candidate.id)
        elif candidate.source == "community":
            self.visited_communities.add(candidate.id)

@dataclass
class LazySearchResult:
    """Lazy Search 結果"""
    query: str
    answer: str
    claims: list[Claim]
    citations: list["Citation"]
    
    # メタデータ
    search_level_reached: SearchLevel
    llm_calls: int
    tokens_used: int
    total_time_ms: float
    
    # 内部状態（デバッグ用）
    final_state: LazySearchState | None = None
```

---

## 3. 詳細設計

### 3.1 RelevanceTester

```python
class RelevanceTester:
    """関連性テスト"""
    
    RELEVANCE_PROMPT = """
以下のテキストが、クエリに対してどの程度関連しているか判定してください。

クエリ: {query}

テキスト:
{text}

判定基準:
- HIGH: クエリの回答に直接役立つ情報を含む
- MEDIUM: 関連するが、直接的な回答ではない
- LOW: ほとんど関連がない

回答は HIGH, MEDIUM, LOW のいずれかのみを出力してください。
"""
    
    def __init__(self, llm_client: "LLMClientProtocol"):
        self.llm_client = llm_client
    
    def test(self, query: str, text: str) -> RelevanceScore:
        """単一テキストの関連性をテスト"""
        prompt = self.RELEVANCE_PROMPT.format(query=query, text=text[:1000])
        
        response = self.llm_client.generate(prompt).strip().upper()
        
        if "HIGH" in response:
            return RelevanceScore.HIGH
        elif "MEDIUM" in response:
            return RelevanceScore.MEDIUM
        else:
            return RelevanceScore.LOW
    
    def test_batch(
        self,
        query: str,
        texts: list[str],
        parallel: bool = True
    ) -> list[RelevanceScore]:
        """バッチで関連性をテスト"""
        
        if parallel and len(texts) > 1:
            # 並列処理（APIレート制限に注意）
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self.test, query, text)
                    for text in texts
                ]
                results = [f.result() for f in futures]
            return results
        else:
            return [self.test(query, text) for text in texts]
    
    def filter_relevant(
        self,
        query: str,
        candidates: list[SearchCandidate],
        min_relevance: RelevanceScore = RelevanceScore.MEDIUM
    ) -> list[tuple[SearchCandidate, RelevanceScore]]:
        """関連性でフィルタリング"""
        texts = [c.text for c in candidates]
        scores = self.test_batch(query, texts)
        
        results = []
        for candidate, score in zip(candidates, scores):
            if score.value >= min_relevance.value:
                results.append((candidate, score))
        
        return results
```

### 3.2 ClaimExtractor

```python
class ClaimExtractor:
    """クレーム抽出"""
    
    EXTRACTION_PROMPT = """
以下のテキストから、クエリに関連する主要な主張（claim）を抽出してください。

クエリ: {query}

テキスト:
{text}

出力形式:
各主張を1行ずつ、「- 」で始めて記述してください。
主張は事実に基づく簡潔な文にしてください。

例:
- Transformerは自己注意機構を使用する
- BERTは双方向のコンテキストを学習する
"""
    
    def __init__(self, llm_client: "LLMClientProtocol"):
        self.llm_client = llm_client
    
    def extract(
        self,
        query: str,
        text: str,
        source_text_unit_id: str = "",
        source_document_id: str = ""
    ) -> list[Claim]:
        """テキストからクレームを抽出"""
        prompt = self.EXTRACTION_PROMPT.format(query=query, text=text[:2000])
        
        response = self.llm_client.generate(prompt)
        
        claims = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                claim_text = line[2:].strip()
                if claim_text:
                    claims.append(Claim(
                        text=claim_text,
                        source_text_unit_id=source_text_unit_id,
                        source_document_id=source_document_id
                    ))
        
        return claims
    
    def extract_batch(
        self,
        query: str,
        candidates: list[SearchCandidate]
    ) -> list[Claim]:
        """バッチでクレーム抽出"""
        all_claims = []
        
        for candidate in candidates:
            claims = self.extract(
                query,
                candidate.text,
                source_text_unit_id=candidate.id,
                source_document_id=candidate.metadata.get("document_id", "")
            )
            all_claims.extend(claims)
        
        return self._merge_duplicates(all_claims)
    
    def _merge_duplicates(self, claims: list[Claim]) -> list[Claim]:
        """重複クレームをマージ"""
        # 簡易的な重複判定（将来的にはembedding類似度）
        seen = set()
        unique = []
        
        for claim in claims:
            # 正規化
            normalized = claim.text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(claim)
        
        return unique
```

### 3.3 IterativeDeepener

```python
class IterativeDeepener:
    """動的深化"""
    
    SUFFICIENCY_PROMPT = """
以下の情報で、クエリに十分に回答できますか？

クエリ: {query}

収集された情報:
{claims}

判定:
- SUFFICIENT: 十分な情報がある
- INSUFFICIENT: もっと情報が必要

回答は SUFFICIENT または INSUFFICIENT のみを出力してください。
"""
    
    def __init__(
        self,
        llm_client: "LLMClientProtocol",
        community_searcher: "CommunitySearcher",
        graph_searcher: "GraphSearcher",
        max_iterations: int = 5,
        max_llm_calls: int = 20
    ):
        self.llm_client = llm_client
        self.community_searcher = community_searcher
        self.graph_searcher = graph_searcher
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
    
    def should_deepen(self, state: LazySearchState) -> bool:
        """深化すべきか判定"""
        # 停止条件
        if state.llm_calls >= self.max_llm_calls:
            return False
        
        if not state.priority_queue:
            return False
        
        # LLMで十分性を判定
        claims_text = "\n".join(f"- {c.text}" for c in state.claims[:20])
        prompt = self.SUFFICIENCY_PROMPT.format(
            query=state.query,
            claims=claims_text
        )
        
        response = self.llm_client.generate(prompt).strip().upper()
        state.llm_calls += 1
        
        return "INSUFFICIENT" in response
    
    def get_next_candidates(
        self,
        state: LazySearchState,
        batch_size: int = 5
    ) -> list[SearchCandidate]:
        """次の候補を取得"""
        candidates = []
        
        for _ in range(batch_size):
            candidate = state.pop_candidate()
            if candidate is None:
                break
            
            # 訪問済みスキップ
            if candidate.id in state.visited_text_units:
                continue
            if candidate.id in state.visited_communities:
                continue
            
            candidates.append(candidate)
        
        return candidates
    
    def expand_from_community(
        self,
        community_id: str,
        state: LazySearchState
    ) -> list[SearchCandidate]:
        """コミュニティから展開"""
        # コミュニティ内のTextUnitを取得
        text_units = self.community_searcher.get_text_units(community_id)
        
        candidates = []
        for tu in text_units:
            if tu.id not in state.visited_text_units:
                candidates.append(SearchCandidate(
                    id=tu.id,
                    source="community",
                    priority=0.5,  # コミュニティ展開は中程度の優先度
                    level=SearchLevel.LEVEL_1,
                    text=tu.text,
                    metadata={"document_id": tu.document_id}
                ))
        
        return candidates
    
    def expand_from_graph(
        self,
        node_id: str,
        state: LazySearchState,
        hop: int = 1
    ) -> list[SearchCandidate]:
        """グラフから展開"""
        # 隣接ノードを取得
        neighbors = self.graph_searcher.get_neighbors(node_id, hop=hop)
        
        candidates = []
        for neighbor in neighbors:
            if neighbor.id not in state.visited_text_units:
                candidates.append(SearchCandidate(
                    id=neighbor.id,
                    source="graph",
                    priority=neighbor.edge_weight * 0.8,
                    level=SearchLevel.LEVEL_1,
                    text=neighbor.text,
                    metadata={"document_id": neighbor.document_id}
                ))
        
        return candidates
```

### 3.4 CommunitySearcher

```python
class CommunitySearcher:
    """コミュニティ検索"""
    
    def __init__(
        self,
        level1_index: "Level1Index",
        embedding_client: "EmbeddingClientProtocol"
    ):
        self.index = level1_index
        self.embedding_client = embedding_client
        
        # コミュニティ埋め込みをキャッシュ
        self._community_embeddings = None
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[SearchCandidate]:
        """クエリに関連するコミュニティを検索"""
        
        # コミュニティの代表フレーズを埋め込み
        if self._community_embeddings is None:
            self._build_community_embeddings()
        
        # クエリ埋め込み
        query_embedding = self.embedding_client.embed(query)
        
        # 類似度検索
        scores = []
        for comm, emb in self._community_embeddings.items():
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((comm, score))
        
        # Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_communities = scores[:top_k]
        
        candidates = []
        for comm_id, score in top_communities:
            comm = self._get_community(comm_id)
            candidates.append(SearchCandidate(
                id=comm_id,
                source="community",
                priority=score,
                level=SearchLevel.LEVEL_1,
                text=", ".join(comm.representative_phrases),
                metadata={
                    "size": comm.size,
                    "node_ids": comm.node_ids
                }
            ))
        
        return candidates
    
    def get_text_units(self, community_id: str) -> list["TextUnit"]:
        """コミュニティに属するTextUnitを取得"""
        comm = self._get_community(community_id)
        
        # ノードからTextUnitを取得
        text_unit_ids = set()
        for node_id in comm.node_ids:
            node = self._get_node(node_id)
            text_unit_ids.update(node.text_unit_ids)
        
        # TextUnitを読み込み
        # (Level 0 の Parquet から)
        return self._load_text_units(list(text_unit_ids))
    
    def _build_community_embeddings(self):
        """コミュニティ埋め込みを構築"""
        self._community_embeddings = {}
        
        for comm in self.index.communities:
            # 代表フレーズを結合して埋め込み
            text = " ".join(comm.representative_phrases)
            emb = self.embedding_client.embed(text)
            self._community_embeddings[comm.id] = emb
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """コサイン類似度"""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _get_community(self, community_id: str) -> "Community":
        """コミュニティを取得"""
        for comm in self.index.communities:
            if comm.id == community_id:
                return comm
        raise ValueError(f"Community not found: {community_id}")
    
    def _get_node(self, node_id: str) -> "NounPhraseNode":
        """ノードを取得"""
        for node in self.index.nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"Node not found: {node_id}")
    
    def _load_text_units(self, text_unit_ids: list[str]) -> list["TextUnit"]:
        """TextUnitを読み込み"""
        import pyarrow.parquet as pq
        
        table = pq.read_table("./output/index/level_0/text_units.parquet")
        df = table.to_pandas()
        
        filtered = df[df["id"].isin(text_unit_ids)]
        
        return [
            TextUnit(
                id=row["id"],
                document_id=row["document_id"],
                text=row["text"],
                chunk_index=row["chunk_index"]
            )
            for _, row in filtered.iterrows()
        ]
```

### 3.5 LazySearchEngine (Facade)

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class LazySearchConfig:
    """Lazy Search 設定"""
    # 検索設定
    initial_top_k: int = 20
    batch_size: int = 5
    
    # 深化設定
    max_iterations: int = 5
    max_llm_calls: int = 20
    
    # 関連性フィルタ
    min_relevance: RelevanceScore = RelevanceScore.MEDIUM
    
    # コミュニティ検索
    community_top_k: int = 3
    
    # 出力設定
    include_debug_state: bool = False


class LazySearchEngine:
    """Lazy Search エンジン"""
    
    def __init__(
        self,
        vector_search_engine: "VectorSearchEngine",
        level1_index: "Level1Index",
        embedding_client: "EmbeddingClientProtocol",
        llm_client: "LLMClientProtocol",
        config: LazySearchConfig | None = None
    ):
        self.vector_engine = vector_search_engine
        self.community_searcher = CommunitySearcher(level1_index, embedding_client)
        self.relevance_tester = RelevanceTester(llm_client)
        self.claim_extractor = ClaimExtractor(llm_client)
        self.deepener = IterativeDeepener(
            llm_client,
            self.community_searcher,
            None  # GraphSearcher (Phase 2)
        )
        self.llm_client = llm_client
        self.config = config or LazySearchConfig()
    
    def search(
        self,
        query: str,
        max_level: SearchLevel = SearchLevel.LEVEL_1
    ) -> LazySearchResult:
        """Lazy Search を実行"""
        start_time = time.time()
        
        # 状態初期化
        state = LazySearchState(query=query)
        
        # Step 1: Level 0 初期検索
        initial_results = self._initial_search(query, state)
        
        # Step 2: 関連性テスト
        relevant_candidates = self._test_relevance(query, initial_results, state)
        
        # Step 3: クレーム抽出
        self._extract_claims(query, relevant_candidates, state)
        
        # Step 4: 動的深化（必要に応じて）
        iteration = 0
        while (
            max_level.value >= SearchLevel.LEVEL_1.value and
            iteration < self.config.max_iterations and
            self.deepener.should_deepen(state)
        ):
            self._deepen(state)
            iteration += 1
        
        # Step 5: 最終回答生成
        answer = self._synthesize_answer(query, state)
        
        total_time = (time.time() - start_time) * 1000
        
        return LazySearchResult(
            query=query,
            answer=answer.answer,
            claims=state.claims,
            citations=answer.citations,
            search_level_reached=state.current_level,
            llm_calls=state.llm_calls,
            tokens_used=state.tokens_used,
            total_time_ms=total_time,
            final_state=state if self.config.include_debug_state else None
        )
    
    def _initial_search(
        self,
        query: str,
        state: LazySearchState
    ) -> list[SearchCandidate]:
        """Level 0 初期検索"""
        # ベクトル検索
        vector_results = self.vector_engine.search(
            query,
            top_k=self.config.initial_top_k,
            synthesize=False
        )
        
        candidates = []
        for hit in vector_results.search_results.hits:
            candidates.append(SearchCandidate(
                id=hit.text_unit_id,
                source="vector",
                priority=hit.score,
                level=SearchLevel.LEVEL_0,
                text=hit.text,
                metadata={
                    "document_id": hit.document_id,
                    "document_title": hit.document_title
                }
            ))
        
        # コミュニティ検索
        community_candidates = self.community_searcher.search(
            query,
            top_k=self.config.community_top_k
        )
        
        # すべて優先度キューに追加
        for c in candidates + community_candidates:
            state.add_candidate(c)
        
        state.current_level = SearchLevel.LEVEL_0
        return candidates
    
    def _test_relevance(
        self,
        query: str,
        candidates: list[SearchCandidate],
        state: LazySearchState
    ) -> list[SearchCandidate]:
        """関連性テスト"""
        results = self.relevance_tester.filter_relevant(
            query,
            candidates,
            min_relevance=self.config.min_relevance
        )
        
        state.llm_calls += len(candidates)
        
        relevant = [c for c, score in results]
        
        # 訪問済みにマーク
        for c in relevant:
            state.mark_visited(c)
            state.context.append(c.text)
        
        return relevant
    
    def _extract_claims(
        self,
        query: str,
        candidates: list[SearchCandidate],
        state: LazySearchState
    ):
        """クレーム抽出"""
        claims = self.claim_extractor.extract_batch(query, candidates)
        state.claims.extend(claims)
        state.llm_calls += len(candidates)
    
    def _deepen(self, state: LazySearchState):
        """深化イテレーション"""
        # 次の候補を取得
        next_candidates = self.deepener.get_next_candidates(
            state,
            batch_size=self.config.batch_size
        )
        
        if not next_candidates:
            return
        
        # コミュニティ展開
        for candidate in next_candidates:
            if candidate.source == "community":
                expanded = self.deepener.expand_from_community(
                    candidate.id,
                    state
                )
                for c in expanded:
                    state.add_candidate(c)
        
        # 関連性テスト
        text_candidates = [c for c in next_candidates if c.source != "community"]
        if text_candidates:
            relevant = self._test_relevance(state.query, text_candidates, state)
            
            if relevant:
                self._extract_claims(state.query, relevant, state)
        
        state.current_level = SearchLevel.LEVEL_1
    
    def _synthesize_answer(
        self,
        query: str,
        state: LazySearchState
    ) -> "SynthesizedAnswer":
        """最終回答を合成"""
        # クレームからコンテキスト構築
        context_parts = []
        source_map = {}
        
        for i, claim in enumerate(state.claims[:20]):
            context_parts.append(f"[{i+1}] {claim.text}")
            source_map[i+1] = claim
        
        context_text = "\n".join(context_parts)
        
        # 回答生成
        prompt = f"""
以下の情報に基づいて、質問に回答してください。

## 収集された情報
{context_text}

## 質問
{query}

## 指示
- 情報を統合して簡潔に回答してください
- 使用した情報源は [1], [2] のように引用してください
"""
        
        answer = self.llm_client.generate(prompt)
        state.llm_calls += 1
        
        # 引用抽出
        import re
        cited_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
        
        citations = []
        for idx in sorted(cited_indices):
            if idx in source_map:
                claim = source_map[idx]
                citations.append(Citation(
                    text_unit_id=claim.source_text_unit_id,
                    document_id=claim.source_document_id,
                    document_title="",
                    text_snippet=claim.text,
                    relevance_score=claim.confidence
                ))
        
        return SynthesizedAnswer(
            answer=answer,
            citations=citations
        )
```

---

## 4. 設定

```yaml
# config/lazy_search.yaml

lazy_search:
  # 初期検索設定
  initial:
    top_k: 20
    include_communities: true
    community_top_k: 3
  
  # 関連性テスト
  relevance:
    min_score: medium  # low | medium | high
    batch_parallel: true
    max_workers: 5
  
  # 動的深化
  deepening:
    enabled: true
    max_iterations: 5
    max_llm_calls: 20
    batch_size: 5
  
  # クレーム抽出
  claims:
    max_per_text: 5
    merge_duplicates: true
  
  # 出力
  output:
    max_claims_in_answer: 20
    include_debug_state: false

# LLM設定（各コンポーネント用）
llm:
  relevance_test:
    model: llama3:8b-instruct-q4_K_M
    temperature: 0.1
    max_tokens: 10
  
  claim_extraction:
    model: llama3:8b-instruct-q4_K_M
    temperature: 0.3
    max_tokens: 500
  
  synthesis:
    model: llama3:8b-instruct-q4_K_M
    temperature: 0.5
    max_tokens: 1000
```

---

## 5. テスト計画

### 5.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_relevance_high | RelevanceTester | HIGH判定 |
| test_relevance_low | RelevanceTester | LOW判定 |
| test_claim_extraction | ClaimExtractor | クレームリストを返す |
| test_claim_merge | ClaimExtractor._merge_duplicates | 重複削除 |
| test_should_deepen | IterativeDeepener | bool返却 |
| test_community_search | CommunitySearcher | 候補リスト返却 |

### 5.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_lazy_search_level0 | LazySearchEngine (Level 0のみ) | 回答生成 |
| test_lazy_search_level1 | LazySearchEngine (Level 1まで) | コミュニティ展開 |
| test_lazy_search_no_deepen | LazySearchEngine (十分な情報) | 早期終了 |

### 5.3 パフォーマンス/コストテスト

| テストケース | 条件 | 期待結果 |
|-------------|------|---------|
| test_llm_call_limit | max_llm_calls=20 | 上限で停止 |
| test_lazy_vs_baseline | 同一クエリ | Lazy < Baseline (LLM calls) |
| test_response_time | 標準クエリ | < 30秒 |

---

## 6. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-005-01 | RelevanceTester 実装 | 2h | - |
| TASK-005-02 | ClaimExtractor 実装 | 2h | - |
| TASK-005-03 | IterativeDeepener 実装 | 3h | - |
| TASK-005-04 | CommunitySearcher 実装 | 3h | FEAT-003 |
| TASK-005-05 | LazySearchState 実装 | 1h | - |
| TASK-005-06 | LazySearchEngine 実装 | 4h | TASK-005-01~05 |
| TASK-005-07 | 単体テスト作成 | 2h | TASK-005-01~06 |
| TASK-005-08 | 統合テスト作成 | 2h | TASK-005-07 |
| TASK-005-09 | コスト比較テスト | 1h | TASK-005-08 |
| **合計** | | **20h** | |

---

## 7. 受入基準

- [ ] Level 0 ベクトル検索から開始できる
- [ ] 関連性テストでフィルタリングできる
- [ ] 関連チャンクからクレームを抽出できる
- [ ] 十分性判定で早期終了できる
- [ ] Level 1 コミュニティを展開できる
- [ ] 動的深化で追加情報を収集できる
- [ ] LLMコールを指定上限で停止できる
- [ ] Baseline RAGより少ないLLMコストで同等品質の回答を生成
- [ ] 引用付きの回答を生成できる

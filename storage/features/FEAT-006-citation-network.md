# FEAT-006: Citation Network

**フィーチャーID**: FEAT-006  
**名称**: 引用ネットワーク構築・分析  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

学術論文間の引用関係を構築・分析し、検索精度と文献探索を強化する機能。

### 1.1 スコープ

```
Documents → Reference Extraction → Citation Graph → Analysis Metrics
```

- **入力**: 解析済みドキュメント (AcademicPaperDocument[])
- **処理**: 引用関係抽出、グラフ構築、メトリクス計算
- **出力**: Citation Graph + Analysis結果
- **特徴**: 影響力分析、関連論文探索、引用チェーン可視化

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-CIT-NET-001 | 引用グラフ構築 | P0 |
| FR-CIT-NET-002 | 被引用数計算 | P0 |
| FR-CIT-NET-003 | 引用チェーン分析 | P0 |
| FR-CIT-NET-004 | 影響力スコア | P1 |
| FR-CIT-ANA-001 | 関連論文推薦 | P1 |
| FR-CIT-ANA-002 | 引用パターン分析 | P2 |

### 1.3 依存関係

- **依存**: FEAT-001 (Document Processing)
- **被依存**: FEAT-005 (Lazy Search - 将来的な統合), FEAT-007 (Python API)

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Citation Network System                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ ReferenceResolver│    │ CitationGraph    │    │ MetricsCalc   │ │
│  │                  │    │ Builder          │    │               │ │
│  │ - resolve()      │───▶│ - build()        │───▶│ - calculate() │ │
│  │ - match()        │    │ - add_edge()     │    │ - pagerank()  │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│         │                        │                       │          │
│         ▼                        ▼                       ▼          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ ExternalAPI      │    │ Citation Graph   │    │ Metrics       │ │
│  │ (CrossRef/S2)    │    │ (NetworkX)       │    │ (DataFrame)   │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Analysis Tools                            │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │   │
│  │  │ PathFinder    │  │ Recommender   │  │ Visualizer      │  │   │
│  │  │ (引用チェーン) │  │ (関連論文)    │  │ (可視化)        │  │   │
│  │  └───────────────┘  └───────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 出力ディレクトリ構造

```
output/
└── citation/
    ├── citation_graph.parquet      # 引用グラフ（エッジリスト）
    ├── document_metrics.parquet    # ドキュメントメトリクス
    └── citation_graph.graphml      # NetworkX形式（可視化用）
```

### 2.3 クラス図

```python
from dataclasses import dataclass, field
from typing import Protocol
from enum import Enum
from datetime import datetime

# === Enums ===

class ReferenceMatchStatus(Enum):
    """参照マッチ状態"""
    MATCHED = "matched"          # コーパス内で一致
    EXTERNAL = "external"        # 外部論文
    UNRESOLVED = "unresolved"    # 解決できず

# === Protocols ===

class ReferenceResolverProtocol(Protocol):
    """参照解決プロトコル"""
    def resolve(self, reference: "Reference") -> "ResolvedReference": ...
    def resolve_batch(self, references: list["Reference"]) -> list["ResolvedReference"]: ...

class CitationGraphBuilderProtocol(Protocol):
    """引用グラフビルダープロトコル"""
    def build(self, documents: list["AcademicPaperDocument"]) -> "CitationGraph": ...
    def add_citation(self, source_id: str, target_id: str, metadata: dict = None) -> None: ...

class MetricsCalculatorProtocol(Protocol):
    """メトリクス計算プロトコル"""
    def calculate_all(self, graph: "CitationGraph") -> "DocumentMetrics": ...
    def calculate_pagerank(self, graph: "CitationGraph") -> dict[str, float]: ...

# === Data Classes ===

@dataclass
class ResolvedReference:
    """解決済み参照"""
    original_reference: "Reference"
    status: ReferenceMatchStatus
    
    # マッチした場合
    matched_document_id: str | None = None
    match_confidence: float = 0.0
    
    # 外部APIからの情報
    external_doi: str | None = None
    external_title: str | None = None

@dataclass
class CitationEdge:
    """引用エッジ"""
    source_id: str  # 引用元ドキュメント
    target_id: str  # 引用先ドキュメント
    
    # メタデータ
    citation_context: str = ""  # 引用箇所のテキスト
    citation_type: str = ""     # "background" | "method" | "comparison" | "extension"
    
    # 解決情報
    is_internal: bool = True    # コーパス内の引用か
    confidence: float = 1.0

@dataclass
class CitationGraph:
    """引用グラフ"""
    nodes: list[str]  # document_ids
    edges: list[CitationEdge]
    
    # メタデータ
    node_metadata: dict[str, dict] = field(default_factory=dict)
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    @property
    def internal_edges(self) -> list[CitationEdge]:
        return [e for e in self.edges if e.is_internal]

@dataclass
class DocumentMetrics:
    """ドキュメントメトリクス"""
    document_id: str
    
    # 基本メトリクス
    citation_count: int = 0      # 被引用数
    reference_count: int = 0     # 参照数
    
    # ネットワークメトリクス
    pagerank: float = 0.0
    hub_score: float = 0.0       # HITS hub
    authority_score: float = 0.0 # HITS authority
    
    # 年次情報
    year: int | None = None
    
    # 影響力指標
    influence_score: float = 0.0  # 総合影響力スコア

@dataclass
class CitationPath:
    """引用パス"""
    source_id: str
    target_id: str
    path: list[str]  # document_ids
    
    @property
    def length(self) -> int:
        return len(self.path) - 1
    
    @property
    def intermediate_nodes(self) -> list[str]:
        return self.path[1:-1]

@dataclass
class RelatedPaper:
    """関連論文"""
    document_id: str
    title: str
    similarity_score: float
    relationship_type: str  # "cites" | "cited_by" | "co_citation" | "bibliographic_coupling"
```

---

## 3. 詳細設計

### 3.1 ReferenceResolver

```python
from difflib import SequenceMatcher

class ReferenceResolver:
    """参照解決"""
    
    def __init__(
        self,
        document_index: dict[str, "AcademicPaperDocument"],
        use_external_api: bool = False
    ):
        self.document_index = document_index
        self.use_external_api = use_external_api
        
        # タイトルインデックス（高速マッチング用）
        self.title_index = self._build_title_index()
        
        # DOIインデックス
        self.doi_index = {
            doc.doi: doc_id
            for doc_id, doc in document_index.items()
            if doc.doi
        }
    
    def _build_title_index(self) -> dict[str, str]:
        """タイトルインデックスを構築"""
        index = {}
        for doc_id, doc in self.document_index.items():
            normalized = self._normalize_title(doc.title)
            index[normalized] = doc_id
        return index
    
    def _normalize_title(self, title: str) -> str:
        """タイトルを正規化"""
        import re
        # 小文字化、特殊文字除去
        normalized = title.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def resolve(self, reference: "Reference") -> ResolvedReference:
        """参照を解決"""
        
        # 1. DOIでマッチ
        if reference.doi:
            if reference.doi in self.doi_index:
                return ResolvedReference(
                    original_reference=reference,
                    status=ReferenceMatchStatus.MATCHED,
                    matched_document_id=self.doi_index[reference.doi],
                    match_confidence=1.0
                )
        
        # 2. タイトルでマッチ
        if reference.title:
            normalized = self._normalize_title(reference.title)
            
            # 完全一致
            if normalized in self.title_index:
                return ResolvedReference(
                    original_reference=reference,
                    status=ReferenceMatchStatus.MATCHED,
                    matched_document_id=self.title_index[normalized],
                    match_confidence=1.0
                )
            
            # ファジーマッチ
            best_match = self._fuzzy_match_title(normalized)
            if best_match:
                return ResolvedReference(
                    original_reference=reference,
                    status=ReferenceMatchStatus.MATCHED,
                    matched_document_id=best_match[0],
                    match_confidence=best_match[1]
                )
        
        # 3. 外部API（オプション）
        if self.use_external_api:
            external = self._resolve_external(reference)
            if external:
                return external
        
        # 4. 解決できず
        return ResolvedReference(
            original_reference=reference,
            status=ReferenceMatchStatus.UNRESOLVED
        )
    
    def _fuzzy_match_title(
        self,
        normalized_title: str,
        threshold: float = 0.85
    ) -> tuple[str, float] | None:
        """タイトルのファジーマッチ"""
        best_match = None
        best_score = 0.0
        
        for title, doc_id in self.title_index.items():
            score = SequenceMatcher(None, normalized_title, title).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (doc_id, score)
        
        return best_match
    
    def _resolve_external(self, reference: "Reference") -> ResolvedReference | None:
        """外部APIで解決（CrossRef / Semantic Scholar）"""
        # TODO: CrossRef API 実装
        # TODO: Semantic Scholar API 実装
        return None
    
    def resolve_batch(
        self,
        references: list["Reference"]
    ) -> list[ResolvedReference]:
        """バッチで解決"""
        return [self.resolve(ref) for ref in references]
```

### 3.2 CitationGraphBuilder

```python
import networkx as nx

class CitationGraphBuilder:
    """引用グラフビルダー"""
    
    def __init__(self, resolver: ReferenceResolver):
        self.resolver = resolver
        self.graph = nx.DiGraph()
    
    def build(
        self,
        documents: list["AcademicPaperDocument"]
    ) -> CitationGraph:
        """引用グラフを構築"""
        
        # 1. ノード追加
        for doc in documents:
            self.graph.add_node(
                doc.id,
                title=doc.title,
                year=doc.year,
                authors=[a.name for a in doc.authors],
                doi=doc.doi
            )
        
        # 2. エッジ追加（引用関係）
        edges = []
        for doc in documents:
            for ref in doc.references:
                resolved = self.resolver.resolve(ref)
                
                if resolved.status == ReferenceMatchStatus.MATCHED:
                    edge = CitationEdge(
                        source_id=doc.id,
                        target_id=resolved.matched_document_id,
                        is_internal=True,
                        confidence=resolved.match_confidence
                    )
                    edges.append(edge)
                    
                    self.graph.add_edge(
                        doc.id,
                        resolved.matched_document_id,
                        confidence=resolved.match_confidence
                    )
                
                elif resolved.status == ReferenceMatchStatus.EXTERNAL:
                    # 外部ノードを追加
                    external_id = f"external_{resolved.external_doi or hash(ref.title)}"
                    
                    if external_id not in self.graph:
                        self.graph.add_node(
                            external_id,
                            title=resolved.external_title or ref.title,
                            is_external=True
                        )
                    
                    edge = CitationEdge(
                        source_id=doc.id,
                        target_id=external_id,
                        is_internal=False
                    )
                    edges.append(edge)
        
        # 3. CitationGraph構築
        nodes = list(self.graph.nodes())
        node_metadata = {
            n: dict(self.graph.nodes[n])
            for n in nodes
        }
        
        return CitationGraph(
            nodes=nodes,
            edges=edges,
            node_metadata=node_metadata
        )
    
    def add_citation(
        self,
        source_id: str,
        target_id: str,
        metadata: dict = None
    ):
        """引用を追加"""
        self.graph.add_edge(source_id, target_id, **(metadata or {}))
    
    def get_networkx_graph(self) -> nx.DiGraph:
        """NetworkXグラフを取得"""
        return self.graph
    
    def export_graphml(self, filepath: str):
        """GraphML形式でエクスポート"""
        nx.write_graphml(self.graph, filepath)
```

### 3.3 MetricsCalculator

```python
import networkx as nx
import numpy as np

class MetricsCalculator:
    """メトリクス計算"""
    
    def __init__(self):
        pass
    
    def calculate_all(
        self,
        graph: CitationGraph,
        nx_graph: nx.DiGraph
    ) -> list[DocumentMetrics]:
        """全メトリクスを計算"""
        
        # PageRank
        pagerank = self.calculate_pagerank(nx_graph)
        
        # HITS
        hub_scores, authority_scores = self.calculate_hits(nx_graph)
        
        # 基本メトリクス
        in_degree = dict(nx_graph.in_degree())
        out_degree = dict(nx_graph.out_degree())
        
        # DocumentMetrics作成
        metrics_list = []
        for node in graph.nodes:
            year = graph.node_metadata.get(node, {}).get("year")
            
            metrics = DocumentMetrics(
                document_id=node,
                citation_count=in_degree.get(node, 0),
                reference_count=out_degree.get(node, 0),
                pagerank=pagerank.get(node, 0.0),
                hub_score=hub_scores.get(node, 0.0),
                authority_score=authority_scores.get(node, 0.0),
                year=year
            )
            
            # 影響力スコア（加重平均）
            metrics.influence_score = self._calculate_influence(metrics)
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def calculate_pagerank(
        self,
        graph: nx.DiGraph,
        alpha: float = 0.85
    ) -> dict[str, float]:
        """PageRankを計算"""
        return nx.pagerank(graph, alpha=alpha)
    
    def calculate_hits(
        self,
        graph: nx.DiGraph
    ) -> tuple[dict[str, float], dict[str, float]]:
        """HITS (Hubs and Authorities) を計算"""
        try:
            hubs, authorities = nx.hits(graph, max_iter=100, normalized=True)
            return hubs, authorities
        except nx.PowerIterationFailedConvergence:
            # 収束しない場合は空を返す
            return {}, {}
    
    def _calculate_influence(self, metrics: DocumentMetrics) -> float:
        """影響力スコアを計算（加重平均）"""
        # 正規化された各スコアの加重平均
        weights = {
            "citation_count": 0.3,
            "pagerank": 0.4,
            "authority_score": 0.3
        }
        
        # 簡易的な正規化（ログスケール）
        citation_norm = np.log1p(metrics.citation_count) / 10
        
        score = (
            weights["citation_count"] * min(citation_norm, 1.0) +
            weights["pagerank"] * min(metrics.pagerank * 100, 1.0) +
            weights["authority_score"] * metrics.authority_score
        )
        
        return score
```

### 3.4 CitationAnalyzer

```python
class CitationAnalyzer:
    """引用分析"""
    
    def __init__(
        self,
        graph: CitationGraph,
        nx_graph: nx.DiGraph,
        metrics: list[DocumentMetrics]
    ):
        self.graph = graph
        self.nx_graph = nx_graph
        self.metrics = {m.document_id: m for m in metrics}
    
    def find_citation_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> CitationPath | None:
        """引用パスを探索"""
        try:
            path = nx.shortest_path(
                self.nx_graph,
                source=source_id,
                target=target_id
            )
            
            if len(path) - 1 <= max_length:
                return CitationPath(
                    source_id=source_id,
                    target_id=target_id,
                    path=path
                )
        except nx.NetworkXNoPath:
            pass
        
        return None
    
    def get_citation_chain(
        self,
        document_id: str,
        direction: str = "both",
        depth: int = 2
    ) -> dict[str, list[str]]:
        """引用チェーンを取得"""
        result = {
            "cites": [],      # この論文が引用している
            "cited_by": []    # この論文を引用している
        }
        
        if direction in ["both", "out"]:
            # 前方探索（引用先）
            result["cites"] = self._bfs_traverse(
                document_id,
                direction="out",
                depth=depth
            )
        
        if direction in ["both", "in"]:
            # 後方探索（被引用元）
            result["cited_by"] = self._bfs_traverse(
                document_id,
                direction="in",
                depth=depth
            )
        
        return result
    
    def _bfs_traverse(
        self,
        start_id: str,
        direction: str,
        depth: int
    ) -> list[str]:
        """BFSで探索"""
        visited = set()
        queue = [(start_id, 0)]
        result = []
        
        while queue:
            node, d = queue.pop(0)
            
            if node in visited:
                continue
            visited.add(node)
            
            if node != start_id:
                result.append(node)
            
            if d < depth:
                if direction == "out":
                    neighbors = self.nx_graph.successors(node)
                else:
                    neighbors = self.nx_graph.predecessors(node)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))
        
        return result
    
    def find_related_papers(
        self,
        document_id: str,
        top_k: int = 10
    ) -> list[RelatedPaper]:
        """関連論文を推薦"""
        related = []
        
        # 1. 直接引用
        for target in self.nx_graph.successors(document_id):
            related.append(RelatedPaper(
                document_id=target,
                title=self.graph.node_metadata.get(target, {}).get("title", ""),
                similarity_score=1.0,
                relationship_type="cites"
            ))
        
        # 2. 被引用
        for source in self.nx_graph.predecessors(document_id):
            related.append(RelatedPaper(
                document_id=source,
                title=self.graph.node_metadata.get(source, {}).get("title", ""),
                similarity_score=0.9,
                relationship_type="cited_by"
            ))
        
        # 3. 共引用（co-citation）
        co_cited = self._find_co_citations(document_id)
        for doc_id, count in co_cited[:top_k]:
            if doc_id != document_id and doc_id not in [r.document_id for r in related]:
                related.append(RelatedPaper(
                    document_id=doc_id,
                    title=self.graph.node_metadata.get(doc_id, {}).get("title", ""),
                    similarity_score=0.5 + 0.3 * min(count / 5, 1.0),
                    relationship_type="co_citation"
                ))
        
        # 4. 書誌結合（bibliographic coupling）
        biblio_coupled = self._find_bibliographic_coupling(document_id)
        for doc_id, count in biblio_coupled[:top_k]:
            if doc_id != document_id and doc_id not in [r.document_id for r in related]:
                related.append(RelatedPaper(
                    document_id=doc_id,
                    title=self.graph.node_metadata.get(doc_id, {}).get("title", ""),
                    similarity_score=0.4 + 0.3 * min(count / 5, 1.0),
                    relationship_type="bibliographic_coupling"
                ))
        
        # スコアでソート
        related.sort(key=lambda x: x.similarity_score, reverse=True)
        return related[:top_k]
    
    def _find_co_citations(self, document_id: str) -> list[tuple[str, int]]:
        """共引用を検索（同じ論文から引用されている）"""
        citing_papers = set(self.nx_graph.predecessors(document_id))
        
        co_cited_count = {}
        for citing in citing_papers:
            for cited in self.nx_graph.successors(citing):
                if cited != document_id:
                    co_cited_count[cited] = co_cited_count.get(cited, 0) + 1
        
        return sorted(co_cited_count.items(), key=lambda x: x[1], reverse=True)
    
    def _find_bibliographic_coupling(self, document_id: str) -> list[tuple[str, int]]:
        """書誌結合を検索（同じ論文を引用している）"""
        cited_papers = set(self.nx_graph.successors(document_id))
        
        coupling_count = {}
        for cited in cited_papers:
            for citing in self.nx_graph.predecessors(cited):
                if citing != document_id:
                    coupling_count[citing] = coupling_count.get(citing, 0) + 1
        
        return sorted(coupling_count.items(), key=lambda x: x[1], reverse=True)
    
    def get_most_influential(self, top_k: int = 10) -> list[DocumentMetrics]:
        """最も影響力のある論文を取得"""
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda x: x.influence_score,
            reverse=True
        )
        return sorted_metrics[:top_k]
```

### 3.5 CitationNetworkManager (Facade)

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class CitationNetworkBuildResult:
    """構築結果"""
    node_count: int
    edge_count: int
    internal_edge_count: int
    output_path: Path


class CitationNetworkManager:
    """引用ネットワーク管理"""
    
    def __init__(
        self,
        output_path: Path = Path("./output/citation")
    ):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.graph: CitationGraph | None = None
        self.nx_graph: nx.DiGraph | None = None
        self.metrics: list[DocumentMetrics] | None = None
        self.analyzer: CitationAnalyzer | None = None
    
    def build(
        self,
        documents: list["AcademicPaperDocument"],
        use_external_api: bool = False
    ) -> CitationNetworkBuildResult:
        """引用ネットワークを構築"""
        
        # 1. ドキュメントインデックス構築
        doc_index = {doc.id: doc for doc in documents}
        
        # 2. 参照解決
        resolver = ReferenceResolver(
            doc_index,
            use_external_api=use_external_api
        )
        
        # 3. グラフ構築
        builder = CitationGraphBuilder(resolver)
        self.graph = builder.build(documents)
        self.nx_graph = builder.get_networkx_graph()
        
        # 4. メトリクス計算
        calculator = MetricsCalculator()
        self.metrics = calculator.calculate_all(self.graph, self.nx_graph)
        
        # 5. アナライザー初期化
        self.analyzer = CitationAnalyzer(
            self.graph,
            self.nx_graph,
            self.metrics
        )
        
        # 6. 永続化
        self._save()
        
        return CitationNetworkBuildResult(
            node_count=self.graph.node_count,
            edge_count=self.graph.edge_count,
            internal_edge_count=len(self.graph.internal_edges),
            output_path=self.output_path
        )
    
    def _save(self):
        """永続化"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # エッジリスト
        edge_records = [
            {
                "source_id": e.source_id,
                "target_id": e.target_id,
                "is_internal": e.is_internal,
                "confidence": e.confidence,
                "citation_context": e.citation_context,
                "citation_type": e.citation_type
            }
            for e in self.graph.edges
        ]
        table = pa.Table.from_pylist(edge_records)
        pq.write_table(table, self.output_path / "citation_graph.parquet")
        
        # メトリクス
        metrics_records = [
            {
                "document_id": m.document_id,
                "citation_count": m.citation_count,
                "reference_count": m.reference_count,
                "pagerank": m.pagerank,
                "hub_score": m.hub_score,
                "authority_score": m.authority_score,
                "year": m.year,
                "influence_score": m.influence_score
            }
            for m in self.metrics
        ]
        table = pa.Table.from_pylist(metrics_records)
        pq.write_table(table, self.output_path / "document_metrics.parquet")
        
        # GraphML
        nx.write_graphml(self.nx_graph, self.output_path / "citation_graph.graphml")
    
    def load(self):
        """読み込み"""
        import pyarrow.parquet as pq
        
        # エッジ
        edge_table = pq.read_table(self.output_path / "citation_graph.parquet")
        edges = [
            CitationEdge(**r)
            for r in edge_table.to_pylist()
        ]
        
        # メトリクス
        metrics_table = pq.read_table(self.output_path / "document_metrics.parquet")
        self.metrics = [
            DocumentMetrics(**r)
            for r in metrics_table.to_pylist()
        ]
        
        # NetworkX
        self.nx_graph = nx.read_graphml(self.output_path / "citation_graph.graphml")
        
        # Graph復元
        nodes = list(self.nx_graph.nodes())
        node_metadata = {n: dict(self.nx_graph.nodes[n]) for n in nodes}
        
        self.graph = CitationGraph(
            nodes=nodes,
            edges=edges,
            node_metadata=node_metadata
        )
        
        # アナライザー
        self.analyzer = CitationAnalyzer(
            self.graph,
            self.nx_graph,
            self.metrics
        )
    
    # === 分析API ===
    
    def get_citation_chain(
        self,
        document_id: str,
        depth: int = 2
    ) -> dict[str, list[str]]:
        """引用チェーンを取得"""
        return self.analyzer.get_citation_chain(document_id, depth=depth)
    
    def find_citation_path(
        self,
        source_id: str,
        target_id: str
    ) -> CitationPath | None:
        """引用パスを探索"""
        return self.analyzer.find_citation_path(source_id, target_id)
    
    def find_related_papers(
        self,
        document_id: str,
        top_k: int = 10
    ) -> list[RelatedPaper]:
        """関連論文を推薦"""
        return self.analyzer.find_related_papers(document_id, top_k)
    
    def get_most_influential(self, top_k: int = 10) -> list[DocumentMetrics]:
        """最も影響力のある論文"""
        return self.analyzer.get_most_influential(top_k)
    
    def get_metrics(self, document_id: str) -> DocumentMetrics | None:
        """ドキュメントのメトリクスを取得"""
        return self.analyzer.metrics.get(document_id)
```

---

## 4. 設定

```yaml
# config/citation_network.yaml

citation_network:
  output_path: ./output/citation
  
  # 参照解決
  resolver:
    fuzzy_match_threshold: 0.85
    use_external_api: false
    
    # 外部API設定（オプション）
    external_api:
      crossref:
        enabled: false
        email: ""  # CrossRef API要件
      semantic_scholar:
        enabled: false
        api_key: ""
  
  # メトリクス計算
  metrics:
    pagerank_alpha: 0.85
    hits_max_iterations: 100
    
    # 影響力スコアの重み
    influence_weights:
      citation_count: 0.3
      pagerank: 0.4
      authority_score: 0.3
  
  # 分析設定
  analysis:
    max_path_length: 5
    default_chain_depth: 2
    related_papers_top_k: 10
```

---

## 5. テスト計画

### 5.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_resolve_by_doi | ReferenceResolver | DOI一致で解決 |
| test_resolve_by_title | ReferenceResolver | タイトル一致で解決 |
| test_fuzzy_match | ReferenceResolver | ファジーマッチで解決 |
| test_build_graph | CitationGraphBuilder | グラフ構築成功 |
| test_pagerank | MetricsCalculator | PageRank計算 |
| test_hits | MetricsCalculator | HITS計算 |

### 5.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_full_build | CitationNetworkManager.build | 全アーティファクト生成 |
| test_citation_chain | CitationAnalyzer | 引用チェーン取得 |
| test_find_path | CitationAnalyzer | パス探索成功 |
| test_related_papers | CitationAnalyzer | 関連論文推薦 |

### 5.3 パフォーマンステスト

| テストケース | 条件 | 期待結果 |
|-------------|------|---------|
| test_build_performance | 1000 docs | < 60 sec |
| test_path_search | 10000 nodes | < 100 ms |
| test_related_papers | 10000 nodes | < 500 ms |

---

## 6. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-006-01 | ReferenceResolver 実装 | 3h | - |
| TASK-006-02 | CitationGraphBuilder 実装 | 2h | TASK-006-01 |
| TASK-006-03 | MetricsCalculator 実装 | 2h | - |
| TASK-006-04 | CitationAnalyzer 実装 | 3h | TASK-006-02, 03 |
| TASK-006-05 | CitationNetworkManager 実装 | 2h | TASK-006-01~04 |
| TASK-006-06 | Parquet永続化 実装 | 1h | TASK-006-05 |
| TASK-006-07 | 単体テスト作成 | 2h | TASK-006-01~06 |
| TASK-006-08 | 統合テスト作成 | 2h | TASK-006-07 |
| **合計** | | **17h** | |

---

## 7. 受入基準

- [ ] DOIまたはタイトルで参照を解決できる
- [ ] コーパス内の引用関係を抽出してグラフ構築できる
- [ ] 被引用数、PageRank、HITSを計算できる
- [ ] 引用チェーン（前方・後方）を探索できる
- [ ] 2点間の引用パスを探索できる
- [ ] 共引用・書誌結合に基づく関連論文推薦ができる
- [ ] 最も影響力のある論文をランキングできる
- [ ] Parquet + GraphML形式で永続化できる

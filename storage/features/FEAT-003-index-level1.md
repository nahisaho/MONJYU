# FEAT-003: Index Level 1 (Lazy)

**フィーチャーID**: FEAT-003  
**名称**: インデックス Level 1 (LazyGraphRAG 基盤)  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

LLMコストゼロで名詞句グラフとコミュニティを構築するLevel 1インデックス。LazyGraphRAGの基盤となる軽量グラフ構造を提供する。

### 1.1 スコープ

```
TextUnit[] → NLP処理 → 名詞句グラフ → コミュニティ検出
```

- **入力**: Level 0 の TextUnit[]
- **処理**: NLPによるキーワード/名詞句/NER抽出、グラフ構築、コミュニティ検出
- **出力**: NLP特徴量、名詞句グラフ、コミュニティ
- **コスト**: **LLMコスト $0**

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-IDX-L1-001 | キーワード抽出 | P0 |
| FR-IDX-L1-002 | 名詞句抽出 | P0 |
| FR-IDX-L1-003 | 固有表現認識 | P0 |
| FR-IDX-L1-004 | 名詞句グラフ構築 | P0 |
| FR-IDX-L1-005 | コミュニティ検出 | P0 |

### 1.3 依存関係

- **依存**: FEAT-002 (Index Level 0)
- **被依存**: FEAT-005 (Lazy Search)

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Index Level 1 Builder                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ NLPProcessor     │    │ GraphBuilder     │    │ Community     │ │
│  │                  │    │                  │    │ Detector      │ │
│  │ - extract_kw()   │───▶│ - build()        │───▶│ - detect()    │ │
│  │ - extract_np()   │    │ - add_edge()     │    │ - hierarchy() │ │
│  │ - extract_ner()  │    └──────────────────┘    └───────────────┘ │
│  └──────────────────┘             │                      │          │
│         │                         ▼                      ▼          │
│         ▼                  ┌──────────────┐      ┌───────────────┐ │
│  ┌──────────────────┐      │ NounPhrase   │      │ Community[]   │ │
│  │ NLPFeatures[]    │      │ Graph        │      │               │ │
│  └──────────────────┘      └──────────────┘      └───────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      NLP Libraries                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │ spaCy       │  │ NLTK/RAKE   │  │ NetworkX + Leiden   │  │  │
│  │  │ (NER, POS)  │  │ (Keywords)  │  │ (Graph + Community) │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 出力ディレクトリ構造

```
output/
└── index/
    └── level_1/
        ├── nlp_features.parquet       # NLP抽出結果
        ├── noun_phrase_graph.parquet  # 名詞句共起グラフ
        └── communities_l1.parquet     # NLPベースコミュニティ
```

### 2.3 クラス図

```python
from dataclasses import dataclass, field
from typing import Protocol

# === Protocols ===

class NLPProcessorProtocol(Protocol):
    """NLP処理プロトコル"""
    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]: ...
    def extract_noun_phrases(self, text: str) -> list[str]: ...
    def extract_entities(self, text: str) -> list[tuple[str, str]]: ...  # (entity, type)

class GraphBuilderProtocol(Protocol):
    """グラフビルダープロトコル"""
    def add_node(self, node_id: str, attributes: dict) -> None: ...
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None: ...
    def build_from_cooccurrence(self, documents: list[list[str]], window_size: int = 5) -> None: ...

class CommunityDetectorProtocol(Protocol):
    """コミュニティ検出プロトコル"""
    def detect(self, graph: "Graph", resolution: float = 1.0) -> list["Community"]: ...
    def detect_hierarchical(self, graph: "Graph", levels: int = 3) -> list[list["Community"]]: ...

# === Data Classes ===

@dataclass
class NLPFeatures:
    """NLP抽出特徴量"""
    text_unit_id: str
    keywords: list[str]
    noun_phrases: list[str]
    entities: list[tuple[str, str]]  # (entity, type)
    
    # TF-IDF スコア（オプション）
    keyword_scores: dict[str, float] = field(default_factory=dict)

@dataclass
class NounPhraseNode:
    """名詞句ノード"""
    id: str
    phrase: str
    frequency: int
    document_ids: list[str]
    text_unit_ids: list[str]
    
    # NER情報（該当する場合）
    entity_type: str | None = None

@dataclass
class NounPhraseEdge:
    """名詞句エッジ（共起関係）"""
    source: str
    target: str
    weight: float
    cooccurrence_count: int
    document_ids: list[str]

@dataclass
class Community:
    """コミュニティ"""
    id: str
    level: int
    node_ids: list[str]
    
    # 代表的な名詞句（頻度順）
    representative_phrases: list[str]
    
    # メタデータ
    size: int = 0
    internal_edges: int = 0
    
    # 親コミュニティ（階層構造）
    parent_id: str | None = None

@dataclass
class Level1Index:
    """Level 1 インデックス"""
    nlp_features: list[NLPFeatures]
    nodes: list[NounPhraseNode]
    edges: list[NounPhraseEdge]
    communities: list[Community]
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    @property
    def community_count(self) -> int:
        return len(self.communities)
```

---

## 3. 詳細設計

### 3.1 NLPProcessor

```python
import spacy
from collections import Counter

class NLPProcessor:
    """NLP処理クラス"""
    
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        # 日本語対応
        if "ja" in model:
            self.is_japanese = True
        else:
            self.is_japanese = False
    
    def process(self, text: str) -> NLPFeatures:
        """テキストからNLP特徴量を抽出"""
        doc = self.nlp(text)
        
        return NLPFeatures(
            text_unit_id="",  # 後で設定
            keywords=self.extract_keywords(text),
            noun_phrases=self.extract_noun_phrases(doc),
            entities=self.extract_entities(doc)
        )
    
    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """キーワード抽出（RAKE or TF-IDF）"""
        from rake_nltk import Rake
        
        rake = Rake()
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()[:top_k]
        
        return keywords
    
    def extract_noun_phrases(self, doc) -> list[str]:
        """名詞句抽出"""
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            # 前後の冠詞や決定詞を除去
            phrase = chunk.text.strip()
            if len(phrase) > 2:  # 短すぎるものは除外
                noun_phrases.append(phrase.lower())
        
        return noun_phrases
    
    def extract_entities(self, doc) -> list[tuple[str, str]]:
        """固有表現認識"""
        entities = []
        
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        return entities
    
    def process_batch(
        self,
        text_units: list["TextUnit"]
    ) -> list[NLPFeatures]:
        """バッチ処理"""
        results = []
        
        # spaCyのパイプライン処理
        texts = [tu.text for tu in text_units]
        docs = list(self.nlp.pipe(texts, batch_size=50))
        
        for tu, doc in zip(text_units, docs):
            features = NLPFeatures(
                text_unit_id=tu.id,
                keywords=self.extract_keywords(tu.text),
                noun_phrases=self.extract_noun_phrases(doc),
                entities=self.extract_entities(doc)
            )
            results.append(features)
        
        return results


class AcademicNLPProcessor(NLPProcessor):
    """学術論文向けNLP処理"""
    
    # 学術論文でよく出る固有表現タイプ
    ACADEMIC_ENTITY_TYPES = {
        "PERSON": "RESEARCHER",
        "ORG": "ORGANIZATION",
        "PRODUCT": "MODEL",
        "WORK_OF_ART": "PAPER",
    }
    
    def __init__(self, model: str = "en_core_web_sm"):
        super().__init__(model)
        
        # カスタム辞書（学術用語）
        self.academic_terms = self._load_academic_terms()
    
    def _load_academic_terms(self) -> set[str]:
        """学術用語辞書を読み込み"""
        # 事前定義の学術用語
        return {
            "transformer", "attention", "bert", "gpt", "lstm", "cnn",
            "neural network", "deep learning", "machine learning",
            "natural language processing", "computer vision",
            "reinforcement learning", "supervised learning",
            "unsupervised learning", "transfer learning",
            # ... 拡張可能
        }
    
    def extract_entities(self, doc) -> list[tuple[str, str]]:
        """学術論文向け固有表現認識"""
        entities = []
        
        # spaCyのNER
        for ent in doc.ents:
            entity_type = self.ACADEMIC_ENTITY_TYPES.get(ent.label_, ent.label_)
            entities.append((ent.text, entity_type))
        
        # 学術用語の検出
        text_lower = doc.text.lower()
        for term in self.academic_terms:
            if term in text_lower:
                entities.append((term, "CONCEPT"))
        
        return entities
    
    def extract_methods(self, text: str) -> list[str]:
        """手法名の抽出"""
        # パターンマッチング
        import re
        
        patterns = [
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # CamelCase (BERT, GPT)
            r'\b([A-Z]{2,}(?:-[A-Z0-9]+)*)\b',      # ACRONYM (LSTM, CNN)
        ]
        
        methods = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            methods.extend(matches)
        
        return list(set(methods))
```

### 3.2 GraphBuilder

```python
import networkx as nx
from collections import defaultdict

class NounPhraseGraphBuilder:
    """名詞句グラフビルダー"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_metadata = {}
        self.edge_metadata = {}
    
    def build_from_features(
        self,
        nlp_features_list: list[NLPFeatures],
        text_units: list["TextUnit"],
        window_size: int = 5,
        min_frequency: int = 2
    ) -> tuple[list[NounPhraseNode], list[NounPhraseEdge]]:
        """NLP特徴量から名詞句グラフを構築"""
        
        # 1. 名詞句の頻度と出現場所を集計
        phrase_stats = defaultdict(lambda: {
            "frequency": 0,
            "document_ids": set(),
            "text_unit_ids": set(),
            "entity_type": None
        })
        
        for features, tu in zip(nlp_features_list, text_units):
            for phrase in features.noun_phrases:
                phrase_stats[phrase]["frequency"] += 1
                phrase_stats[phrase]["document_ids"].add(tu.document_id)
                phrase_stats[phrase]["text_unit_ids"].add(tu.id)
            
            for entity, entity_type in features.entities:
                entity_lower = entity.lower()
                if entity_lower in phrase_stats:
                    phrase_stats[entity_lower]["entity_type"] = entity_type
        
        # 2. 低頻度の名詞句をフィルタリング
        filtered_phrases = {
            phrase: stats
            for phrase, stats in phrase_stats.items()
            if stats["frequency"] >= min_frequency
        }
        
        # 3. ノード作成
        nodes = []
        phrase_to_id = {}
        for i, (phrase, stats) in enumerate(filtered_phrases.items()):
            node_id = f"np_{i}"
            phrase_to_id[phrase] = node_id
            
            nodes.append(NounPhraseNode(
                id=node_id,
                phrase=phrase,
                frequency=stats["frequency"],
                document_ids=list(stats["document_ids"]),
                text_unit_ids=list(stats["text_unit_ids"]),
                entity_type=stats["entity_type"]
            ))
            
            self.graph.add_node(node_id, phrase=phrase, **stats)
        
        # 4. 共起関係からエッジ作成
        edges = []
        cooccurrence = defaultdict(lambda: {
            "count": 0,
            "document_ids": set()
        })
        
        for features, tu in zip(nlp_features_list, text_units):
            phrases_in_unit = [
                p for p in features.noun_phrases
                if p in phrase_to_id
            ]
            
            # ウィンドウ内の共起をカウント
            for i, p1 in enumerate(phrases_in_unit):
                for j in range(i + 1, min(i + window_size + 1, len(phrases_in_unit))):
                    p2 = phrases_in_unit[j]
                    if p1 != p2:
                        key = tuple(sorted([p1, p2]))
                        cooccurrence[key]["count"] += 1
                        cooccurrence[key]["document_ids"].add(tu.document_id)
        
        # 5. エッジ作成
        for (p1, p2), stats in cooccurrence.items():
            if stats["count"] >= 1:  # 最低共起回数
                source_id = phrase_to_id[p1]
                target_id = phrase_to_id[p2]
                weight = stats["count"]
                
                edges.append(NounPhraseEdge(
                    source=source_id,
                    target=target_id,
                    weight=weight,
                    cooccurrence_count=stats["count"],
                    document_ids=list(stats["document_ids"])
                ))
                
                self.graph.add_edge(source_id, target_id, weight=weight)
        
        return nodes, edges
    
    def get_networkx_graph(self) -> nx.Graph:
        """NetworkXグラフを取得"""
        return self.graph
```

### 3.3 CommunityDetector

```python
import networkx as nx
from typing import Optional

class LeidenCommunityDetector:
    """Leidenアルゴリズムによるコミュニティ検出"""
    
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
    
    def detect(
        self,
        graph: nx.Graph,
        resolution: Optional[float] = None
    ) -> list[Community]:
        """コミュニティを検出"""
        import leidenalg as la
        import igraph as ig
        
        resolution = resolution or self.resolution
        
        # NetworkX → igraph 変換
        ig_graph = ig.Graph.from_networkx(graph)
        
        # Leiden アルゴリズム実行
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight" if "weight" in graph.edges(data=True) else None
        )
        
        # Community オブジェクト作成
        communities = []
        for i, community_nodes in enumerate(partition):
            node_ids = [ig_graph.vs[n]["_nx_name"] for n in community_nodes]
            
            # 代表的な名詞句を取得
            representative_phrases = self._get_representative_phrases(
                graph, node_ids, top_k=5
            )
            
            communities.append(Community(
                id=f"community_l1_{i}",
                level=1,
                node_ids=node_ids,
                representative_phrases=representative_phrases,
                size=len(node_ids),
                internal_edges=self._count_internal_edges(graph, node_ids)
            ))
        
        return communities
    
    def detect_hierarchical(
        self,
        graph: nx.Graph,
        levels: int = 3,
        resolution_multiplier: float = 2.0
    ) -> list[list[Community]]:
        """階層的コミュニティを検出"""
        all_levels = []
        current_resolution = self.resolution
        
        for level in range(levels):
            communities = self.detect(graph, current_resolution)
            
            # レベル情報を更新
            for comm in communities:
                comm.level = level
                comm.id = f"community_l1_lv{level}_{comm.id.split('_')[-1]}"
            
            all_levels.append(communities)
            
            # 次のレベルは粗い粒度
            current_resolution *= resolution_multiplier
        
        # 親子関係を設定
        self._set_parent_relationships(all_levels)
        
        return all_levels
    
    def _get_representative_phrases(
        self,
        graph: nx.Graph,
        node_ids: list[str],
        top_k: int = 5
    ) -> list[str]:
        """代表的な名詞句を取得（頻度順）"""
        phrases = []
        for node_id in node_ids:
            if node_id in graph.nodes:
                phrase = graph.nodes[node_id].get("phrase", "")
                frequency = graph.nodes[node_id].get("frequency", 0)
                phrases.append((phrase, frequency))
        
        # 頻度でソート
        phrases.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in phrases[:top_k]]
    
    def _count_internal_edges(self, graph: nx.Graph, node_ids: list[str]) -> int:
        """コミュニティ内のエッジ数をカウント"""
        node_set = set(node_ids)
        count = 0
        for u, v in graph.edges():
            if u in node_set and v in node_set:
                count += 1
        return count
    
    def _set_parent_relationships(self, all_levels: list[list[Community]]) -> None:
        """親子関係を設定"""
        for i in range(len(all_levels) - 1):
            child_level = all_levels[i]
            parent_level = all_levels[i + 1]
            
            # 各子コミュニティの最も重複が多い親を見つける
            for child in child_level:
                child_nodes = set(child.node_ids)
                best_parent = None
                best_overlap = 0
                
                for parent in parent_level:
                    parent_nodes = set(parent.node_ids)
                    overlap = len(child_nodes & parent_nodes)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent
                
                if best_parent:
                    child.parent_id = best_parent.id
```

### 3.4 Level1IndexBuilder (Facade)

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Level1BuildResult:
    """Level 1 インデックス構築結果"""
    nlp_features_count: int
    node_count: int
    edge_count: int
    community_count: int
    output_path: Path


class Level1IndexBuilder:
    """Level 1 インデックスビルダー"""
    
    def __init__(
        self,
        nlp_processor: NLPProcessorProtocol | None = None,
        output_path: Path = Path("./output/index/level_1")
    ):
        self.nlp_processor = nlp_processor or AcademicNLPProcessor()
        self.graph_builder = NounPhraseGraphBuilder()
        self.community_detector = LeidenCommunityDetector()
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def build(
        self,
        text_units: list["TextUnit"],
        window_size: int = 5,
        min_frequency: int = 2,
        community_levels: int = 3,
        show_progress: bool = True
    ) -> Level1BuildResult:
        """Level 1 インデックスを構築"""
        
        # 1. NLP特徴量抽出
        nlp_features = self.nlp_processor.process_batch(text_units)
        
        # 2. 名詞句グラフ構築
        nodes, edges = self.graph_builder.build_from_features(
            nlp_features,
            text_units,
            window_size=window_size,
            min_frequency=min_frequency
        )
        
        # 3. コミュニティ検出
        graph = self.graph_builder.get_networkx_graph()
        communities_hierarchy = self.community_detector.detect_hierarchical(
            graph,
            levels=community_levels
        )
        
        # 全レベルのコミュニティをフラット化
        all_communities = []
        for level_communities in communities_hierarchy:
            all_communities.extend(level_communities)
        
        # 4. 永続化
        self._save_nlp_features(nlp_features)
        self._save_graph(nodes, edges)
        self._save_communities(all_communities)
        
        return Level1BuildResult(
            nlp_features_count=len(nlp_features),
            node_count=len(nodes),
            edge_count=len(edges),
            community_count=len(all_communities),
            output_path=self.output_path
        )
    
    def _save_nlp_features(self, features: list[NLPFeatures]) -> None:
        """NLP特徴量を永続化"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        records = []
        for f in features:
            records.append({
                "text_unit_id": f.text_unit_id,
                "keywords": f.keywords,
                "noun_phrases": f.noun_phrases,
                "entities": [{"text": e[0], "type": e[1]} for e in f.entities],
            })
        
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.output_path / "nlp_features.parquet")
    
    def _save_graph(self, nodes: list[NounPhraseNode], edges: list[NounPhraseEdge]) -> None:
        """グラフを永続化"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # ノード
        node_records = []
        for n in nodes:
            node_records.append({
                "id": n.id,
                "phrase": n.phrase,
                "frequency": n.frequency,
                "document_ids": n.document_ids,
                "text_unit_ids": n.text_unit_ids,
                "entity_type": n.entity_type,
            })
        
        table = pa.Table.from_pylist(node_records)
        pq.write_table(table, self.output_path / "noun_phrase_nodes.parquet")
        
        # エッジ
        edge_records = []
        for e in edges:
            edge_records.append({
                "source": e.source,
                "target": e.target,
                "weight": e.weight,
                "cooccurrence_count": e.cooccurrence_count,
                "document_ids": e.document_ids,
            })
        
        table = pa.Table.from_pylist(edge_records)
        pq.write_table(table, self.output_path / "noun_phrase_edges.parquet")
    
    def _save_communities(self, communities: list[Community]) -> None:
        """コミュニティを永続化"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        records = []
        for c in communities:
            records.append({
                "id": c.id,
                "level": c.level,
                "node_ids": c.node_ids,
                "representative_phrases": c.representative_phrases,
                "size": c.size,
                "internal_edges": c.internal_edges,
                "parent_id": c.parent_id,
            })
        
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.output_path / "communities_l1.parquet")
    
    def load(self) -> Level1Index:
        """Level 1 インデックスを読み込み"""
        import pyarrow.parquet as pq
        
        # NLP特徴量
        nlp_table = pq.read_table(self.output_path / "nlp_features.parquet")
        nlp_features = [
            NLPFeatures(
                text_unit_id=r["text_unit_id"],
                keywords=r["keywords"],
                noun_phrases=r["noun_phrases"],
                entities=[(e["text"], e["type"]) for e in r["entities"]]
            )
            for r in nlp_table.to_pylist()
        ]
        
        # ノード
        node_table = pq.read_table(self.output_path / "noun_phrase_nodes.parquet")
        nodes = [
            NounPhraseNode(**r)
            for r in node_table.to_pylist()
        ]
        
        # エッジ
        edge_table = pq.read_table(self.output_path / "noun_phrase_edges.parquet")
        edges = [
            NounPhraseEdge(**r)
            for r in edge_table.to_pylist()
        ]
        
        # コミュニティ
        comm_table = pq.read_table(self.output_path / "communities_l1.parquet")
        communities = [
            Community(**r)
            for r in comm_table.to_pylist()
        ]
        
        return Level1Index(
            nlp_features=nlp_features,
            nodes=nodes,
            edges=edges,
            communities=communities
        )
```

---

## 4. 設定

```yaml
# config/index_level1.yaml

index_level1:
  output_path: ./output/index/level_1
  
  # NLP設定
  nlp:
    model: en_core_web_sm  # en_core_web_sm | en_core_web_md | ja_core_news_sm
    use_academic_processor: true
    
    # キーワード抽出
    keyword_extraction:
      method: rake  # rake | tfidf
      top_k: 10
    
    # 名詞句抽出
    noun_phrase:
      min_length: 2
      max_length: 50
  
  # グラフ構築設定
  graph:
    window_size: 5        # 共起ウィンドウサイズ
    min_frequency: 2      # 最小出現頻度
    min_cooccurrence: 1   # 最小共起回数
  
  # コミュニティ検出設定
  community:
    algorithm: leiden     # leiden | louvain
    resolution: 1.0       # 解像度パラメータ
    levels: 3             # 階層レベル数
    resolution_multiplier: 2.0  # 各レベル間の解像度倍率
```

---

## 5. テスト計画

### 5.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_extract_keywords | NLPProcessor.extract_keywords | キーワードリストを返す |
| test_extract_noun_phrases | NLPProcessor.extract_noun_phrases | 名詞句リストを返す |
| test_extract_entities | NLPProcessor.extract_entities | (entity, type)リストを返す |
| test_build_graph | GraphBuilder.build_from_features | ノード・エッジを返す |
| test_detect_communities | CommunityDetector.detect | コミュニティリストを返す |
| test_hierarchical_communities | CommunityDetector.detect_hierarchical | 階層的コミュニティを返す |

### 5.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_level1_build | Level1IndexBuilder.build | 全アーティファクトを生成 |
| test_level1_load | Level1IndexBuilder.load | 保存データを正しく読み込み |
| test_academic_nlp | AcademicNLPProcessor | 学術用語を認識 |

### 5.3 パフォーマンステスト

| テストケース | 条件 | 期待結果 |
|-------------|------|---------|
| test_nlp_throughput | 1000 TextUnits | > 100 units/sec |
| test_graph_build | 10000 名詞句 | < 1 min |
| test_community_detection | 10000 ノード | < 30 sec |

---

## 6. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-003-01 | NLPProcessor Protocol 定義 | 1h | - |
| TASK-003-02 | NLPProcessor 実装 | 3h | TASK-003-01 |
| TASK-003-03 | AcademicNLPProcessor 実装 | 2h | TASK-003-02 |
| TASK-003-04 | NounPhraseGraphBuilder 実装 | 3h | TASK-003-02 |
| TASK-003-05 | LeidenCommunityDetector 実装 | 3h | TASK-003-04 |
| TASK-003-06 | Level1IndexBuilder 実装 | 2h | TASK-003-01~05 |
| TASK-003-07 | Parquet永続化 実装 | 2h | TASK-003-06 |
| TASK-003-08 | 単体テスト作成 | 2h | TASK-003-01~07 |
| TASK-003-09 | 統合テスト作成 | 2h | TASK-003-08 |
| **合計** | | **20h** | |

---

## 7. 受入基準

- [ ] spaCyを使ってキーワード、名詞句、固有表現を抽出できる
- [ ] 名詞句の共起関係からグラフを構築できる
- [ ] Leidenアルゴリズムで階層的コミュニティを検出できる
- [ ] 全処理がLLMコストゼロで完了する
- [ ] 1000 TextUnitsを1分以内に処理できる
- [ ] Parquet形式で nlp_features, graph, communities を永続化できる
- [ ] 学術論文の専門用語（Transformer, BERT等）を認識できる

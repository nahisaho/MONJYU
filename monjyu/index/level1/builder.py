# Level 1 Index Builder
"""
Builder for Level 1 (LazyGraphRAG foundation) index.

Coordinates NLP processing, graph construction, and community detection.
All processing with ZERO LLM cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from monjyu.graph.base import Community, NounPhraseEdge, NounPhraseNode
from monjyu.graph.community_detector import LeidenCommunityDetector
from monjyu.graph.noun_phrase_graph import NounPhraseGraphBuilder
from monjyu.nlp.base import NLPFeatures
from monjyu.nlp.spacy_processor import SpacyNLPProcessor

if TYPE_CHECKING:
    pass


@dataclass
class Level1IndexConfig:
    """Level 1 インデックス設定
    
    Attributes:
        output_dir: 出力ディレクトリ
        spacy_model: spaCyモデル名
        min_frequency: 最小出現頻度
        window_size: 共起ウィンドウサイズ
        resolution: コミュニティ検出の解像度
        hierarchical_levels: 階層コミュニティのレベル数
        show_progress: 進捗表示
    """
    output_dir: str | Path = "./output/index/level_1"
    
    # NLP設定
    spacy_model: str = "en_core_web_sm"
    academic_mode: bool = True
    
    # グラフ設定
    min_frequency: int = 2
    window_size: int = 5
    
    # コミュニティ検出設定
    resolution: float = 1.0
    hierarchical_levels: int = 3
    
    # 処理設定
    batch_size: int = 50
    show_progress: bool = True


@dataclass
class Level1Index:
    """Level 1 インデックス
    
    Attributes:
        nlp_features: NLP特徴量リスト
        nodes: 名詞句ノードリスト
        edges: 名詞句エッジリスト
        communities: コミュニティリスト
        output_dir: 出力ディレクトリ
    """
    nlp_features: list[NLPFeatures]
    nodes: list[NounPhraseNode]
    edges: list[NounPhraseEdge]
    communities: list[Community]
    output_dir: Path
    
    @property
    def feature_count(self) -> int:
        """NLP特徴量数"""
        return len(self.nlp_features)
    
    @property
    def node_count(self) -> int:
        """ノード数"""
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        """エッジ数"""
        return len(self.edges)
    
    @property
    def community_count(self) -> int:
        """コミュニティ数"""
        return len(self.communities)


class Level1IndexBuilder:
    """Level 1 インデックスビルダー
    
    NLP処理、グラフ構築、コミュニティ検出を統合して
    Level 1インデックスを構築する。
    
    **LLMコスト: $0**
    
    Example:
        >>> config = Level1IndexConfig(
        ...     output_dir="./output/index/level_1",
        ... )
        >>> builder = Level1IndexBuilder(config)
        >>> index = builder.build(text_units)
        >>> print(f"Built index with {index.community_count} communities")
    
    Attributes:
        config: インデックス設定
    """
    
    def __init__(self, config: Level1IndexConfig | None = None) -> None:
        """初期化
        
        Args:
            config: インデックス設定
        """
        self.config = config or Level1IndexConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # コンポーネント初期化（遅延）
        self._nlp_processor: SpacyNLPProcessor | None = None
        self._graph_builder: NounPhraseGraphBuilder | None = None
        self._community_detector: LeidenCommunityDetector | None = None
    
    @property
    def nlp_processor(self) -> SpacyNLPProcessor:
        """NLPプロセッサを取得（遅延初期化）"""
        if self._nlp_processor is None:
            self._nlp_processor = SpacyNLPProcessor(
                model=self.config.spacy_model,
                academic_mode=self.config.academic_mode,
            )
        return self._nlp_processor
    
    @property
    def graph_builder(self) -> NounPhraseGraphBuilder:
        """グラフビルダーを取得（遅延初期化）"""
        if self._graph_builder is None:
            self._graph_builder = NounPhraseGraphBuilder(
                min_frequency=self.config.min_frequency,
                window_size=self.config.window_size,
            )
        return self._graph_builder
    
    @property
    def community_detector(self) -> LeidenCommunityDetector:
        """コミュニティ検出器を取得（遅延初期化）"""
        if self._community_detector is None:
            self._community_detector = LeidenCommunityDetector(
                resolution=self.config.resolution,
            )
        return self._community_detector
    
    def build(
        self,
        text_units: list[Any],
    ) -> Level1Index:
        """Level 1 インデックスを構築
        
        Args:
            text_units: TextUnitリスト
            
        Returns:
            構築されたインデックス
        """
        if self.config.show_progress:
            print(f"Building Level 1 index...")
            print(f"  Text Units: {len(text_units)}")
        
        # 1. NLP処理
        nlp_features = self._process_nlp(text_units)
        
        # 2. グラフ構築
        nodes, edges = self._build_graph(nlp_features, text_units)
        
        # 3. コミュニティ検出
        communities = self._detect_communities()
        
        # 4. Parquet保存
        self._save_to_parquet(nlp_features, nodes, edges, communities)
        
        if self.config.show_progress:
            print(f"Level 1 index built successfully!")
            print(f"  Nodes: {len(nodes)}")
            print(f"  Edges: {len(edges)}")
            print(f"  Communities: {len(communities)}")
        
        return Level1Index(
            nlp_features=nlp_features,
            nodes=nodes,
            edges=edges,
            communities=communities,
            output_dir=self.output_dir,
        )
    
    def _process_nlp(
        self,
        text_units: list[Any],
    ) -> list[NLPFeatures]:
        """NLP処理
        
        Args:
            text_units: TextUnitリスト
            
        Returns:
            NLP特徴量リスト
        """
        if self.config.show_progress:
            print(f"  Processing NLP...")
        
        texts = [getattr(tu, "text", str(tu)) for tu in text_units]
        tu_ids = [getattr(tu, "id", f"tu_{i}") for i, tu in enumerate(text_units)]
        
        features = self.nlp_processor.process_batch(
            texts,
            tu_ids,
            batch_size=self.config.batch_size,
        )
        
        if self.config.show_progress:
            total_keywords = sum(len(f.keywords) for f in features)
            total_np = sum(len(f.noun_phrases) for f in features)
            total_entities = sum(len(f.entities) for f in features)
            print(f"    Keywords: {total_keywords}")
            print(f"    Noun Phrases: {total_np}")
            print(f"    Entities: {total_entities}")
        
        return features
    
    def _build_graph(
        self,
        nlp_features: list[NLPFeatures],
        text_units: list[Any],
    ) -> tuple[list[NounPhraseNode], list[NounPhraseEdge]]:
        """グラフ構築
        
        Args:
            nlp_features: NLP特徴量リスト
            text_units: TextUnitリスト
            
        Returns:
            (ノードリスト, エッジリスト)
        """
        if self.config.show_progress:
            print(f"  Building noun phrase graph...")
        
        nodes, edges = self.graph_builder.build_from_features(
            nlp_features,
            text_units,
            window_size=self.config.window_size,
        )
        
        if self.config.show_progress:
            print(f"    Nodes: {len(nodes)}")
            print(f"    Edges: {len(edges)}")
        
        return nodes, edges
    
    def _detect_communities(self) -> list[Community]:
        """コミュニティ検出
        
        Returns:
            コミュニティリスト
        """
        if self.config.show_progress:
            print(f"  Detecting communities...")
        
        graph = self.graph_builder.get_networkx_graph()
        
        if graph.number_of_nodes() == 0:
            return []
        
        # 階層的コミュニティ検出
        if self.config.hierarchical_levels > 1:
            all_levels = self.community_detector.detect_hierarchical(
                graph,
                levels=self.config.hierarchical_levels,
            )
            # 全レベルをフラットに
            communities = []
            for level_communities in all_levels:
                communities.extend(level_communities)
        else:
            communities = self.community_detector.detect(graph)
        
        if self.config.show_progress:
            print(f"    Communities: {len(communities)}")
        
        return communities
    
    def _save_to_parquet(
        self,
        nlp_features: list[NLPFeatures],
        nodes: list[NounPhraseNode],
        edges: list[NounPhraseEdge],
        communities: list[Community],
    ) -> None:
        """Parquetに保存
        
        Args:
            nlp_features: NLP特徴量リスト
            nodes: ノードリスト
            edges: エッジリスト
            communities: コミュニティリスト
        """
        if self.config.show_progress:
            print(f"  Saving to Parquet...")
        
        # NLP特徴量
        self._save_nlp_features(nlp_features)
        
        # グラフ（ノードとエッジ）
        self._save_graph(nodes, edges)
        
        # コミュニティ
        self._save_communities(communities)
    
    def _save_nlp_features(self, features: list[NLPFeatures]) -> None:
        """NLP特徴量を保存"""
        if not features:
            return
        
        records = []
        for f in features:
            records.append({
                "text_unit_id": f.text_unit_id,
                "keywords": f.keywords,
                "noun_phrases": f.noun_phrases,
                "entities_text": [e[0] for e in f.entities],
                "entities_type": [e[1] for e in f.entities],
            })
        
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.output_dir / "nlp_features.parquet")
    
    def _save_graph(
        self,
        nodes: list[NounPhraseNode],
        edges: list[NounPhraseEdge],
    ) -> None:
        """グラフを保存"""
        # ノード
        if nodes:
            node_records = [n.to_dict() for n in nodes]
            table = pa.Table.from_pylist(node_records)
            pq.write_table(table, self.output_dir / "noun_phrase_nodes.parquet")
        
        # エッジ
        if edges:
            edge_records = [e.to_dict() for e in edges]
            table = pa.Table.from_pylist(edge_records)
            pq.write_table(table, self.output_dir / "noun_phrase_edges.parquet")
    
    def _save_communities(self, communities: list[Community]) -> None:
        """コミュニティを保存"""
        if not communities:
            return
        
        records = [c.to_dict() for c in communities]
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.output_dir / "communities_l1.parquet")
    
    def load(self) -> Level1Index | None:
        """既存インデックスを読み込み
        
        Returns:
            読み込んだインデックス（存在しない場合はNone）
        """
        nlp_features_path = self.output_dir / "nlp_features.parquet"
        if not nlp_features_path.exists():
            return None
        
        # NLP特徴量
        nlp_features = self._load_nlp_features()
        
        # グラフ
        nodes = self._load_nodes()
        edges = self._load_edges()
        
        # コミュニティ
        communities = self._load_communities()
        
        return Level1Index(
            nlp_features=nlp_features,
            nodes=nodes,
            edges=edges,
            communities=communities,
            output_dir=self.output_dir,
        )
    
    def _load_nlp_features(self) -> list[NLPFeatures]:
        """NLP特徴量を読み込み"""
        path = self.output_dir / "nlp_features.parquet"
        if not path.exists():
            return []
        
        table = pq.read_table(path)
        records = table.to_pylist()
        
        features = []
        for r in records:
            entities = list(zip(
                r.get("entities_text", []),
                r.get("entities_type", []),
            ))
            
            features.append(NLPFeatures(
                text_unit_id=r["text_unit_id"],
                keywords=r.get("keywords", []),
                noun_phrases=r.get("noun_phrases", []),
                entities=entities,
            ))
        
        return features
    
    def _load_nodes(self) -> list[NounPhraseNode]:
        """ノードを読み込み"""
        path = self.output_dir / "noun_phrase_nodes.parquet"
        if not path.exists():
            return []
        
        table = pq.read_table(path)
        records = table.to_pylist()
        
        return [NounPhraseNode.from_dict(r) for r in records]
    
    def _load_edges(self) -> list[NounPhraseEdge]:
        """エッジを読み込み"""
        path = self.output_dir / "noun_phrase_edges.parquet"
        if not path.exists():
            return []
        
        table = pq.read_table(path)
        records = table.to_pylist()
        
        return [NounPhraseEdge.from_dict(r) for r in records]
    
    def _load_communities(self) -> list[Community]:
        """コミュニティを読み込み"""
        path = self.output_dir / "communities_l1.parquet"
        if not path.exists():
            return []
        
        table = pq.read_table(path)
        records = table.to_pylist()
        
        return [Community.from_dict(r) for r in records]
    
    def get_stats(self) -> dict[str, Any]:
        """インデックスの統計情報を取得
        
        Returns:
            統計情報
        """
        stats = {
            "nlp_features_count": 0,
            "nodes_count": 0,
            "edges_count": 0,
            "communities_count": 0,
        }
        
        # ファイルサイズも追加
        for name, file in [
            ("nlp_features", "nlp_features.parquet"),
            ("nodes", "noun_phrase_nodes.parquet"),
            ("edges", "noun_phrase_edges.parquet"),
            ("communities", "communities_l1.parquet"),
        ]:
            path = self.output_dir / file
            if path.exists():
                table = pq.read_table(path)
                stats[f"{name}_count"] = table.num_rows
        
        return stats

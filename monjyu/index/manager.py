# Progressive Index Manager
"""
FEAT-015: Progressive Index Manager

段階的インデックス構築を管理するマネージャー。
Level 0 から Level 4 まで順次構築をオーケストレーション。

Index Levels:
- Level 0: Baseline RAG (Chunks + Embeddings)
- Level 1: LazyGraphRAG (NLP Graph + Communities)
- Level 2: Entity Graph (LLM Entity Extraction)
- Level 3: Community Reports (LLM Report Generation)
- Level 4: Pre-extracted Claims (LLM Claim Extraction)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from monjyu.document.models import AcademicPaperDocument, TextUnit
    from monjyu.embedding.base import EmbeddingClientProtocol
    from monjyu.index.base import VectorIndexerProtocol

logger = logging.getLogger(__name__)


class IndexLevel(IntEnum):
    """インデックスレベル
    
    各レベルは前のレベルに依存し、段階的に構築される。
    """
    RAW = 0       # Baseline RAG: Chunks + Embeddings
    LAZY = 1      # LazyGraphRAG: NLP Graph + Communities
    PARTIAL = 2   # GraphRAG: LLM Entities + Relationships
    FULL = 3      # GraphRAG: Community Reports
    ENHANCED = 4  # Pre-extracted Claims


@dataclass
class LevelStatus:
    """レベルステータス
    
    Attributes:
        level: インデックスレベル
        is_built: 構築済みか
        built_at: 構築日時 (ISO 8601)
        document_count: ドキュメント数
        chunk_count: チャンク数
        stats: レベル固有の統計
    """
    level: IndexLevel
    is_built: bool = False
    built_at: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "level": self.level.value,
            "level_name": self.level.name,
            "is_built": self.is_built,
            "built_at": self.built_at,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "stats": self.stats,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LevelStatus:
        """辞書から作成"""
        return cls(
            level=IndexLevel(data["level"]),
            is_built=data.get("is_built", False),
            built_at=data.get("built_at"),
            document_count=data.get("document_count", 0),
            chunk_count=data.get("chunk_count", 0),
            stats=data.get("stats", {}),
        )


@dataclass
class ProgressiveIndexState:
    """Progressive インデックス状態
    
    Attributes:
        id: インデックスID
        name: インデックス名
        current_level: 現在のレベル
        level_status: 各レベルのステータス
        output_dir: 出力ディレクトリ
        created_at: 作成日時
        updated_at: 更新日時
    """
    id: str
    name: str
    current_level: IndexLevel
    level_status: dict[IndexLevel, LevelStatus]
    output_dir: Path
    created_at: str
    updated_at: str
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "name": self.name,
            "current_level": self.current_level.value,
            "level_status": {
                level.value: status.to_dict()
                for level, status in self.level_status.items()
            },
            "output_dir": str(self.output_dir),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressiveIndexState:
        """辞書から作成"""
        return cls(
            id=data["id"],
            name=data["name"],
            current_level=IndexLevel(data["current_level"]),
            level_status={
                IndexLevel(int(k)): LevelStatus.from_dict(v)
                for k, v in data.get("level_status", {}).items()
            },
            output_dir=Path(data["output_dir"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
    
    def save(self, path: Path | None = None) -> None:
        """状態を保存"""
        save_path = path or (self.output_dir / "state.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> ProgressiveIndexState:
        """状態を読み込み"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_level_status(self, level: IndexLevel) -> LevelStatus:
        """レベルステータスを取得"""
        if level not in self.level_status:
            self.level_status[level] = LevelStatus(level=level)
        return self.level_status[level]
    
    def is_level_built(self, level: IndexLevel) -> bool:
        """指定レベルが構築済みか"""
        return self.get_level_status(level).is_built


@dataclass
class ProgressiveIndexConfig:
    """Progressive インデックス設定
    
    Attributes:
        output_dir: 出力ベースディレクトリ
        index_name: インデックス名
        
        # Level 0 設定
        embedding_strategy: 埋め込み戦略
        index_strategy: インデックス戦略
        ollama_model: Ollamaモデル名
        
        # Level 1 設定
        spacy_model: spaCyモデル名
        
        # Level 2-4 設定
        llm_model: LLMモデル名
        
        # 共通設定
        batch_size: バッチサイズ
        show_progress: 進捗表示
    """
    output_dir: str | Path = "./output/progressive_index"
    index_name: str = "monjyu_index"
    
    # Level 0
    embedding_strategy: str = "ollama"
    index_strategy: str = "lancedb"
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    azure_openai_deployment: str | None = None
    azure_search_endpoint: str | None = None
    
    # Level 1
    spacy_model: str = "en_core_web_sm"
    min_frequency: int = 2
    resolution: float = 1.0
    
    # Level 2-4
    llm_model: str = "gpt-4o-mini"
    
    # Common
    batch_size: int = 50
    show_progress: bool = True


class ProgressiveIndexManager:
    """Progressive インデックスマネージャー
    
    段階的にインデックスを構築・管理する。
    
    Example:
        >>> config = ProgressiveIndexConfig(
        ...     output_dir="./output/index",
        ...     index_name="my_index",
        ... )
        >>> manager = ProgressiveIndexManager(config)
        >>> 
        >>> # Level 0 構築
        >>> state = await manager.build_level(IndexLevel.RAW, documents, text_units)
        >>> 
        >>> # Level 1 まで構築
        >>> state = await manager.build_to_level(IndexLevel.LAZY)
        >>> 
        >>> # 現在の状態確認
        >>> print(f"Current level: {state.current_level.name}")
    """
    
    def __init__(self, config: ProgressiveIndexConfig | None = None) -> None:
        """初期化
        
        Args:
            config: インデックス設定
        """
        self.config = config or ProgressiveIndexConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 状態
        self._state: ProgressiveIndexState | None = None
        
        # ビルダー（遅延初期化）
        self._level0_builder = None
        self._level1_builder = None
        self._entity_extractor = None
        self._relationship_extractor = None
        self._community_report_generator = None
        self._claim_extractor = None
    
    @property
    def state(self) -> ProgressiveIndexState:
        """現在の状態を取得"""
        if self._state is None:
            self._state = self._load_or_create_state()
        return self._state
    
    def _load_or_create_state(self) -> ProgressiveIndexState:
        """状態を読み込みまたは作成"""
        state_path = self.output_dir / "state.json"
        
        if state_path.exists():
            logger.info(f"Loading existing state from {state_path}")
            return ProgressiveIndexState.load(state_path)
        
        # 新規作成
        import uuid
        now = datetime.now(timezone.utc).isoformat()
        
        state = ProgressiveIndexState(
            id=str(uuid.uuid4()),
            name=self.config.index_name,
            current_level=IndexLevel.RAW,
            level_status={},
            output_dir=self.output_dir,
            created_at=now,
            updated_at=now,
        )
        
        # 各レベルの初期ステータスを設定
        for level in IndexLevel:
            state.level_status[level] = LevelStatus(level=level)
        
        state.save()
        logger.info(f"Created new state: {state.id}")
        return state
    
    def _get_level0_builder(self):
        """Level 0 ビルダーを取得"""
        if self._level0_builder is None:
            from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
            
            config = Level0IndexConfig(
                output_dir=self.output_dir / "level_0",
                embedding_strategy=self.config.embedding_strategy,
                index_strategy=self.config.index_strategy,
                ollama_model=self.config.ollama_model,
                ollama_base_url=self.config.ollama_base_url,
                batch_size=self.config.batch_size,
                show_progress=self.config.show_progress,
            )
            self._level0_builder = Level0IndexBuilder(config)
        return self._level0_builder
    
    def _get_level1_builder(self):
        """Level 1 ビルダーを取得"""
        if self._level1_builder is None:
            from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
            
            config = Level1IndexConfig(
                output_dir=self.output_dir / "level_1",
                spacy_model=self.config.spacy_model,
                min_frequency=self.config.min_frequency,
                resolution=self.config.resolution,
                batch_size=self.config.batch_size,
                show_progress=self.config.show_progress,
            )
            self._level1_builder = Level1IndexBuilder(config)
        return self._level1_builder
    
    def _get_entity_extractor(self):
        """Entity Extractor を取得"""
        if self._entity_extractor is None:
            from monjyu.index.entity_extractor.llm_extractor import LLMEntityExtractor
            self._entity_extractor = LLMEntityExtractor()
        return self._entity_extractor
    
    def _get_relationship_extractor(self):
        """Relationship Extractor を取得"""
        if self._relationship_extractor is None:
            from monjyu.index.relationship_extractor.llm_extractor import LLMRelationshipExtractor
            self._relationship_extractor = LLMRelationshipExtractor()
        return self._relationship_extractor
    
    def _get_community_report_generator(self):
        """Community Report Generator を取得"""
        if self._community_report_generator is None:
            from monjyu.index.community_report_generator.generator import CommunityReportGenerator
            self._community_report_generator = CommunityReportGenerator()
        return self._community_report_generator
    
    def _get_claim_extractor(self):
        """Claim Extractor を取得"""
        if self._claim_extractor is None:
            from monjyu.lazy.claim_extractor import ClaimExtractor
            self._claim_extractor = ClaimExtractor()
        return self._claim_extractor
    
    async def build_level(
        self,
        level: IndexLevel,
        documents: list["AcademicPaperDocument"] | None = None,
        text_units: list["TextUnit"] | None = None,
        force: bool = False,
    ) -> ProgressiveIndexState:
        """指定レベルを構築
        
        Args:
            level: 構築するレベル
            documents: ドキュメント（Level 0 構築時に必要）
            text_units: TextUnit（Level 0 構築時に必要）
            force: 既存を上書きするか
            
        Returns:
            更新された状態
        """
        state = self.state
        
        # 既に構築済みの場合
        if state.is_level_built(level) and not force:
            logger.info(f"Level {level.name} is already built. Use force=True to rebuild.")
            return state
        
        # 前提レベルのチェック
        if level > IndexLevel.RAW:
            prev_level = IndexLevel(level - 1)
            if not state.is_level_built(prev_level):
                msg = f"Level {prev_level.name} must be built before {level.name}"
                raise ValueError(msg)
        
        logger.info(f"Building Level {level.name}...")
        
        # レベル別の構築
        if level == IndexLevel.RAW:
            await self._build_level_0(documents, text_units)
        elif level == IndexLevel.LAZY:
            await self._build_level_1()
        elif level == IndexLevel.PARTIAL:
            await self._build_level_2()
        elif level == IndexLevel.FULL:
            await self._build_level_3()
        elif level == IndexLevel.ENHANCED:
            await self._build_level_4()
        
        # ステータス更新
        now = datetime.now(timezone.utc).isoformat()
        status = state.get_level_status(level)
        status.is_built = True
        status.built_at = now
        
        if documents:
            status.document_count = len(documents)
        if text_units:
            status.chunk_count = len(text_units)
        
        state.current_level = level
        state.updated_at = now
        state.save()
        
        logger.info(f"Level {level.name} built successfully")
        return state
    
    async def build_to_level(
        self,
        target_level: IndexLevel,
        documents: list["AcademicPaperDocument"] | None = None,
        text_units: list["TextUnit"] | None = None,
        force: bool = False,
    ) -> ProgressiveIndexState:
        """目標レベルまで順次構築
        
        Args:
            target_level: 目標レベル
            documents: ドキュメント（Level 0 構築時に必要）
            text_units: TextUnit（Level 0 構築時に必要）
            force: 既存を上書きするか
            
        Returns:
            更新された状態
        """
        state = self.state
        
        # 現在のレベルから目標まで順次構築
        start_level = 0 if force else (state.current_level.value + 1)
        
        for level_value in range(start_level, target_level.value + 1):
            level = IndexLevel(level_value)
            
            # Level 0 はドキュメントが必要
            if level == IndexLevel.RAW:
                if documents is None or text_units is None:
                    msg = "documents and text_units are required for Level 0"
                    raise ValueError(msg)
                await self.build_level(level, documents, text_units, force)
            else:
                await self.build_level(level, force=force)
        
        return self.state
    
    async def _build_level_0(
        self,
        documents: list["AcademicPaperDocument"] | None,
        text_units: list["TextUnit"] | None,
    ) -> None:
        """Level 0 を構築"""
        if documents is None or text_units is None:
            msg = "documents and text_units are required for Level 0"
            raise ValueError(msg)
        
        builder = self._get_level0_builder()
        index = await builder.build(documents, text_units)
        
        # 統計を更新
        status = self.state.get_level_status(IndexLevel.RAW)
        status.document_count = index.document_count
        status.chunk_count = index.text_unit_count
        status.stats = {
            "embedding_model": index.embedding_model,
            "embedding_dimensions": index.embedding_dimensions,
            "embedding_count": index.embedding_count,
        }
    
    async def _build_level_1(self) -> None:
        """Level 1 を構築"""
        # Level 0 のデータを読み込み
        level0_dir = self.output_dir / "level_0"
        
        # TextUnit を読み込み
        from monjyu.storage.parquet import ParquetStorage
        storage = ParquetStorage(level0_dir)
        text_units = storage.load_text_units()
        
        builder = self._get_level1_builder()
        index = await builder.build(text_units)
        
        # 統計を更新
        status = self.state.get_level_status(IndexLevel.LAZY)
        status.chunk_count = len(text_units)
        status.stats = {
            "node_count": index.node_count,
            "edge_count": index.edge_count,
            "community_count": index.community_count,
        }
    
    async def _build_level_2(self) -> None:
        """Level 2 を構築 (Entity Extraction)"""
        # Level 0 のデータを読み込み
        level0_dir = self.output_dir / "level_0"
        
        from monjyu.storage.parquet import ParquetStorage
        storage = ParquetStorage(level0_dir)
        text_units = storage.load_text_units()
        
        # エンティティ抽出
        extractor = self._get_entity_extractor()
        entities = await extractor.extract_batch([tu.content for tu in text_units])
        
        # 関係抽出
        rel_extractor = self._get_relationship_extractor()
        relationships = await rel_extractor.extract_batch([tu.content for tu in text_units])
        
        # 保存
        level2_dir = self.output_dir / "level_2"
        level2_dir.mkdir(parents=True, exist_ok=True)
        
        with open(level2_dir / "entities.json", "w", encoding="utf-8") as f:
            json.dump([e.to_dict() if hasattr(e, 'to_dict') else e.__dict__ for e in entities], f, indent=2, ensure_ascii=False)
        
        with open(level2_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() if hasattr(r, 'to_dict') else r.__dict__ for r in relationships], f, indent=2, ensure_ascii=False)
        
        # 統計を更新
        status = self.state.get_level_status(IndexLevel.PARTIAL)
        status.stats = {
            "entity_count": len(entities),
            "relationship_count": len(relationships),
        }
    
    async def _build_level_3(self) -> None:
        """Level 3 を構築 (Community Reports)"""
        # Level 2 のデータを読み込み
        level2_dir = self.output_dir / "level_2"
        
        with open(level2_dir / "entities.json", encoding="utf-8") as f:
            entities_data = json.load(f)
        
        with open(level2_dir / "relationships.json", encoding="utf-8") as f:
            relationships_data = json.load(f)
        
        # コミュニティレポート生成
        generator = self._get_community_report_generator()
        reports = await generator.generate_from_entities(entities_data, relationships_data)
        
        # 保存
        level3_dir = self.output_dir / "level_3"
        level3_dir.mkdir(parents=True, exist_ok=True)
        
        with open(level3_dir / "community_reports.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() if hasattr(r, 'to_dict') else r for r in reports], f, indent=2, ensure_ascii=False)
        
        # 統計を更新
        status = self.state.get_level_status(IndexLevel.FULL)
        status.stats = {
            "report_count": len(reports),
        }
    
    async def _build_level_4(self) -> None:
        """Level 4 を構築 (Pre-extracted Claims)"""
        # Level 0 のデータを読み込み
        level0_dir = self.output_dir / "level_0"
        
        from monjyu.storage.parquet import ParquetStorage
        storage = ParquetStorage(level0_dir)
        text_units = storage.load_text_units()
        
        # クレーム抽出
        extractor = self._get_claim_extractor()
        claims = await extractor.extract_batch([tu.content for tu in text_units])
        
        # 保存
        level4_dir = self.output_dir / "level_4"
        level4_dir.mkdir(parents=True, exist_ok=True)
        
        with open(level4_dir / "claims.json", "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2, ensure_ascii=False)
        
        # 統計を更新
        status = self.state.get_level_status(IndexLevel.ENHANCED)
        status.stats = {
            "claim_count": len(claims),
        }
    
    def get_status_summary(self) -> dict[str, Any]:
        """ステータスサマリーを取得"""
        state = self.state
        
        levels_summary = []
        for level in IndexLevel:
            status = state.get_level_status(level)
            levels_summary.append({
                "level": level.value,
                "name": level.name,
                "is_built": status.is_built,
                "built_at": status.built_at,
                "stats": status.stats,
            })
        
        return {
            "id": state.id,
            "name": state.name,
            "current_level": state.current_level.name,
            "levels": levels_summary,
            "output_dir": str(state.output_dir),
        }


def create_progressive_index_manager(
    output_dir: str | Path = "./output/progressive_index",
    index_name: str = "monjyu_index",
    **kwargs,
) -> ProgressiveIndexManager:
    """ProgressiveIndexManager を作成するファクトリ関数
    
    Args:
        output_dir: 出力ディレクトリ
        index_name: インデックス名
        **kwargs: その他の設定
        
    Returns:
        ProgressiveIndexManager インスタンス
    """
    config = ProgressiveIndexConfig(
        output_dir=output_dir,
        index_name=index_name,
        **kwargs,
    )
    return ProgressiveIndexManager(config)

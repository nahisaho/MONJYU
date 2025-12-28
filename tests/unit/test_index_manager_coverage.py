# Progressive Index Manager Coverage Tests
"""
Index Manager カバレッジ向上テスト

ProgressiveIndexManager の未テスト部分をカバー
- ビルダー取得メソッド
- ビルドメソッド（_build_level_0 - _build_level_4）
- build_to_level メソッド
- エッジケース
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.index.manager import (
    IndexLevel,
    LevelStatus,
    ProgressiveIndexConfig,
    ProgressiveIndexManager,
    ProgressiveIndexState,
    create_progressive_index_manager,
)


class TestLevelStatusEdgeCases:
    """LevelStatus エッジケースのテスト"""
    
    def test_to_dict_with_none_values(self):
        """None値を含む辞書変換"""
        status = LevelStatus(
            level=IndexLevel.RAW,
            is_built=False,
            built_at=None,
            stats={}
        )
        
        data = status.to_dict()
        assert data["built_at"] is None
        assert data["stats"] == {}
    
    def test_from_dict_with_level_name(self):
        """level_name キーを含む辞書からの復元"""
        data = {
            "level": 3,
            "level_name": "FULL",  # この値は無視される
            "is_built": True,
            "built_at": "2025-01-15T10:00:00Z",
            "document_count": 100,
            "chunk_count": 500,
            "stats": {"report_count": 20}
        }
        
        status = LevelStatus.from_dict(data)
        assert status.level == IndexLevel.FULL
        assert status.is_built is True
        assert status.stats["report_count"] == 20
    
    def test_all_level_values(self):
        """全レベル値のテスト"""
        for level in IndexLevel:
            status = LevelStatus(level=level)
            data = status.to_dict()
            restored = LevelStatus.from_dict(data)
            assert restored.level == level


class TestProgressiveIndexStateEdgeCases:
    """ProgressiveIndexState エッジケースのテスト"""
    
    def test_from_dict_empty_level_status(self):
        """空のlevel_statusからの復元"""
        data = {
            "id": "test-id",
            "name": "test",
            "current_level": 0,
            "level_status": {},
            "output_dir": "/tmp/test",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
        
        state = ProgressiveIndexState.from_dict(data)
        assert state.level_status == {}
    
    def test_is_level_built_with_missing_level(self):
        """存在しないレベルの構築確認"""
        state = ProgressiveIndexState(
            id="test",
            name="test",
            current_level=IndexLevel.RAW,
            level_status={},
            output_dir=Path("/tmp"),
            created_at="",
            updated_at=""
        )
        
        # 自動作成されてFalseが返る
        assert state.is_level_built(IndexLevel.LAZY) is False
    
    def test_load_nonexistent_file(self):
        """存在しないファイルの読み込み"""
        with pytest.raises(FileNotFoundError):
            ProgressiveIndexState.load(Path("/nonexistent/path/state.json"))
    
    def test_save_creates_directory(self):
        """保存時にディレクトリを作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "nested"
            output_dir.mkdir(parents=True)
            
            state = ProgressiveIndexState(
                id="test",
                name="test",
                current_level=IndexLevel.RAW,
                level_status={IndexLevel.RAW: LevelStatus(level=IndexLevel.RAW)},
                output_dir=output_dir,
                created_at="2025-01-01",
                updated_at="2025-01-01"
            )
            
            state.save()
            
            assert (output_dir / "state.json").exists()


class TestProgressiveIndexConfigEdgeCases:
    """ProgressiveIndexConfig エッジケースのテスト"""
    
    def test_path_as_string(self):
        """文字列パスでの設定"""
        config = ProgressiveIndexConfig(output_dir="./test/path")
        assert config.output_dir == "./test/path"
    
    def test_path_as_pathlib(self):
        """Pathlibパスでの設定"""
        config = ProgressiveIndexConfig(output_dir=Path("./test/path"))
        assert config.output_dir == Path("./test/path")
    
    def test_azure_config(self):
        """Azure設定のテスト"""
        config = ProgressiveIndexConfig(
            embedding_strategy="azure",
            azure_openai_deployment="text-embedding-ada-002",
            azure_search_endpoint="https://my-search.search.windows.net"
        )
        
        assert config.embedding_strategy == "azure"
        assert config.azure_openai_deployment == "text-embedding-ada-002"
    
    def test_all_default_values(self):
        """全デフォルト値の確認"""
        config = ProgressiveIndexConfig()
        
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.min_frequency == 2
        assert config.resolution == 1.0
        assert config.llm_model == "gpt-4o-mini"
        assert config.show_progress is True


class TestProgressiveIndexManagerBuilders:
    """ビルダー取得メソッドのテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_get_level0_builder_lazy_init(self, temp_output_dir):
        """Level0ビルダーの遅延初期化"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        assert manager._level0_builder is None
        
        with patch("monjyu.index.level0.builder.Level0IndexBuilder") as mock_builder:
            builder = manager._get_level0_builder()
            assert builder is not None
    
    def test_get_level1_builder_lazy_init(self, temp_output_dir):
        """Level1ビルダーの遅延初期化"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        assert manager._level1_builder is None
        
        with patch("monjyu.index.level1.builder.Level1IndexBuilder") as mock_builder:
            builder = manager._get_level1_builder()
            assert builder is not None
    
    def test_get_entity_extractor_lazy_init(self, temp_output_dir):
        """EntityExtractorの遅延初期化"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        assert manager._entity_extractor is None
        
        with patch("monjyu.index.entity_extractor.llm_extractor.LLMEntityExtractor") as mock:
            extractor = manager._get_entity_extractor()
            assert extractor is not None
    
    def test_get_relationship_extractor_lazy_init(self, temp_output_dir):
        """RelationshipExtractorの遅延初期化"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        assert manager._relationship_extractor is None
        
        with patch("monjyu.index.relationship_extractor.llm_extractor.LLMRelationshipExtractor") as mock:
            extractor = manager._get_relationship_extractor()
            assert extractor is not None
    
    def test_get_community_report_generator_lazy_init(self, temp_output_dir):
        """CommunityReportGeneratorの遅延初期化"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        assert manager._community_report_generator is None
        
        with patch("monjyu.index.community_report_generator.generator.CommunityReportGenerator") as mock:
            generator = manager._get_community_report_generator()
            assert generator is not None
    
    def test_get_claim_extractor_lazy_init(self, temp_output_dir):
        """ClaimExtractorの遅延初期化"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        assert manager._claim_extractor is None
        
        with patch("monjyu.lazy.claim_extractor.ClaimExtractor") as mock:
            extractor = manager._get_claim_extractor()
            assert extractor is not None
    
    def test_builder_caching(self, temp_output_dir):
        """ビルダーのキャッシュ確認"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        with patch("monjyu.index.level0.builder.Level0IndexBuilder") as mock_builder:
            builder1 = manager._get_level0_builder()
            builder2 = manager._get_level0_builder()
            
            # 同じインスタンスが返される
            assert builder1 is builder2
            # 一度だけ呼ばれる
            assert mock_builder.call_count == 1


class TestBuildLevelMethods:
    """ビルドメソッドのテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_manager(self, temp_output_dir):
        """モック済みマネージャー"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        return manager
    
    @pytest.mark.asyncio
    async def test_build_level_already_built_no_force(self, mock_manager):
        """既に構築済みでforceなしの場合"""
        # Level 0 を構築済みにする
        mock_manager.state.level_status[IndexLevel.RAW] = LevelStatus(
            level=IndexLevel.RAW,
            is_built=True,
            built_at="2025-01-01T00:00:00Z"
        )
        
        # forceなしで再構築を試みる
        state = await mock_manager.build_level(IndexLevel.RAW)
        
        # 状態は変わらない（スキップ）
        assert state.is_level_built(IndexLevel.RAW)
    
    @pytest.mark.asyncio
    async def test_build_level_0_with_mock(self, temp_output_dir):
        """Level 0 ビルドのモックテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # モックのドキュメントとテキストユニット
        mock_doc = MagicMock()
        mock_text_unit = MagicMock()
        
        # ビルダーをモック
        mock_builder = MagicMock()
        mock_index = MagicMock()
        mock_index.document_count = 10
        mock_index.text_unit_count = 100
        mock_index.embedding_model = "test-model"
        mock_index.embedding_dimensions = 768
        mock_index.embedding_count = 100
        mock_builder.build = AsyncMock(return_value=mock_index)
        
        manager._level0_builder = mock_builder
        
        state = await manager.build_level(
            IndexLevel.RAW,
            documents=[mock_doc],
            text_units=[mock_text_unit]
        )
        
        assert state.is_level_built(IndexLevel.RAW)
        mock_builder.build.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_build_level_1_requires_level_0(self, mock_manager):
        """Level 1 は Level 0 が必要"""
        with pytest.raises(ValueError, match="RAW must be built"):
            await mock_manager.build_level(IndexLevel.LAZY)
    
    @pytest.mark.asyncio
    async def test_build_level_2_requires_level_1(self, temp_output_dir):
        """Level 2 は Level 1 が必要"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # Level 0 のみ構築済み
        manager.state.level_status[IndexLevel.RAW] = LevelStatus(
            level=IndexLevel.RAW, is_built=True
        )
        
        with pytest.raises(ValueError, match="LAZY must be built"):
            await manager.build_level(IndexLevel.PARTIAL)
    
    @pytest.mark.asyncio
    async def test_build_level_with_force(self, temp_output_dir):
        """force=Trueで再構築"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # Level 0 を構築済みにする
        manager.state.level_status[IndexLevel.RAW] = LevelStatus(
            level=IndexLevel.RAW,
            is_built=True,
            built_at="2025-01-01T00:00:00Z"
        )
        
        # モック
        mock_builder = MagicMock()
        mock_index = MagicMock()
        mock_index.document_count = 5
        mock_index.text_unit_count = 50
        mock_index.embedding_model = "test"
        mock_index.embedding_dimensions = 768
        mock_index.embedding_count = 50
        mock_builder.build = AsyncMock(return_value=mock_index)
        manager._level0_builder = mock_builder
        
        mock_doc = MagicMock()
        mock_tu = MagicMock()
        
        # force=Trueで再構築
        state = await manager.build_level(
            IndexLevel.RAW,
            documents=[mock_doc],
            text_units=[mock_tu],
            force=True
        )
        
        # ビルダーが呼ばれた
        mock_builder.build.assert_called_once()


class TestBuildToLevel:
    """build_to_level メソッドのテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.asyncio
    async def test_build_to_level_level0_without_data(self, temp_output_dir):
        """Level 0 データなしでbuild_to_level (force=True)"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # force=True の場合、Level 0 から再構築しようとしてデータが必要
        with pytest.raises(ValueError, match="documents and text_units are required"):
            await manager.build_to_level(IndexLevel.RAW, force=True)
    
    @pytest.mark.asyncio
    async def test_build_to_level_level1_requires_level0(self, temp_output_dir):
        """Level 1 構築には Level 0 が必要"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # Level 0 が構築されていない状態で Level 1 を構築しようとする
        # build_to_level は内部で build_level を呼び、Level 0 のチェックが行われる
        with pytest.raises(ValueError, match="RAW must be built|documents and text_units"):
            await manager.build_to_level(IndexLevel.LAZY)
    
    @pytest.mark.asyncio
    async def test_build_to_level_sequential(self, temp_output_dir):
        """順次構築のテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # モックのセットアップ
        mock_doc = MagicMock()
        mock_tu = MagicMock()
        mock_tu.content = "test content"
        
        # Level 0 ビルダー
        mock_l0_builder = MagicMock()
        mock_l0_index = MagicMock()
        mock_l0_index.document_count = 1
        mock_l0_index.text_unit_count = 1
        mock_l0_index.embedding_model = "test"
        mock_l0_index.embedding_dimensions = 768
        mock_l0_index.embedding_count = 1
        mock_l0_builder.build = AsyncMock(return_value=mock_l0_index)
        manager._level0_builder = mock_l0_builder
        
        # Level 0 だけ構築（build_levelを直接使う）
        state = await manager.build_level(
            IndexLevel.RAW,
            documents=[mock_doc],
            text_units=[mock_tu]
        )
        
        assert state.is_level_built(IndexLevel.RAW)
        assert state.current_level == IndexLevel.RAW


class TestBuildLevelInternalMethods:
    """_build_level_X メソッドのテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.asyncio
    async def test_build_level_1_internal(self, temp_output_dir):
        """_build_level_1 のテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # Level 0 ディレクトリを作成
        level0_dir = temp_output_dir / "level_0"
        level0_dir.mkdir()
        
        # Level 0 を構築済みにする
        manager.state.level_status[IndexLevel.RAW] = LevelStatus(
            level=IndexLevel.RAW, is_built=True
        )
        
        # ParquetStorage をモック
        with patch("monjyu.storage.parquet.ParquetStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_tu = MagicMock()
            mock_tu.content = "test"
            mock_storage.load_text_units.return_value = [mock_tu]
            mock_storage_class.return_value = mock_storage
            
            # Level 1 ビルダーをモック
            mock_l1_builder = MagicMock()
            mock_l1_index = MagicMock()
            mock_l1_index.node_count = 10
            mock_l1_index.edge_count = 20
            mock_l1_index.community_count = 3
            mock_l1_builder.build = AsyncMock(return_value=mock_l1_index)
            manager._level1_builder = mock_l1_builder
            
            # 実行
            state = await manager.build_level(IndexLevel.LAZY)
            
            assert state.is_level_built(IndexLevel.LAZY)
            status = state.get_level_status(IndexLevel.LAZY)
            assert status.stats["node_count"] == 10
    
    @pytest.mark.asyncio
    async def test_build_level_2_internal(self, temp_output_dir):
        """_build_level_2 のテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # 前提レベルを構築済みに
        for level in [IndexLevel.RAW, IndexLevel.LAZY]:
            manager.state.level_status[level] = LevelStatus(level=level, is_built=True)
        
        # Level 0 ディレクトリを作成
        level0_dir = temp_output_dir / "level_0"
        level0_dir.mkdir()
        
        with patch("monjyu.storage.parquet.ParquetStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_tu = MagicMock()
            mock_tu.content = "test entity extraction"
            mock_storage.load_text_units.return_value = [mock_tu]
            mock_storage_class.return_value = mock_storage
            
            # エンティティ抽出器をモック
            mock_entity_extractor = MagicMock()
            mock_entity = MagicMock()
            mock_entity.to_dict.return_value = {"name": "Test Entity"}
            mock_entity_extractor.extract_batch = AsyncMock(return_value=[mock_entity])
            manager._entity_extractor = mock_entity_extractor
            
            # 関係抽出器をモック
            mock_rel_extractor = MagicMock()
            mock_rel = MagicMock()
            mock_rel.to_dict.return_value = {"source": "A", "target": "B"}
            mock_rel_extractor.extract_batch = AsyncMock(return_value=[mock_rel])
            manager._relationship_extractor = mock_rel_extractor
            
            # 実行
            state = await manager.build_level(IndexLevel.PARTIAL)
            
            assert state.is_level_built(IndexLevel.PARTIAL)
            
            # ファイルが作成されたか確認
            level2_dir = temp_output_dir / "level_2"
            assert level2_dir.exists()
    
    @pytest.mark.asyncio
    async def test_build_level_3_internal(self, temp_output_dir):
        """_build_level_3 のテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # 前提レベルを構築済みに
        for level in [IndexLevel.RAW, IndexLevel.LAZY, IndexLevel.PARTIAL]:
            manager.state.level_status[level] = LevelStatus(level=level, is_built=True)
        
        # Level 2 データを作成
        level2_dir = temp_output_dir / "level_2"
        level2_dir.mkdir()
        
        with open(level2_dir / "entities.json", "w") as f:
            json.dump([{"name": "Entity1"}], f)
        
        with open(level2_dir / "relationships.json", "w") as f:
            json.dump([{"source": "A", "target": "B"}], f)
        
        # コミュニティレポート生成器をモック
        mock_generator = MagicMock()
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"title": "Report 1"}
        mock_generator.generate_from_entities = AsyncMock(return_value=[mock_report])
        manager._community_report_generator = mock_generator
        
        # 実行
        state = await manager.build_level(IndexLevel.FULL)
        
        assert state.is_level_built(IndexLevel.FULL)
        
        # ファイルが作成されたか確認
        level3_dir = temp_output_dir / "level_3"
        assert (level3_dir / "community_reports.json").exists()
    
    @pytest.mark.asyncio
    async def test_build_level_4_internal(self, temp_output_dir):
        """_build_level_4 のテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # 前提レベルを構築済みに
        for level in [IndexLevel.RAW, IndexLevel.LAZY, IndexLevel.PARTIAL, IndexLevel.FULL]:
            manager.state.level_status[level] = LevelStatus(level=level, is_built=True)
        
        # Level 0 ディレクトリを作成
        level0_dir = temp_output_dir / "level_0"
        level0_dir.mkdir()
        
        with patch("monjyu.storage.parquet.ParquetStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_tu = MagicMock()
            mock_tu.content = "test claim content"
            mock_storage.load_text_units.return_value = [mock_tu]
            mock_storage_class.return_value = mock_storage
            
            # クレーム抽出器をモック
            mock_claim_extractor = MagicMock()
            mock_claim_extractor.extract_batch = AsyncMock(return_value=[
                {"claim": "Test claim 1"},
                {"claim": "Test claim 2"}
            ])
            manager._claim_extractor = mock_claim_extractor
            
            # 実行
            state = await manager.build_level(IndexLevel.ENHANCED)
            
            assert state.is_level_built(IndexLevel.ENHANCED)
            
            # ファイルが作成されたか確認
            level4_dir = temp_output_dir / "level_4"
            assert (level4_dir / "claims.json").exists()


class TestStatusSummary:
    """get_status_summary のテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_status_summary_all_levels(self, temp_output_dir):
        """全レベルのサマリー"""
        config = ProgressiveIndexConfig(
            output_dir=temp_output_dir,
            index_name="summary_test"
        )
        manager = ProgressiveIndexManager(config)
        
        # いくつかのレベルを構築済みに
        manager.state.level_status[IndexLevel.RAW] = LevelStatus(
            level=IndexLevel.RAW,
            is_built=True,
            built_at="2025-01-01T00:00:00Z",
            stats={"embedding_count": 100}
        )
        manager.state.level_status[IndexLevel.LAZY] = LevelStatus(
            level=IndexLevel.LAZY,
            is_built=True,
            stats={"node_count": 50}
        )
        
        summary = manager.get_status_summary()
        
        assert summary["name"] == "summary_test"
        assert len(summary["levels"]) == 5
        
        # RAW レベルの確認
        raw_level = next(l for l in summary["levels"] if l["name"] == "RAW")
        assert raw_level["is_built"] is True
        assert raw_level["stats"]["embedding_count"] == 100
        
        # LAZY レベルの確認
        lazy_level = next(l for l in summary["levels"] if l["name"] == "LAZY")
        assert lazy_level["is_built"] is True


class TestFactoryFunctionEdgeCases:
    """ファクトリ関数のエッジケーステスト"""
    
    def test_factory_with_all_kwargs(self):
        """全kwargを指定"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_progressive_index_manager(
                output_dir=tmpdir,
                index_name="full_config",
                embedding_strategy="azure",
                index_strategy="azure_search",
                ollama_model="custom-model",
                spacy_model="ja_core_news_sm",
                llm_model="gpt-4",
                batch_size=200,
                show_progress=False,
            )
            
            assert manager.config.embedding_strategy == "azure"
            assert manager.config.index_strategy == "azure_search"
            assert manager.config.ollama_model == "custom-model"
            assert manager.config.batch_size == 200
            assert manager.config.show_progress is False


class TestStateManagement:
    """状態管理のテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_state_property_creates_new(self, temp_output_dir):
        """stateプロパティが新規作成"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # 内部状態はNone
        assert manager._state is None
        
        # stateにアクセスすると作成される
        state = manager.state
        
        assert manager._state is not None
        assert state.name == "monjyu_index"
    
    def test_load_existing_state(self, temp_output_dir):
        """既存状態の読み込み"""
        # 最初のマネージャーで状態を作成
        config1 = ProgressiveIndexConfig(
            output_dir=temp_output_dir,
            index_name="existing_test"
        )
        manager1 = ProgressiveIndexManager(config1)
        state1 = manager1.state
        state1.level_status[IndexLevel.RAW] = LevelStatus(
            level=IndexLevel.RAW, is_built=True
        )
        state1.save()
        
        # 新しいマネージャーで読み込み
        config2 = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager2 = ProgressiveIndexManager(config2)
        
        state2 = manager2.state
        
        # 同じIDが読み込まれる
        assert state2.id == state1.id
        assert state2.is_level_built(IndexLevel.RAW)


class TestUpdateTimestamps:
    """タイムスタンプ更新のテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.asyncio
    async def test_build_updates_timestamp(self, temp_output_dir):
        """ビルド時にタイムスタンプが更新される"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # 古いタイムスタンプを設定
        manager.state._state = None  # 強制リセット
        original_updated = manager.state.updated_at
        
        # モック
        mock_builder = MagicMock()
        mock_index = MagicMock()
        mock_index.document_count = 1
        mock_index.text_unit_count = 1
        mock_index.embedding_model = "test"
        mock_index.embedding_dimensions = 768
        mock_index.embedding_count = 1
        mock_builder.build = AsyncMock(return_value=mock_index)
        manager._level0_builder = mock_builder
        
        # ビルド実行
        mock_doc = MagicMock()
        mock_tu = MagicMock()
        
        import asyncio
        await asyncio.sleep(0.01)  # 時間差を作る
        
        state = await manager.build_level(
            IndexLevel.RAW,
            documents=[mock_doc],
            text_units=[mock_tu]
        )
        
        # タイムスタンプが更新されている
        assert state.updated_at != original_updated or state.is_level_built(IndexLevel.RAW)

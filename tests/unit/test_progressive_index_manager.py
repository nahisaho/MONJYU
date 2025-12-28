# Progressive Index Manager Tests
"""
FEAT-015: Progressive Index Manager テスト

ProgressiveIndexManager のユニットテスト
"""

import json
import tempfile
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


class TestIndexLevel:
    """IndexLevel のテスト"""
    
    def test_level_values(self):
        """レベル値のテスト"""
        assert IndexLevel.RAW == 0
        assert IndexLevel.LAZY == 1
        assert IndexLevel.PARTIAL == 2
        assert IndexLevel.FULL == 3
        assert IndexLevel.ENHANCED == 4
    
    def test_level_order(self):
        """レベル順序のテスト"""
        assert IndexLevel.RAW < IndexLevel.LAZY
        assert IndexLevel.LAZY < IndexLevel.PARTIAL
        assert IndexLevel.PARTIAL < IndexLevel.FULL
        assert IndexLevel.FULL < IndexLevel.ENHANCED


class TestLevelStatus:
    """LevelStatus のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        status = LevelStatus(level=IndexLevel.RAW)
        
        assert status.level == IndexLevel.RAW
        assert status.is_built is False
        assert status.built_at is None
        assert status.document_count == 0
        assert status.chunk_count == 0
        assert status.stats == {}
    
    def test_to_dict(self):
        """辞書変換のテスト"""
        status = LevelStatus(
            level=IndexLevel.LAZY,
            is_built=True,
            built_at="2025-01-01T00:00:00Z",
            document_count=10,
            chunk_count=100,
            stats={"node_count": 50},
        )
        
        data = status.to_dict()
        
        assert data["level"] == 1
        assert data["level_name"] == "LAZY"
        assert data["is_built"] is True
        assert data["document_count"] == 10
        assert data["stats"]["node_count"] == 50
    
    def test_from_dict(self):
        """辞書からの作成テスト"""
        data = {
            "level": 2,
            "is_built": True,
            "built_at": "2025-01-01T00:00:00Z",
            "document_count": 5,
            "chunk_count": 50,
            "stats": {"entity_count": 100},
        }
        
        status = LevelStatus.from_dict(data)
        
        assert status.level == IndexLevel.PARTIAL
        assert status.is_built is True
        assert status.stats["entity_count"] == 100


class TestProgressiveIndexState:
    """ProgressiveIndexState のテスト"""
    
    def test_to_dict_and_from_dict(self):
        """辞書変換と復元のテスト"""
        state = ProgressiveIndexState(
            id="test-id",
            name="test-index",
            current_level=IndexLevel.LAZY,
            level_status={
                IndexLevel.RAW: LevelStatus(level=IndexLevel.RAW, is_built=True),
                IndexLevel.LAZY: LevelStatus(level=IndexLevel.LAZY, is_built=True),
            },
            output_dir=Path("/tmp/test"),
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )
        
        data = state.to_dict()
        restored = ProgressiveIndexState.from_dict(data)
        
        assert restored.id == "test-id"
        assert restored.name == "test-index"
        assert restored.current_level == IndexLevel.LAZY
        assert restored.is_level_built(IndexLevel.RAW)
        assert restored.is_level_built(IndexLevel.LAZY)
    
    def test_save_and_load(self):
        """保存と読み込みのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_index"
            output_dir.mkdir()
            
            state = ProgressiveIndexState(
                id="test-id",
                name="test-index",
                current_level=IndexLevel.RAW,
                level_status={
                    IndexLevel.RAW: LevelStatus(level=IndexLevel.RAW, is_built=True),
                },
                output_dir=output_dir,
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
            )
            
            # 保存
            state.save()
            
            # 読み込み
            loaded = ProgressiveIndexState.load(output_dir / "state.json")
            
            assert loaded.id == "test-id"
            assert loaded.is_level_built(IndexLevel.RAW)
    
    def test_get_level_status(self):
        """レベルステータス取得のテスト"""
        state = ProgressiveIndexState(
            id="test",
            name="test",
            current_level=IndexLevel.RAW,
            level_status={},
            output_dir=Path("/tmp"),
            created_at="",
            updated_at="",
        )
        
        # 存在しないレベルを取得（自動作成）
        status = state.get_level_status(IndexLevel.LAZY)
        
        assert status.level == IndexLevel.LAZY
        assert status.is_built is False


class TestProgressiveIndexConfig:
    """ProgressiveIndexConfig のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        config = ProgressiveIndexConfig()
        
        assert config.output_dir == "./output/progressive_index"
        assert config.index_name == "monjyu_index"
        assert config.embedding_strategy == "ollama"
        assert config.index_strategy == "lancedb"
        assert config.spacy_model == "en_core_web_sm"
        assert config.batch_size == 50
    
    def test_custom_values(self):
        """カスタム値のテスト"""
        config = ProgressiveIndexConfig(
            output_dir="/custom/path",
            index_name="custom_index",
            embedding_strategy="azure",
            batch_size=100,
        )
        
        assert config.output_dir == "/custom/path"
        assert config.index_name == "custom_index"
        assert config.embedding_strategy == "azure"
        assert config.batch_size == 100


class TestProgressiveIndexManager:
    """ProgressiveIndexManager のテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_init(self, temp_output_dir):
        """初期化のテスト"""
        config = ProgressiveIndexConfig(
            output_dir=temp_output_dir,
            index_name="test_index",
        )
        manager = ProgressiveIndexManager(config)
        
        assert manager.config.index_name == "test_index"
        assert manager.output_dir == temp_output_dir
    
    def test_state_creation(self, temp_output_dir):
        """状態作成のテスト"""
        config = ProgressiveIndexConfig(
            output_dir=temp_output_dir,
            index_name="test_index",
        )
        manager = ProgressiveIndexManager(config)
        
        state = manager.state
        
        assert state.name == "test_index"
        assert state.current_level == IndexLevel.RAW
        assert not state.is_level_built(IndexLevel.RAW)
        
        # state.json が作成されている
        assert (temp_output_dir / "state.json").exists()
    
    def test_state_persistence(self, temp_output_dir):
        """状態の永続化テスト"""
        config = ProgressiveIndexConfig(
            output_dir=temp_output_dir,
            index_name="test_index",
        )
        
        # 最初のマネージャー
        manager1 = ProgressiveIndexManager(config)
        state_id = manager1.state.id
        
        # 新しいマネージャー（同じディレクトリ）
        manager2 = ProgressiveIndexManager(config)
        
        # 同じ状態が読み込まれる
        assert manager2.state.id == state_id
    
    def test_get_status_summary(self, temp_output_dir):
        """ステータスサマリーのテスト"""
        config = ProgressiveIndexConfig(
            output_dir=temp_output_dir,
            index_name="test_index",
        )
        manager = ProgressiveIndexManager(config)
        
        summary = manager.get_status_summary()
        
        assert summary["name"] == "test_index"
        assert summary["current_level"] == "RAW"
        assert len(summary["levels"]) == 5  # RAW, LAZY, PARTIAL, FULL, ENHANCED
    
    @pytest.mark.asyncio
    async def test_build_level_requires_previous(self, temp_output_dir):
        """前提レベルが必要なテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # Level 0 なしで Level 1 を構築しようとする
        with pytest.raises(ValueError, match="RAW must be built"):
            await manager.build_level(IndexLevel.LAZY)
    
    @pytest.mark.asyncio
    async def test_build_level_0_requires_data(self, temp_output_dir):
        """Level 0 はデータが必要なテスト"""
        config = ProgressiveIndexConfig(output_dir=temp_output_dir)
        manager = ProgressiveIndexManager(config)
        
        # データなしで Level 0 を構築しようとする
        with pytest.raises(ValueError, match="documents and text_units are required"):
            await manager.build_level(IndexLevel.RAW)


class TestFactoryFunction:
    """ファクトリ関数のテスト"""
    
    def test_create_progressive_index_manager(self):
        """ファクトリ関数のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_progressive_index_manager(
                output_dir=tmpdir,
                index_name="test",
                batch_size=100,
            )
            
            assert isinstance(manager, ProgressiveIndexManager)
            assert manager.config.index_name == "test"
            assert manager.config.batch_size == 100

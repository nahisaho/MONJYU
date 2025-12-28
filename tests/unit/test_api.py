# MONJYU API Unit Tests
"""
FEAT-007: Python API - 単体テスト
"""

import pytest
from pathlib import Path
from typing import Any


# ========== Base Types Tests ==========


class TestSearchMode:
    """SearchMode のテスト"""

    def test_enum_values(self):
        from monjyu.api.base import SearchMode

        assert SearchMode.VECTOR.value == "vector"
        assert SearchMode.LAZY.value == "lazy"
        assert SearchMode.AUTO.value == "auto"


class TestIndexLevel:
    """IndexLevel のテスト"""

    def test_enum_values(self):
        from monjyu.api.base import IndexLevel

        assert IndexLevel.LEVEL_0.value == 0
        assert IndexLevel.LEVEL_1.value == 1


class TestIndexStatus:
    """IndexStatus のテスト"""

    def test_enum_values(self):
        from monjyu.api.base import IndexStatus

        assert IndexStatus.NOT_BUILT.value == "not_built"
        assert IndexStatus.BUILDING.value == "building"
        assert IndexStatus.READY.value == "ready"
        assert IndexStatus.ERROR.value == "error"


class TestMONJYUConfig:
    """MONJYUConfig のテスト"""

    def test_defaults(self):
        from monjyu.api.base import MONJYUConfig, SearchMode, IndexLevel

        config = MONJYUConfig()

        assert config.output_path == Path("./output")
        assert config.environment == "local"
        assert config.default_search_mode == SearchMode.LAZY
        assert IndexLevel.LEVEL_0 in config.index_levels

    def test_custom_values(self):
        from monjyu.api.base import MONJYUConfig, SearchMode

        config = MONJYUConfig(
            output_path=Path("/custom/path"),
            environment="azure",
            default_search_mode=SearchMode.VECTOR,
            default_top_k=20,
        )

        assert config.output_path == Path("/custom/path")
        assert config.environment == "azure"
        assert config.default_search_mode == SearchMode.VECTOR
        assert config.default_top_k == 20

    def test_string_path_conversion(self):
        from monjyu.api.base import MONJYUConfig

        config = MONJYUConfig(output_path="./my_output")  # type: ignore

        assert isinstance(config.output_path, Path)
        assert config.output_path == Path("./my_output")


class TestMONJYUStatus:
    """MONJYUStatus のテスト"""

    def test_defaults(self):
        from monjyu.api.base import MONJYUStatus, IndexStatus

        status = MONJYUStatus()

        assert status.index_status == IndexStatus.NOT_BUILT
        assert status.document_count == 0
        assert status.is_ready is False

    def test_is_ready(self):
        from monjyu.api.base import MONJYUStatus, IndexStatus

        status = MONJYUStatus(index_status=IndexStatus.READY)
        assert status.is_ready is True

        status2 = MONJYUStatus(index_status=IndexStatus.BUILDING)
        assert status2.is_ready is False


class TestCitation:
    """Citation のテスト"""

    def test_creation(self):
        from monjyu.api.base import Citation

        citation = Citation(
            doc_id="doc1",
            title="Test Paper",
            text="Some text",
            relevance_score=0.95,
        )

        assert citation.doc_id == "doc1"
        assert citation.title == "Test Paper"
        assert citation.relevance_score == 0.95


class TestSearchResult:
    """SearchResult のテスト"""

    def test_creation(self):
        from monjyu.api.base import SearchResult, SearchMode, Citation

        result = SearchResult(
            query="test query",
            answer="test answer",
            citations=[
                Citation(doc_id="d1", title="Paper 1"),
                Citation(doc_id="d2", title="Paper 2"),
            ],
            search_mode=SearchMode.LAZY,
            search_level=1,
            total_time_ms=100.5,
            llm_calls=2,
        )

        assert result.query == "test query"
        assert result.answer == "test answer"
        assert result.citation_count == 2
        assert result.search_mode == SearchMode.LAZY


class TestDocumentInfo:
    """DocumentInfo のテスト"""

    def test_creation(self):
        from monjyu.api.base import DocumentInfo

        doc = DocumentInfo(
            id="doc1",
            title="Test Paper",
            authors=["Author A", "Author B"],
            year=2023,
            doi="10.1234/test",
        )

        assert doc.id == "doc1"
        assert len(doc.authors) == 2
        assert doc.year == 2023


# ========== ConfigManager Tests ==========


class TestConfigManager:
    """ConfigManager のテスト"""

    def test_default_config(self):
        from monjyu.api.config import ConfigManager

        manager = ConfigManager()
        config = manager.load()

        assert config is not None
        assert config.environment == "local"

    def test_from_dict(self):
        from monjyu.api.config import ConfigManager
        from monjyu.api.base import SearchMode

        data = {
            "environment": "azure",
            "default_search_mode": "vector",
            "default_top_k": 20,
        }

        manager = ConfigManager.from_dict(data)

        assert manager.config.environment == "azure"
        assert manager.config.default_search_mode == SearchMode.VECTOR
        assert manager.config.default_top_k == 20

    def test_from_yaml(self, tmp_path):
        from monjyu.api.config import ConfigManager
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "environment": "local",
            "output_path": str(tmp_path / "output"),
            "default_top_k": 15,
        }))

        manager = ConfigManager.from_yaml(config_file)

        assert manager.config.default_top_k == 15

    def test_save_config(self, tmp_path):
        from monjyu.api.config import ConfigManager
        from monjyu.api.base import MONJYUConfig
        import yaml

        config = MONJYUConfig(
            output_path=tmp_path / "output",
            default_top_k=25,
        )

        manager = ConfigManager.from_config(config)

        save_path = tmp_path / "saved_config.yaml"
        manager.save(save_path)

        assert save_path.exists()

        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["default_top_k"] == 25

    def test_load_config_helper(self, tmp_path):
        from monjyu.api.config import load_config
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "chunk_size": 2000,
        }))

        config = load_config(config_file)

        assert config.chunk_size == 2000


# ========== StateManager Tests ==========


class TestStateManager:
    """StateManager のテスト"""

    def test_initial_state(self, tmp_path):
        from monjyu.api.state import StateManager
        from monjyu.api.base import IndexStatus

        manager = StateManager(tmp_path)
        status = manager.load()

        assert status.index_status == IndexStatus.NOT_BUILT
        assert status.document_count == 0

    def test_update_and_save(self, tmp_path):
        from monjyu.api.state import StateManager
        from monjyu.api.base import IndexStatus

        manager = StateManager(tmp_path)
        manager.load()

        manager.update(
            index_status=IndexStatus.READY,
            document_count=100,
        )

        assert manager.status.index_status == IndexStatus.READY
        assert manager.status.document_count == 100

        # ファイルが保存されたか確認
        assert (tmp_path / "monjyu_state.json").exists()

    def test_load_existing_state(self, tmp_path):
        from monjyu.api.state import StateManager
        from monjyu.api.base import IndexStatus, IndexLevel
        import json

        # 既存の状態ファイルを作成
        state_file = tmp_path / "monjyu_state.json"
        state_file.write_text(json.dumps({
            "index_status": "ready",
            "index_levels_built": [0, 1],
            "document_count": 50,
            "text_unit_count": 500,
        }))

        manager = StateManager(tmp_path)
        status = manager.load()

        assert status.index_status == IndexStatus.READY
        assert IndexLevel.LEVEL_0 in status.index_levels_built
        assert status.document_count == 50

    def test_reset(self, tmp_path):
        from monjyu.api.state import StateManager
        from monjyu.api.base import IndexStatus

        manager = StateManager(tmp_path)
        manager.update(index_status=IndexStatus.READY, document_count=100)

        status = manager.reset()

        assert status.index_status == IndexStatus.NOT_BUILT
        assert status.document_count == 0


# ========== ComponentFactory Tests ==========


class TestComponentFactory:
    """ComponentFactory のテスト"""

    def test_initialization(self):
        from monjyu.api.factory import ComponentFactory
        from monjyu.api.base import MONJYUConfig

        config = MONJYUConfig()
        factory = ComponentFactory(config)

        assert factory.config == config

    def test_get_embedding_client(self):
        from monjyu.api.factory import ComponentFactory
        from monjyu.api.base import MONJYUConfig

        config = MONJYUConfig()
        factory = ComponentFactory(config)

        client = factory.get_embedding_client()
        assert client is not None

        # キャッシュテスト
        client2 = factory.get_embedding_client()
        assert client is client2

    def test_get_llm_client(self):
        from monjyu.api.factory import ComponentFactory
        from monjyu.api.base import MONJYUConfig

        config = MONJYUConfig()
        factory = ComponentFactory(config)

        client = factory.get_llm_client()
        assert client is not None

    def test_get_citation_network_manager(self):
        from monjyu.api.factory import ComponentFactory
        from monjyu.api.base import MONJYUConfig

        config = MONJYUConfig()
        factory = ComponentFactory(config)

        manager = factory.get_citation_network_manager()
        assert manager is not None

    def test_clear_cache(self):
        from monjyu.api.factory import ComponentFactory
        from monjyu.api.base import MONJYUConfig

        config = MONJYUConfig()
        factory = ComponentFactory(config)

        client1 = factory.get_embedding_client()
        factory.clear_cache()
        client2 = factory.get_embedding_client()

        # キャッシュクリア後は新しいインスタンス
        assert client1 is not client2


class TestMockClients:
    """モッククライアントのテスト"""

    def test_mock_embedding_client(self):
        from monjyu.api.factory import MockEmbeddingClient

        client = MockEmbeddingClient()

        vector = client.embed("test text")
        assert isinstance(vector, list)
        assert len(vector) == 384

        vectors = client.embed_batch(["text1", "text2"])
        assert len(vectors) == 2

    def test_mock_llm_client(self):
        from monjyu.api.factory import MockLLMClient

        client = MockLLMClient()

        response = client.generate("Hello")
        assert isinstance(response, str)
        assert "Mock Response" in response

    def test_mock_vector_searcher(self):
        from monjyu.api.factory import MockVectorSearcher

        data = [{"id": "1", "text": "doc1"}, {"id": "2", "text": "doc2"}]
        searcher = MockVectorSearcher(data)

        results = searcher.search([0.1] * 10, top_k=5)
        assert len(results) == 2


# ========== MONJYU Facade Tests ==========


class TestMONJYU:
    """MONJYU Facadeのテスト"""

    def test_initialization_default(self):
        from monjyu.api import MONJYU

        monjyu = MONJYU()

        assert monjyu.config is not None
        assert monjyu.config.environment == "local"

    def test_initialization_with_dict(self):
        from monjyu.api import MONJYU

        monjyu = MONJYU({
            "environment": "azure",
            "default_top_k": 20,
        })

        assert monjyu.config.environment == "azure"
        assert monjyu.config.default_top_k == 20

    def test_initialization_with_config(self):
        from monjyu.api import MONJYU, MONJYUConfig

        config = MONJYUConfig(default_top_k=30)
        monjyu = MONJYU(config)

        assert monjyu.config.default_top_k == 30

    def test_initialization_with_yaml(self, tmp_path):
        from monjyu.api import MONJYU
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "default_top_k": 25,
        }))

        monjyu = MONJYU(config_file)

        assert monjyu.config.default_top_k == 25

    def test_get_status(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, IndexStatus

        config = MONJYUConfig(output_path=tmp_path)
        monjyu = MONJYU(config)

        status = monjyu.get_status()

        assert status.index_status == IndexStatus.NOT_BUILT

    def test_index_nonexistent_path(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig

        config = MONJYUConfig(output_path=tmp_path)
        monjyu = MONJYU(config)

        with pytest.raises(FileNotFoundError):
            monjyu.index(tmp_path / "nonexistent")

    def test_index_empty_directory(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, IndexStatus

        # 空のディレクトリを作成
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        config = MONJYUConfig(output_path=tmp_path / "output")
        monjyu = MONJYU(config)

        status = monjyu.index(docs_dir, show_progress=False)

        assert status.index_status == IndexStatus.READY
        assert status.document_count == 0

    def test_index_with_files(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, IndexStatus

        # テストファイルを作成
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.md").write_text("# Document 1\n\nThis is test content.")
        (docs_dir / "doc2.txt").write_text("Document 2 content.")

        config = MONJYUConfig(output_path=tmp_path / "output")
        monjyu = MONJYU(config)

        status = monjyu.index(docs_dir, show_progress=False)

        assert status.index_status == IndexStatus.READY
        assert status.document_count == 2
        assert status.text_unit_count > 0

    def test_search_vector(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, SearchMode

        config = MONJYUConfig(output_path=tmp_path)
        monjyu = MONJYU(config)

        result = monjyu.search("test query", mode=SearchMode.VECTOR)

        assert result.query == "test query"
        assert result.search_mode == SearchMode.VECTOR
        assert result.llm_calls == 1

    def test_search_lazy(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, SearchMode

        config = MONJYUConfig(output_path=tmp_path)
        monjyu = MONJYU(config)

        result = monjyu.search("explain the relationship", mode=SearchMode.LAZY)

        assert result.search_mode == SearchMode.LAZY
        assert result.search_level >= 1

    def test_search_auto_simple(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, SearchMode

        config = MONJYUConfig(output_path=tmp_path)
        monjyu = MONJYU(config)

        result = monjyu.search("simple query", mode=SearchMode.AUTO)

        # 短いクエリはVECTORになるはず
        assert result.search_mode == SearchMode.VECTOR

    def test_search_auto_complex(self, tmp_path):
        from monjyu.api import MONJYU, MONJYUConfig, SearchMode

        config = MONJYUConfig(output_path=tmp_path)
        monjyu = MONJYU(config)

        result = monjyu.search(
            "How does the relationship between attention and performance differ?",
            mode=SearchMode.AUTO,
        )

        # 複雑なクエリはLAZYになるはず
        assert result.search_mode == SearchMode.LAZY

    def test_chunk_text(self):
        from monjyu.api.monjyu import MONJYU

        # 短いテキスト
        chunks = MONJYU._chunk_text("short text", chunk_size=100)
        assert len(chunks) == 1

        # 長いテキスト
        long_text = "a" * 500
        chunks = MONJYU._chunk_text(long_text, chunk_size=100, overlap=20)
        assert len(chunks) > 1


class TestCreateMonjyu:
    """create_monjyu ファクトリ関数のテスト"""

    def test_create_default(self):
        from monjyu.api import create_monjyu

        monjyu = create_monjyu()
        assert monjyu is not None

    def test_create_with_dict(self):
        from monjyu.api import create_monjyu

        monjyu = create_monjyu({"default_top_k": 15})
        assert monjyu.config.default_top_k == 15


# ========== Integration-like Tests ==========


class TestMONJYUWorkflow:
    """MONJYUワークフローテスト"""

    def test_full_workflow(self, tmp_path):
        """完全なワークフロー（index -> search）"""
        from monjyu.api import MONJYU, MONJYUConfig, IndexStatus

        # ドキュメント準備
        docs_dir = tmp_path / "papers"
        docs_dir.mkdir()

        (docs_dir / "transformer.md").write_text("""
# Attention Is All You Need

The Transformer architecture relies entirely on self-attention mechanisms.
This paper introduces multi-head attention and positional encoding.
""")

        (docs_dir / "bert.md").write_text("""
# BERT: Pre-training of Deep Bidirectional Transformers

BERT uses bidirectional transformers for language understanding.
It introduces masked language modeling and next sentence prediction.
""")

        # MONJYU初期化
        config = MONJYUConfig(
            output_path=tmp_path / "output",
            chunk_size=500,
            chunk_overlap=50,
        )
        monjyu = MONJYU(config)

        # インデックス構築
        status = monjyu.index(docs_dir, show_progress=False)

        assert status.index_status == IndexStatus.READY
        assert status.document_count == 2

        # 検索
        result = monjyu.search("What is Transformer?")

        assert result.query == "What is Transformer?"
        assert result.answer is not None
        assert result.total_time_ms > 0

    def test_status_persistence(self, tmp_path):
        """状態の永続化テスト"""
        from monjyu.api import MONJYU, MONJYUConfig, IndexStatus

        config = MONJYUConfig(output_path=tmp_path / "output")

        # 最初のインスタンス
        monjyu1 = MONJYU(config)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("Test content")

        monjyu1.index(docs_dir, show_progress=False)

        # 新しいインスタンス
        monjyu2 = MONJYU(config)
        status = monjyu2.get_status()

        assert status.index_status == IndexStatus.READY
        assert status.document_count == 1

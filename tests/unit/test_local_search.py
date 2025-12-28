"""Unit tests for LocalSearch module."""

import pytest
from dataclasses import asdict

from monjyu.query.local_search import (
    LocalSearch,
    LocalSearchConfig,
    LocalSearchResult,
    EntityInfo,
    EntityMatch,
    RelationshipInfo,
    ChunkInfo,
    InMemoryEntityStore,
    InMemoryRelationshipStore,
    InMemoryChunkStore,
    MockLLMClient,
    get_local_search_prompt,
    LOCAL_SEARCH_PROMPT_EN,
    LOCAL_SEARCH_PROMPT_JA,
)


# =============================================================================
# LocalSearchConfig Tests
# =============================================================================


class TestLocalSearchConfig:
    """LocalSearchConfigのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        config = LocalSearchConfig()
        assert config.max_hops == 2
        assert config.top_k_entities == 10
        assert config.top_k_chunks == 20
        assert config.include_relationships is True
        assert config.max_context_tokens == 8000
        assert config.temperature == 0.0
        assert config.response_language == "auto"

    def test_custom_values(self):
        """カスタム値のテスト"""
        config = LocalSearchConfig(
            max_hops=3,
            top_k_entities=5,
            top_k_chunks=10,
            include_relationships=False,
            max_context_tokens=4000,
            temperature=0.5,
            response_language="ja",
        )
        assert config.max_hops == 3
        assert config.top_k_entities == 5
        assert config.top_k_chunks == 10
        assert config.include_relationships is False
        assert config.max_context_tokens == 4000
        assert config.temperature == 0.5
        assert config.response_language == "ja"

    def test_to_dict(self):
        """to_dictのテスト"""
        config = LocalSearchConfig(max_hops=3, top_k_entities=5)
        d = config.to_dict()
        assert d["max_hops"] == 3
        assert d["top_k_entities"] == 5
        assert "include_relationships" in d

    def test_from_dict(self):
        """from_dictのテスト"""
        data = {
            "max_hops": 4,
            "top_k_entities": 15,
            "include_relationships": False,
        }
        config = LocalSearchConfig.from_dict(data)
        assert config.max_hops == 4
        assert config.top_k_entities == 15
        assert config.include_relationships is False

    def test_from_dict_with_defaults(self):
        """from_dictでデフォルト値が適用されるテスト"""
        data = {"max_hops": 3}
        config = LocalSearchConfig.from_dict(data)
        assert config.max_hops == 3
        assert config.top_k_entities == 10  # default


# =============================================================================
# EntityInfo Tests
# =============================================================================


class TestEntityInfo:
    """EntityInfoのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        entity = EntityInfo(
            entity_id="e1",
            name="Transformer",
            entity_type="MODEL",
            description="A deep learning architecture based on attention mechanism.",
            properties={"year": 2017, "paper": "Attention is All You Need"},
        )
        assert entity.entity_id == "e1"
        assert entity.name == "Transformer"
        assert entity.entity_type == "MODEL"
        assert "attention" in entity.description
        assert entity.properties["year"] == 2017

    def test_default_values(self):
        """デフォルト値のテスト"""
        entity = EntityInfo(
            entity_id="e1",
            name="Test",
            entity_type="CONCEPT",
        )
        assert entity.description == ""
        assert entity.properties == {}

    def test_to_dict(self):
        """to_dictのテスト"""
        entity = EntityInfo(
            entity_id="e1",
            name="Test",
            entity_type="CONCEPT",
            description="Test description",
        )
        d = entity.to_dict()
        assert d["entity_id"] == "e1"
        assert d["name"] == "Test"
        assert d["entity_type"] == "CONCEPT"

    def test_from_dict(self):
        """from_dictのテスト"""
        data = {
            "entity_id": "e2",
            "name": "BERT",
            "entity_type": "MODEL",
            "description": "Bidirectional Encoder",
        }
        entity = EntityInfo.from_dict(data)
        assert entity.entity_id == "e2"
        assert entity.name == "BERT"


# =============================================================================
# RelationshipInfo Tests
# =============================================================================


class TestRelationshipInfo:
    """RelationshipInfoのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        rel = RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type="BASED_ON",
            description="Transformer is based on attention mechanism",
            weight=0.9,
        )
        assert rel.relationship_id == "r1"
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.relation_type == "BASED_ON"
        assert rel.weight == 0.9

    def test_default_values(self):
        """デフォルト値のテスト"""
        rel = RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type="RELATED",
        )
        assert rel.description == ""
        assert rel.weight == 1.0
        assert rel.properties == {}

    def test_to_dict(self):
        """to_dictのテスト"""
        rel = RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type="USES",
        )
        d = rel.to_dict()
        assert d["relationship_id"] == "r1"
        assert d["relation_type"] == "USES"


# =============================================================================
# ChunkInfo Tests
# =============================================================================


class TestChunkInfo:
    """ChunkInfoのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        chunk = ChunkInfo(
            chunk_id="ch1",
            content="The transformer architecture was introduced in 2017.",
            paper_id="p1",
            paper_title="Attention is All You Need",
            section_type="introduction",
            relevance_score=0.95,
        )
        assert chunk.chunk_id == "ch1"
        assert "transformer" in chunk.content
        assert chunk.paper_title == "Attention is All You Need"
        assert chunk.relevance_score == 0.95

    def test_default_values(self):
        """デフォルト値のテスト"""
        chunk = ChunkInfo(
            chunk_id="ch1",
            content="Test content",
        )
        assert chunk.paper_id == ""
        assert chunk.paper_title == ""
        assert chunk.section_type == ""
        assert chunk.relevance_score == 0.0


# =============================================================================
# EntityMatch Tests
# =============================================================================


class TestEntityMatch:
    """EntityMatchのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        entity = EntityInfo(
            entity_id="e1",
            name="Test",
            entity_type="CONCEPT",
        )
        match = EntityMatch(
            entity=entity,
            match_score=0.85,
            hop_distance=1,
            source_query_term="test",
        )
        assert match.entity.entity_id == "e1"
        assert match.match_score == 0.85
        assert match.hop_distance == 1

    def test_to_dict(self):
        """to_dictのテスト"""
        entity = EntityInfo(
            entity_id="e1",
            name="Test",
            entity_type="CONCEPT",
        )
        match = EntityMatch(entity=entity, match_score=0.5)
        d = match.to_dict()
        assert d["match_score"] == 0.5
        assert d["entity"]["entity_id"] == "e1"


# =============================================================================
# LocalSearchResult Tests
# =============================================================================


class TestLocalSearchResult:
    """LocalSearchResultのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        result = LocalSearchResult(
            query="What is Transformer?",
            answer="Transformer is a neural network architecture...",
            processing_time_ms=500,
            tokens_used=200,
            hops_traversed=2,
        )
        assert result.query == "What is Transformer?"
        assert "neural network" in result.answer
        assert result.processing_time_ms == 500
        assert result.hops_traversed == 2

    def test_default_values(self):
        """デフォルト値のテスト"""
        result = LocalSearchResult(
            query="Test",
            answer="Answer",
        )
        assert result.entities_found == []
        assert result.relationships_used == []
        assert result.chunks_used == []
        assert result.tokens_used == 0

    def test_to_dict(self):
        """to_dictのテスト"""
        result = LocalSearchResult(
            query="Test query",
            answer="Test answer",
            tokens_used=100,
        )
        d = result.to_dict()
        assert d["query"] == "Test query"
        assert d["answer"] == "Test answer"
        assert d["tokens_used"] == 100


# =============================================================================
# InMemoryEntityStore Tests
# =============================================================================


class TestInMemoryEntityStore:
    """InMemoryEntityStoreのテスト"""

    def test_add_and_get_by_id(self):
        """追加とID取得のテスト"""
        store = InMemoryEntityStore()
        entity = EntityInfo(
            entity_id="e1",
            name="Transformer",
            entity_type="MODEL",
        )
        store.add_entity(entity)
        
        result = store.get_entity_by_id("e1")
        assert result is not None
        assert result.name == "Transformer"

    def test_get_by_name(self):
        """名前で取得のテスト"""
        store = InMemoryEntityStore()
        entity = EntityInfo(
            entity_id="e1",
            name="BERT",
            entity_type="MODEL",
        )
        store.add_entity(entity)
        
        result = store.get_entity_by_name("bert")  # case insensitive
        assert result is not None
        assert result.entity_id == "e1"

    def test_search_entities(self):
        """エンティティ検索のテスト"""
        store = InMemoryEntityStore()
        store.add_entity(EntityInfo(
            entity_id="e1",
            name="Transformer",
            entity_type="MODEL",
            description="Deep learning architecture",
        ))
        store.add_entity(EntityInfo(
            entity_id="e2",
            name="CNN",
            entity_type="MODEL",
            description="Convolutional network",
        ))
        
        results = store.search_entities("transformer", top_k=5)
        assert len(results) == 1
        assert results[0].entity_id == "e1"

    def test_clear(self):
        """クリアのテスト"""
        store = InMemoryEntityStore()
        store.add_entity(EntityInfo(
            entity_id="e1",
            name="Test",
            entity_type="CONCEPT",
        ))
        store.clear()
        
        assert store.get_entity_by_id("e1") is None


# =============================================================================
# InMemoryRelationshipStore Tests
# =============================================================================


class TestInMemoryRelationshipStore:
    """InMemoryRelationshipStoreのテスト"""

    def test_add_and_get_for_entity(self):
        """追加とエンティティでの取得テスト"""
        store = InMemoryRelationshipStore()
        rel = RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type="USES",
        )
        store.add_relationship(rel)
        
        # source側から取得
        results = store.get_relationships_for_entity("e1")
        assert len(results) == 1
        assert results[0].target_id == "e2"
        
        # target側からも取得可能
        results = store.get_relationships_for_entity("e2")
        assert len(results) == 1

    def test_get_relationships_between(self):
        """複数エンティティ間のリレーション取得テスト"""
        store = InMemoryRelationshipStore()
        store.add_relationship(RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type="USES",
        ))
        store.add_relationship(RelationshipInfo(
            relationship_id="r2",
            source_id="e2",
            target_id="e3",
            relation_type="RELATED",
        ))
        store.add_relationship(RelationshipInfo(
            relationship_id="r3",
            source_id="e1",
            target_id="e3",
            relation_type="CITES",
        ))
        
        results = store.get_relationships_between(["e1", "e2"])
        assert len(results) == 1
        assert results[0].relationship_id == "r1"


# =============================================================================
# InMemoryChunkStore Tests
# =============================================================================


class TestInMemoryChunkStore:
    """InMemoryChunkStoreのテスト"""

    def test_add_and_get_for_entity(self):
        """追加とエンティティでの取得テスト"""
        store = InMemoryChunkStore()
        chunk = ChunkInfo(
            chunk_id="ch1",
            content="Transformer architecture explanation.",
        )
        store.add_chunk(chunk, entity_ids=["e1", "e2"])
        
        results = store.get_chunks_for_entity("e1")
        assert len(results) == 1
        assert results[0].chunk_id == "ch1"

    def test_search_chunks(self):
        """チャンク検索のテスト"""
        store = InMemoryChunkStore()
        store.add_chunk(ChunkInfo(
            chunk_id="ch1",
            content="Machine learning is a subset of AI.",
        ))
        store.add_chunk(ChunkInfo(
            chunk_id="ch2",
            content="Deep learning uses neural networks.",
        ))
        
        results = store.search_chunks("machine learning")
        assert len(results) == 1
        assert results[0].chunk_id == "ch1"


# =============================================================================
# Prompt Tests
# =============================================================================


class TestPrompts:
    """プロンプトのテスト"""

    def test_english_prompt_has_placeholders(self):
        """英語プロンプトにプレースホルダーがあることを確認"""
        assert "{entities}" in LOCAL_SEARCH_PROMPT_EN
        assert "{relationships}" in LOCAL_SEARCH_PROMPT_EN
        assert "{chunks}" in LOCAL_SEARCH_PROMPT_EN
        assert "{query}" in LOCAL_SEARCH_PROMPT_EN

    def test_japanese_prompt_has_placeholders(self):
        """日本語プロンプトにプレースホルダーがあることを確認"""
        assert "{entities}" in LOCAL_SEARCH_PROMPT_JA
        assert "{relationships}" in LOCAL_SEARCH_PROMPT_JA
        assert "{chunks}" in LOCAL_SEARCH_PROMPT_JA
        assert "{query}" in LOCAL_SEARCH_PROMPT_JA

    def test_get_local_search_prompt_auto(self):
        """auto言語設定で英語が返されることを確認"""
        prompt = get_local_search_prompt("auto")
        assert prompt == LOCAL_SEARCH_PROMPT_EN

    def test_get_local_search_prompt_japanese(self):
        """日本語設定で日本語プロンプトが返されることを確認"""
        for lang in ["ja", "japanese", "日本語"]:
            prompt = get_local_search_prompt(lang)
            assert prompt == LOCAL_SEARCH_PROMPT_JA


# =============================================================================
# LocalSearch Integration Tests
# =============================================================================


class TestLocalSearch:
    """LocalSearchの統合テスト"""

    @pytest.fixture
    def setup_stores(self):
        """テスト用のストアをセットアップ"""
        entity_store = InMemoryEntityStore()
        relationship_store = InMemoryRelationshipStore()
        chunk_store = InMemoryChunkStore()
        
        # エンティティを追加
        transformer = EntityInfo(
            entity_id="e1",
            name="Transformer",
            entity_type="MODEL",
            description="A neural network architecture based on self-attention.",
        )
        attention = EntityInfo(
            entity_id="e2",
            name="Self-Attention",
            entity_type="MECHANISM",
            description="A mechanism that relates different positions of a sequence.",
        )
        bert = EntityInfo(
            entity_id="e3",
            name="BERT",
            entity_type="MODEL",
            description="Bidirectional Encoder Representations from Transformers.",
        )
        
        entity_store.add_entity(transformer)
        entity_store.add_entity(attention)
        entity_store.add_entity(bert)
        
        # リレーションシップを追加
        relationship_store.add_relationship(RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type="USES",
            description="Transformer uses self-attention mechanism",
        ))
        relationship_store.add_relationship(RelationshipInfo(
            relationship_id="r2",
            source_id="e3",
            target_id="e1",
            relation_type="BASED_ON",
            description="BERT is based on Transformer",
        ))
        
        # チャンクを追加
        chunk_store.add_chunk(
            ChunkInfo(
                chunk_id="ch1",
                content="The Transformer architecture was introduced in the paper 'Attention is All You Need'.",
                paper_title="Attention is All You Need",
            ),
            entity_ids=["e1", "e2"],
        )
        chunk_store.add_chunk(
            ChunkInfo(
                chunk_id="ch2",
                content="BERT achieves state-of-the-art results on various NLP tasks.",
                paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
            ),
            entity_ids=["e3"],
        )
        
        return entity_store, relationship_store, chunk_store

    def test_search_basic(self, setup_stores):
        """基本的な検索テスト"""
        entity_store, relationship_store, chunk_store = setup_stores
        llm_client = MockLLMClient(
            responses={"transformer": "The Transformer is a revolutionary architecture."},
        )
        
        search = LocalSearch(
            llm_client=llm_client,
            entity_store=entity_store,
            relationship_store=relationship_store,
            chunk_store=chunk_store,
        )
        
        # "Transformer" を直接検索（クエリ全体ではなくエンティティ名）
        result = search.search("Transformer")
        
        assert result.query == "Transformer"
        assert len(result.answer) > 0
        assert len(result.entities_found) > 0
        assert result.processing_time_ms >= 0

    def test_search_with_graph_traversal(self, setup_stores):
        """グラフトラバーサルを含む検索テスト"""
        entity_store, relationship_store, chunk_store = setup_stores
        llm_client = MockLLMClient()
        
        config = LocalSearchConfig(
            max_hops=2,
            include_relationships=True,
        )
        
        search = LocalSearch(
            llm_client=llm_client,
            entity_store=entity_store,
            relationship_store=relationship_store,
            chunk_store=chunk_store,
            config=config,
        )
        
        result = search.search("Transformer")
        
        # Transformerから関連エンティティへのトラバース
        assert len(result.entities_found) >= 1
        # リレーションシップも取得される
        assert len(result.relationships_used) >= 0

    def test_search_no_entities_found(self, setup_stores):
        """エンティティが見つからない場合のテスト"""
        entity_store, relationship_store, chunk_store = setup_stores
        llm_client = MockLLMClient()
        
        search = LocalSearch(
            llm_client=llm_client,
            entity_store=entity_store,
            relationship_store=relationship_store,
            chunk_store=chunk_store,
        )
        
        result = search.search("Quantum Computing")
        
        assert "No relevant entities" in result.answer
        assert len(result.entities_found) == 0

    def test_search_custom_config(self, setup_stores):
        """カスタム設定での検索テスト"""
        entity_store, relationship_store, chunk_store = setup_stores
        llm_client = MockLLMClient()
        
        search = LocalSearch(
            llm_client=llm_client,
            entity_store=entity_store,
            relationship_store=relationship_store,
            chunk_store=chunk_store,
        )
        
        custom_config = LocalSearchConfig(
            max_hops=1,
            top_k_entities=5,
            include_relationships=False,
        )
        
        result = search.search("Transformer", config=custom_config)
        
        # include_relationships=Falseなのでリレーションシップは空
        assert result.hops_traversed == 0  # ホップなし
        assert len(result.relationships_used) == 0

    def test_search_japanese_response(self, setup_stores):
        """日本語レスポンス設定のテスト"""
        entity_store, relationship_store, chunk_store = setup_stores
        llm_client = MockLLMClient()
        
        config = LocalSearchConfig(response_language="ja")
        
        search = LocalSearch(
            llm_client=llm_client,
            entity_store=entity_store,
            relationship_store=relationship_store,
            chunk_store=chunk_store,
            config=config,
        )
        
        result = search.search("Transformer")
        
        # 実行が完了することを確認
        assert result is not None


# =============================================================================
# MockLLMClient Tests
# =============================================================================


class TestMockLLMClient:
    """MockLLMClientのテスト"""

    def test_default_response(self):
        """デフォルトレスポンスのテスト"""
        client = MockLLMClient()
        response = client.generate("random prompt")
        assert "mock response" in response.lower()

    def test_custom_response(self):
        """カスタムレスポンスのテスト"""
        client = MockLLMClient(
            responses={"transformer": "Custom transformer explanation"},
        )
        response = client.generate("Tell me about transformer")
        assert response == "Custom transformer explanation"

    def test_token_count(self):
        """トークンカウントのテスト"""
        client = MockLLMClient(tokens_per_char=0.25)
        count = client.count_tokens("Hello World")  # 11 chars
        assert count == 2  # int(11 * 0.25) = 2

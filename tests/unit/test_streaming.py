"""Unit tests for Streaming Response - REQ-QRY-006."""

import asyncio
import time
from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.search.base import Citation, SearchHit, SearchResults, SynthesizedAnswer
from monjyu.search.streaming.types import (
    StreamCallbacks,
    StreamChunk,
    StreamChunkType,
    StreamConfig,
    StreamMetadata,
    StreamState,
)
from monjyu.search.streaming.synthesizer import (
    MockStreamingLLMClient,
    StreamingAnswerSynthesizer,
    create_streaming_synthesizer,
)
from monjyu.search.streaming.engine import (
    StreamingSearchEngine,
    StreamingSearchConfig,
    MultiEngineStreamingSearch,
    create_streaming_engine,
    create_multi_engine_streaming,
)


# ========== Fixtures ==========


@pytest.fixture
def sample_search_hits() -> List[SearchHit]:
    """テスト用検索ヒット"""
    return [
        SearchHit(
            text_unit_id="tu_001",
            document_id="doc_001",
            text="GraphRAGは知識グラフとLLMを組み合わせた検索手法です。",
            score=0.95,
            document_title="GraphRAG入門",
        ),
        SearchHit(
            text_unit_id="tu_002",
            document_id="doc_002",
            text="LazyGraphRAGは効率的なコミュニティ構造を活用します。",
            score=0.85,
            document_title="LazyGraphRAG論文",
        ),
        SearchHit(
            text_unit_id="tu_003",
            document_id="doc_001",
            text="ベクトル検索はセマンティック類似度を計算します。",
            score=0.75,
            document_title="GraphRAG入門",
        ),
    ]


@pytest.fixture
def mock_streaming_llm() -> MockStreamingLLMClient:
    """モックストリーミングLLMクライアント"""
    return MockStreamingLLMClient(delay_ms=10)


@pytest.fixture
def stream_config() -> StreamConfig:
    """ストリーミング設定"""
    return StreamConfig(
        chunk_size=5,
        include_citations=True,
        include_search_results=True,
    )


@pytest.fixture
def mock_search_engine(sample_search_hits: List[SearchHit]) -> MagicMock:
    """モック検索エンジン"""
    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=SearchResults(
            hits=sample_search_hits,
            total_count=len(sample_search_hits),
            search_time_ms=50.0,
        )
    )
    return engine


# ========== StreamConfig Tests ==========


class TestStreamConfig:
    """StreamConfig テスト"""
    
    def test_default_config(self):
        """デフォルト設定"""
        config = StreamConfig()
        
        assert config.chunk_size == 10
        assert config.include_citations is True
        assert config.include_search_results is True
        assert config.search_timeout_seconds == 30.0
        assert config.synthesis_timeout_seconds == 60.0
    
    def test_custom_config(self):
        """カスタム設定"""
        config = StreamConfig(
            chunk_size=5,
            include_citations=False,
            search_timeout_seconds=10.0,
        )
        
        assert config.chunk_size == 5
        assert config.include_citations is False
        assert config.search_timeout_seconds == 10.0
    
    def test_to_dict(self):
        """辞書変換"""
        config = StreamConfig(chunk_size=20)
        data = config.to_dict()
        
        assert data["chunk_size"] == 20
        assert "include_citations" in data
    
    def test_from_dict(self):
        """辞書から復元"""
        data = {"chunk_size": 15, "include_citations": False}
        config = StreamConfig.from_dict(data)
        
        assert config.chunk_size == 15
        assert config.include_citations is False


# ========== StreamMetadata Tests ==========


class TestStreamMetadata:
    """StreamMetadata テスト"""
    
    def test_default_metadata(self):
        """デフォルトメタデータ"""
        metadata = StreamMetadata()
        
        assert metadata.query == ""
        assert metadata.state == StreamState.IDLE
        assert metadata.total_search_hits == 0
    
    def test_search_time_calculation(self):
        """検索時間計算"""
        metadata = StreamMetadata()
        metadata.search_start_time = 1000.0
        metadata.search_end_time = 1000.5
        
        assert metadata.search_time_ms == 500.0
    
    def test_synthesis_time_calculation(self):
        """合成時間計算"""
        metadata = StreamMetadata()
        metadata.synthesis_start_time = 1000.0
        metadata.synthesis_end_time = 1002.0
        
        assert metadata.synthesis_time_ms == 2000.0
    
    def test_to_dict(self):
        """辞書変換"""
        metadata = StreamMetadata(
            query="test query",
            state=StreamState.COMPLETED,
            total_search_hits=10,
        )
        data = metadata.to_dict()
        
        assert data["query"] == "test query"
        assert data["state"] == "completed"
        assert data["total_search_hits"] == 10


# ========== StreamChunk Tests ==========


class TestStreamChunk:
    """StreamChunk テスト"""
    
    def test_text_chunk(self):
        """テキストチャンク作成"""
        chunk = StreamChunk.text_chunk("Hello world", 0)
        
        assert chunk.chunk_type == StreamChunkType.TEXT
        assert chunk.content == "Hello world"
        assert chunk.index == 0
    
    def test_citation_chunk(self):
        """引用チャンク作成"""
        chunk = StreamChunk.citation_chunk(
            citation_id="1",
            document_title="Test Doc",
            text_snippet="Sample text",
            index=5,
        )
        
        assert chunk.chunk_type == StreamChunkType.CITATION
        assert chunk.content == "[1]"
        assert chunk.data["document_title"] == "Test Doc"
    
    def test_error_chunk(self):
        """エラーチャンク作成"""
        chunk = StreamChunk.error_chunk("Connection failed", 10)
        
        assert chunk.chunk_type == StreamChunkType.ERROR
        assert chunk.content == "Connection failed"
        assert chunk.data["error"] == "Connection failed"
    
    def test_end_chunk(self):
        """終了チャンク作成"""
        metadata = StreamMetadata(
            query="test",
            state=StreamState.COMPLETED,
        )
        chunk = StreamChunk.end_chunk(metadata, 20)
        
        assert chunk.chunk_type == StreamChunkType.END
        assert chunk.metadata == metadata
    
    def test_to_dict(self):
        """辞書変換"""
        chunk = StreamChunk.text_chunk("test", 1)
        data = chunk.to_dict()
        
        assert data["chunk_type"] == "text"
        assert data["content"] == "test"
        assert data["index"] == 1
    
    def test_from_dict(self):
        """辞書から復元"""
        data = {
            "chunk_type": "text",
            "content": "restored",
            "index": 5,
            "timestamp": 1000.0,
        }
        chunk = StreamChunk.from_dict(data)
        
        assert chunk.chunk_type == StreamChunkType.TEXT
        assert chunk.content == "restored"
        assert chunk.index == 5


# ========== StreamCallbacks Tests ==========


class TestStreamCallbacks:
    """StreamCallbacks テスト"""
    
    def test_has_callbacks_empty(self):
        """空のコールバック"""
        callbacks = StreamCallbacks()
        assert callbacks.has_callbacks() is False
    
    def test_has_callbacks_with_on_chunk(self):
        """on_chunkコールバックあり"""
        async def on_chunk(chunk):
            pass
        
        callbacks = StreamCallbacks(on_chunk=on_chunk)
        assert callbacks.has_callbacks() is True
    
    def test_has_callbacks_with_sync(self):
        """同期コールバックあり"""
        callbacks = StreamCallbacks(on_chunk_sync=lambda x: None)
        assert callbacks.has_callbacks() is True


# ========== MockStreamingLLMClient Tests ==========


class TestMockStreamingLLMClient:
    """MockStreamingLLMClient テスト"""
    
    def test_model_name(self, mock_streaming_llm: MockStreamingLLMClient):
        """モデル名"""
        assert mock_streaming_llm.model_name == "mock-streaming-llm"
    
    @pytest.mark.asyncio
    async def test_stream_generate(self, mock_streaming_llm: MockStreamingLLMClient):
        """ストリーミング生成"""
        tokens = []
        async for token in mock_streaming_llm.stream_generate("test prompt"):
            tokens.append(token)
        
        full_text = "".join(tokens)
        assert len(tokens) > 0
        assert "[1]" in full_text  # デフォルト応答には引用が含まれる
    
    @pytest.mark.asyncio
    async def test_stream_generate_custom_response(self):
        """カスタム応答"""
        client = MockStreamingLLMClient(delay_ms=5)
        client.set_response("GraphRAG", "GraphRAG is great. [1]")
        
        tokens = []
        async for token in client.stream_generate("Tell me about GraphRAG"):
            tokens.append(token)
        
        full_text = "".join(tokens)
        assert "great" in full_text


# ========== StreamingAnswerSynthesizer Tests ==========


class TestStreamingAnswerSynthesizer:
    """StreamingAnswerSynthesizer テスト"""
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_empty_context(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
        stream_config: StreamConfig,
    ):
        """空コンテキストでのストリーミング"""
        synthesizer = StreamingAnswerSynthesizer(
            llm_client=mock_streaming_llm,
            config=stream_config,
        )
        
        chunks = []
        async for chunk in synthesizer.synthesize_stream("test query", []):
            chunks.append(chunk)
        
        # 開始、テキスト、完了チャンクが含まれる
        chunk_types = [c.chunk_type for c in chunks]
        assert StreamChunkType.SYNTHESIS_START in chunk_types
        assert StreamChunkType.TEXT in chunk_types
        assert StreamChunkType.SYNTHESIS_COMPLETE in chunk_types
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_with_context(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
        stream_config: StreamConfig,
        sample_search_hits: List[SearchHit],
    ):
        """コンテキストありでのストリーミング"""
        synthesizer = StreamingAnswerSynthesizer(
            llm_client=mock_streaming_llm,
            config=stream_config,
        )
        
        chunks = []
        async for chunk in synthesizer.synthesize_stream(
            "What is GraphRAG?",
            sample_search_hits,
        ):
            chunks.append(chunk)
        
        # テキストチャンクが複数含まれる
        text_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.TEXT]
        assert len(text_chunks) > 0
        
        # 完了チャンク
        complete_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.SYNTHESIS_COMPLETE]
        assert len(complete_chunks) == 1
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_with_callbacks(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
        stream_config: StreamConfig,
        sample_search_hits: List[SearchHit],
    ):
        """コールバック付きストリーミング"""
        synthesizer = StreamingAnswerSynthesizer(
            llm_client=mock_streaming_llm,
            config=stream_config,
        )
        
        received_chunks = []
        completed = False
        
        async def on_chunk(chunk):
            received_chunks.append(chunk)
        
        def on_complete(metadata):
            nonlocal completed
            completed = True
        
        callbacks = StreamCallbacks(
            on_chunk=on_chunk,
            on_complete=on_complete,
        )
        
        async for _ in synthesizer.synthesize_stream(
            "test",
            sample_search_hits,
            callbacks=callbacks,
        ):
            pass
        
        assert len(received_chunks) > 0
        assert completed is True
    
    @pytest.mark.asyncio
    async def test_synthesize_non_streaming(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
        stream_config: StreamConfig,
        sample_search_hits: List[SearchHit],
    ):
        """非ストリーミング合成"""
        synthesizer = StreamingAnswerSynthesizer(
            llm_client=mock_streaming_llm,
            config=stream_config,
        )
        
        answer = await synthesizer.synthesize("test query", sample_search_hits)
        
        assert isinstance(answer, SynthesizedAnswer)
        assert len(answer.answer) > 0
        assert answer.model == "mock-streaming-llm"
    
    @pytest.mark.asyncio
    async def test_cancel_streaming(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
        stream_config: StreamConfig,
        sample_search_hits: List[SearchHit],
    ):
        """ストリーミングキャンセル"""
        synthesizer = StreamingAnswerSynthesizer(
            llm_client=mock_streaming_llm,
            config=stream_config,
        )
        
        chunks = []
        async for chunk in synthesizer.synthesize_stream(
            "test",
            sample_search_hits,
        ):
            chunks.append(chunk)
            if len(chunks) >= 2:
                synthesizer.cancel()
                break
        
        # キャンセル前にいくつかのチャンクを受信
        assert len(chunks) >= 2


# ========== StreamingSearchEngine Tests ==========


class TestStreamingSearchEngine:
    """StreamingSearchEngine テスト"""
    
    @pytest.mark.asyncio
    async def test_search_stream(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """ストリーミング検索"""
        config = StreamingSearchConfig(
            top_k=10,
            synthesize=True,
        )
        
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
            config=config,
        )
        
        chunks = []
        async for chunk in engine.search_stream("What is GraphRAG?"):
            chunks.append(chunk)
        
        # 必須チャンクが含まれる
        chunk_types = [c.chunk_type for c in chunks]
        assert StreamChunkType.SEARCH_START in chunk_types
        assert StreamChunkType.SEARCH_COMPLETE in chunk_types
        assert StreamChunkType.SYNTHESIS_START in chunk_types
        assert StreamChunkType.END in chunk_types
    
    @pytest.mark.asyncio
    async def test_search_stream_with_results(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """検索結果ストリーミング"""
        config = StreamingSearchConfig(
            stream_search_results=True,
            stream_result_batch_size=2,
        )
        
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
            config=config,
        )
        
        chunks = []
        async for chunk in engine.search_stream("test"):
            chunks.append(chunk)
        
        # 検索結果チャンクが含まれる
        result_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.SEARCH_RESULT]
        assert len(result_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_search_non_streaming(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """非ストリーミング検索"""
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
        )
        
        result = await engine.search("What is GraphRAG?")
        
        assert result.search_results is not None
        assert len(result.search_results.hits) == 3
        assert result.answer is not None
        assert result.metadata is not None
        assert result.metadata.state == StreamState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_search_without_synthesis(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """合成なし検索"""
        config = StreamingSearchConfig(synthesize=False)
        
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
            config=config,
        )
        
        chunks = []
        async for chunk in engine.search_stream("test"):
            chunks.append(chunk)
        
        # 合成関連チャンクがない
        chunk_types = [c.chunk_type for c in chunks]
        assert StreamChunkType.SYNTHESIS_START not in chunk_types
    
    @pytest.mark.asyncio
    async def test_metadata_access(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """メタデータアクセス"""
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
        )
        
        async for _ in engine.search_stream("test"):
            pass
        
        metadata = engine.metadata
        assert metadata is not None
        assert metadata.query == "test"
        assert metadata.total_search_hits == 3


# ========== MultiEngineStreamingSearch Tests ==========


class TestMultiEngineStreamingSearch:
    """MultiEngineStreamingSearch テスト"""
    
    @pytest.fixture
    def mock_engines(self, sample_search_hits: List[SearchHit]) -> dict:
        """複数モックエンジン"""
        engine1 = MagicMock()
        engine1.search = AsyncMock(
            return_value=SearchResults(
                hits=sample_search_hits[:2],
                total_count=2,
            )
        )
        
        engine2 = MagicMock()
        engine2.search = AsyncMock(
            return_value=SearchResults(
                hits=sample_search_hits[1:],
                total_count=2,
            )
        )
        
        return {"vector": engine1, "lazy": engine2}
    
    @pytest.mark.asyncio
    async def test_multi_engine_search_stream(
        self,
        mock_engines: dict,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """マルチエンジンストリーミング検索"""
        multi_search = MultiEngineStreamingSearch(
            engines=mock_engines,
            llm_client=mock_streaming_llm,
        )
        
        chunks = []
        async for chunk in multi_search.search_stream("test query"):
            chunks.append(chunk)
        
        # 開始チャンクにエンジン情報が含まれる
        start_chunk = next(
            c for c in chunks if c.chunk_type == StreamChunkType.SEARCH_START
        )
        assert "vector" in start_chunk.data["engines"]
        assert "lazy" in start_chunk.data["engines"]
        
        # 進捗チャンクが含まれる
        progress_chunks = [
            c for c in chunks if c.chunk_type == StreamChunkType.SEARCH_PROGRESS
        ]
        assert len(progress_chunks) >= 2  # 各エンジン完了
    
    @pytest.mark.asyncio
    async def test_multi_engine_selected_engines(
        self,
        mock_engines: dict,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """特定エンジンのみ使用"""
        multi_search = MultiEngineStreamingSearch(
            engines=mock_engines,
            llm_client=mock_streaming_llm,
        )
        
        chunks = []
        async for chunk in multi_search.search_stream(
            "test",
            engine_names=["vector"],
        ):
            chunks.append(chunk)
        
        # vectorエンジンのみ
        start_chunk = next(
            c for c in chunks if c.chunk_type == StreamChunkType.SEARCH_START
        )
        assert start_chunk.data["engines"] == ["vector"]


# ========== Factory Function Tests ==========


class TestFactoryFunctions:
    """ファクトリー関数テスト"""
    
    def test_create_streaming_synthesizer(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """create_streaming_synthesizer"""
        synthesizer = create_streaming_synthesizer(mock_streaming_llm)
        
        assert isinstance(synthesizer, StreamingAnswerSynthesizer)
    
    def test_create_streaming_engine(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """create_streaming_engine"""
        engine = create_streaming_engine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
        )
        
        assert isinstance(engine, StreamingSearchEngine)
    
    def test_create_multi_engine_streaming(
        self,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """create_multi_engine_streaming"""
        engines = {"test": MagicMock()}
        
        multi = create_multi_engine_streaming(
            engines=engines,
            llm_client=mock_streaming_llm,
        )
        
        assert isinstance(multi, MultiEngineStreamingSearch)


# ========== Integration Tests ==========


class TestStreamingIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_streaming_pipeline(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """完全なストリーミングパイプライン"""
        config = StreamingSearchConfig(
            top_k=5,
            stream_config=StreamConfig(chunk_size=3),
            notify_search_progress=True,
            stream_search_results=True,
        )
        
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
            config=config,
        )
        
        # コールバック
        received_chunks = []
        errors = []
        completed_metadata = None
        
        async def on_chunk(chunk):
            received_chunks.append(chunk)
        
        def on_error(error):
            errors.append(error)
        
        def on_complete(metadata):
            nonlocal completed_metadata
            completed_metadata = metadata
        
        callbacks = StreamCallbacks(
            on_chunk=on_chunk,
            on_error=on_error,
            on_complete=on_complete,
        )
        
        # 検索実行
        chunks = []
        async for chunk in engine.search_stream(
            "What is GraphRAG?",
            callbacks=callbacks,
        ):
            chunks.append(chunk)
        
        # 検証
        assert len(chunks) > 0
        assert len(errors) == 0
        assert completed_metadata is not None
        assert completed_metadata.state == StreamState.COMPLETED
        
        # チャンク順序確認
        chunk_types = [c.chunk_type for c in chunks]
        assert chunk_types[0] == StreamChunkType.SEARCH_START
        assert chunk_types[-1] == StreamChunkType.END
    
    @pytest.mark.asyncio
    async def test_streaming_with_all_chunk_types(
        self,
        mock_search_engine: MagicMock,
        mock_streaming_llm: MockStreamingLLMClient,
    ):
        """全チャンクタイプの出現確認"""
        config = StreamingSearchConfig(
            stream_search_results=True,
            notify_search_progress=True,
        )
        
        engine = StreamingSearchEngine(
            search_engine=mock_search_engine,
            llm_client=mock_streaming_llm,
            config=config,
        )
        
        chunk_types_seen = set()
        async for chunk in engine.search_stream("test"):
            chunk_types_seen.add(chunk.chunk_type)
        
        # 必須チャンクタイプ
        expected = {
            StreamChunkType.SEARCH_START,
            StreamChunkType.SEARCH_PROGRESS,
            StreamChunkType.SEARCH_RESULT,
            StreamChunkType.SEARCH_COMPLETE,
            StreamChunkType.SYNTHESIS_START,
            StreamChunkType.TEXT,
            StreamChunkType.SYNTHESIS_COMPLETE,
            StreamChunkType.END,
        }
        
        assert expected.issubset(chunk_types_seen)

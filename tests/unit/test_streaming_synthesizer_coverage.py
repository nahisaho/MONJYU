"""Streaming Synthesizer カバレッジ向上テスト"""

import asyncio
import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from monjyu.search.base import Citation, SearchHit, SynthesizedAnswer
from monjyu.search.streaming.types import (
    StreamCallbacks,
    StreamChunk,
    StreamChunkType,
    StreamConfig,
    StreamMetadata,
    StreamState,
)
from monjyu.search.streaming.synthesizer import (
    StreamingAnswerSynthesizer,
    StreamingLLMProtocol,
    StreamingOllamaClient,
    StreamingOpenAIClient,
    StreamingAzureOpenAIClient,
    MockStreamingLLMClient,
    create_streaming_synthesizer,
)


# ========== Fixtures ==========


@pytest.fixture
def sample_search_hits() -> list[SearchHit]:
    """テスト用検索ヒット"""
    return [
        SearchHit(
            text_unit_id="tu_001",
            document_id="doc_001",
            text="GraphRAGは知識グラフとLLMを組み合わせた検索手法です。これは非常に長いテキストで、200文字を超えるため、引用抽出時にスニペットが切り詰められます。" + "追加テキスト " * 30,
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
            document_title=None,  # Noneのケースをテスト
        ),
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """モックLLMクライアント"""
    mock = MagicMock(spec=StreamingLLMProtocol)
    mock.model_name = "test-model"
    
    async def mock_stream_generate(prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        response = "これはテスト回答です。[1] GraphRAGについて説明します。[2] LazyGraphRAGも重要です。[3]"
        for word in response.split():
            yield word + " "
    
    mock.stream_generate = mock_stream_generate
    return mock


@pytest.fixture
def synthesizer(mock_llm: MagicMock) -> StreamingAnswerSynthesizer:
    """テスト用シンセサイザー"""
    return StreamingAnswerSynthesizer(
        llm_client=mock_llm,
        config=StreamConfig(chunk_size=10, include_citations=True),
    )


# ========== StreamingAnswerSynthesizer詳細テスト ==========


class TestStreamingAnswerSynthesizerDetailedCoverage:
    """StreamingAnswerSynthesizerの詳細カバレッジ"""
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_with_sync_callback(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """on_chunk_sync コールバックのテスト"""
        sync_chunks = []
        
        def on_chunk_sync(chunk: StreamChunk):
            sync_chunks.append(chunk)
        
        callbacks = StreamCallbacks(
            on_chunk=AsyncMock(),
            on_chunk_sync=on_chunk_sync,
        )
        
        async for _ in synthesizer.synthesize_stream(
            query="GraphRAGとは？",
            context=sample_search_hits,
            callbacks=callbacks,
        ):
            pass
        
        # 同期コールバックが呼ばれている
        assert len(sync_chunks) > 0
        assert any(chunk.chunk_type == StreamChunkType.TEXT for chunk in sync_chunks)
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_citation_extraction_long_text(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """200文字超のテキストの引用スニペット切り詰めテスト"""
        chunks = []
        
        async for chunk in synthesizer.synthesize_stream(
            query="GraphRAGとは？",
            context=sample_search_hits,
        ):
            chunks.append(chunk)
        
        # 引用チャンクを確認
        citation_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.CITATION]
        
        # 長いテキストの引用はスニペットが切り詰められている
        if citation_chunks:
            for cc in citation_chunks:
                if cc.data and "text_snippet" in cc.data:
                    snippet = cc.data.get("text_snippet", "")
                    # 200文字超は切り詰められる
                    assert len(snippet) <= 203  # 200 + "..."
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_null_document_title(
        self, synthesizer: StreamingAnswerSynthesizer
    ):
        """document_titleがNoneの場合のテスト"""
        hits = [
            SearchHit(
                text_unit_id="tu_001",
                document_id="doc_001",
                text="テストテキスト",
                score=0.95,
                document_title=None,  # Noneをテスト
            )
        ]
        
        chunks = []
        async for chunk in synthesizer.synthesize_stream(
            query="テスト",
            context=hits,
        ):
            chunks.append(chunk)
        
        # エラーなく完了
        assert any(c.chunk_type == StreamChunkType.SYNTHESIS_COMPLETE for c in chunks)
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_error_handling(self, sample_search_hits: list[SearchHit]):
        """エラーハンドリングのテスト"""
        error_mock = MagicMock(spec=StreamingLLMProtocol)
        error_mock.model_name = "error-model"
        
        async def error_stream_generate(prompt: str, **kwargs) -> AsyncGenerator[str, None]:
            yield "開始 "
            raise ValueError("LLMエラー")
        
        error_mock.stream_generate = error_stream_generate
        
        synthesizer = StreamingAnswerSynthesizer(llm_client=error_mock)
        
        error_callback = MagicMock()
        callbacks = StreamCallbacks(on_error=error_callback)
        
        chunks = []
        async for chunk in synthesizer.synthesize_stream(
            query="テスト",
            context=sample_search_hits,
            callbacks=callbacks,
        ):
            chunks.append(chunk)
        
        # エラーチャンクが生成される
        assert any(c.chunk_type == StreamChunkType.ERROR for c in chunks)
        # エラーコールバックが呼ばれる
        error_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_on_complete_callback(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """on_complete コールバックのテスト"""
        complete_callback = MagicMock()
        callbacks = StreamCallbacks(on_complete=complete_callback)
        
        async for _ in synthesizer.synthesize_stream(
            query="テスト",
            context=sample_search_hits,
            callbacks=callbacks,
        ):
            pass
        
        # on_completeが呼ばれる
        complete_callback.assert_called_once()
        # メタデータが渡される
        args = complete_callback.call_args
        assert isinstance(args[0][0], StreamMetadata)
    
    @pytest.mark.asyncio
    async def test_synthesize_stream_custom_system_prompt(
        self, sample_search_hits: list[SearchHit]
    ):
        """カスタムシステムプロンプトのテスト"""
        captured_prompts = []
        
        mock_llm = MagicMock(spec=StreamingLLMProtocol)
        mock_llm.model_name = "test-model"
        
        async def capture_stream_generate(prompt: str, system_prompt: str | None = None, **kwargs):
            captured_prompts.append({"prompt": prompt, "system_prompt": system_prompt})
            yield "回答 "
        
        mock_llm.stream_generate = capture_stream_generate
        
        synthesizer = StreamingAnswerSynthesizer(llm_client=mock_llm)
        
        custom_system = "カスタムシステムプロンプト"
        
        async for _ in synthesizer.synthesize_stream(
            query="テスト",
            context=sample_search_hits,
            system_prompt=custom_system,
        ):
            pass
        
        assert len(captured_prompts) == 1
        assert captured_prompts[0]["system_prompt"] == custom_system
    
    @pytest.mark.asyncio
    async def test_synthesize_with_citations(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """synthesize()メソッド（非ストリーミング）で引用抽出のテスト"""
        result = await synthesizer.synthesize(
            query="GraphRAGとは？",
            context=sample_search_hits,
        )
        
        assert isinstance(result, SynthesizedAnswer)
        assert len(result.citations) > 0
        assert result.tokens_used > 0
        assert result.model == "test-model"
    
    def test_extract_citations_with_invalid_indices(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """無効な引用インデックスのテスト"""
        response = "回答文 [1] [5] [10] [0]"  # [5], [10], [0]は無効
        
        _, citations = synthesizer._extract_citations(response, sample_search_hits)
        
        # 有効な引用のみ抽出される
        assert len(citations) == 1  # [1]のみ有効
    
    def test_estimate_confidence_no_citations(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """引用なしの場合の信頼度推定"""
        confidence = synthesizer._estimate_confidence([], sample_search_hits)
        assert confidence == 0.0
    
    def test_estimate_confidence_with_citations(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """引用ありの場合の信頼度推定"""
        citations = [
            Citation(
                text_unit_id="tu_001",
                document_id="doc_001",
                document_title="Test",
                text_snippet="snippet",
                relevance_score=0.9,
            )
        ]
        
        confidence = synthesizer._estimate_confidence(citations, sample_search_hits)
        assert 0 < confidence <= 1.0
    
    def test_build_context(
        self, synthesizer: StreamingAnswerSynthesizer, sample_search_hits: list[SearchHit]
    ):
        """コンテキスト構築のテスト"""
        context_text = synthesizer._build_context(sample_search_hits)
        
        # インデックスが含まれる
        assert "[1]" in context_text
        assert "[2]" in context_text
        
        # スコアが含まれる
        assert "Score:" in context_text
        
        # ドキュメントタイトルが含まれる（Noneの場合はDocument）
        assert "GraphRAG入門" in context_text
        assert "Document" in context_text  # Noneの場合のフォールバック


# ========== StreamingOllamaClient テスト ==========


class TestStreamingOllamaClient:
    """StreamingOllamaClient テスト"""
    
    def test_init(self):
        """初期化テスト"""
        client = StreamingOllamaClient(
            model="llama3.1:8b",
            host="http://localhost:11434",
        )
        
        assert client.model_name == "llama3.1:8b"
        assert client.host == "http://localhost:11434"
    
    def test_client_property_import_error(self):
        """ollamaインポートエラーのテスト"""
        client = StreamingOllamaClient()
        
        with patch.dict("sys.modules", {"ollama": None}):
            with patch("builtins.__import__", side_effect=ImportError("no ollama")):
                with pytest.raises(RuntimeError, match="ollama package not installed"):
                    _ = client.client
    
    @pytest.mark.asyncio
    async def test_stream_generate_with_mock(self):
        """モック化したstream_generateのテスト"""
        client = StreamingOllamaClient()
        
        # AsyncClientをモック
        mock_async_client = MagicMock()
        
        async def mock_chat(**kwargs):
            # awaitableなジェネレータを返す
            async def gen():
                for word in ["Hello", " ", "World"]:
                    yield {"message": {"content": word}}
            return gen()
        
        mock_async_client.chat = AsyncMock(side_effect=mock_chat)
        client._client = mock_async_client
        
        tokens = []
        async for token in client.stream_generate(
            prompt="テスト",
            system_prompt="システムプロンプト",
            max_tokens=100,
        ):
            tokens.append(token)
        
        assert len(tokens) == 3
        assert "".join(tokens) == "Hello World"


# ========== StreamingOpenAIClient テスト ==========


class TestStreamingOpenAIClient:
    """StreamingOpenAIClient テスト"""
    
    def test_init(self):
        """初期化テスト"""
        client = StreamingOpenAIClient(
            model="gpt-4o-mini",
            api_key="test-key",
        )
        
        assert client.model_name == "gpt-4o-mini"
    
    def test_client_property_import_error(self):
        """openaiインポートエラーのテスト"""
        client = StreamingOpenAIClient()
        
        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("no openai")):
                with pytest.raises(RuntimeError, match="openai package not installed"):
                    _ = client.client
    
    @pytest.mark.asyncio
    async def test_stream_generate_with_mock(self):
        """モック化したstream_generateのテスト"""
        client = StreamingOpenAIClient()
        
        # AsyncOpenAIをモック
        mock_async_client = MagicMock()
        
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " World"
        
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = None  # 終端
        
        async def mock_create(**kwargs):
            async def gen():
                for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
                    yield chunk
            return gen()
        
        mock_async_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        client._client = mock_async_client
        
        tokens = []
        async for token in client.stream_generate(
            prompt="テスト",
            system_prompt="システム",
            max_tokens=100,
        ):
            tokens.append(token)
        
        assert len(tokens) == 2
        assert "".join(tokens) == "Hello World"


# ========== StreamingAzureOpenAIClient テスト ==========


class TestStreamingAzureOpenAIClient:
    """StreamingAzureOpenAIClient テスト"""
    
    def test_init(self):
        """初期化テスト"""
        client = StreamingAzureOpenAIClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment_name="gpt-4",
            api_version="2024-02-01",
        )
        
        assert client.model_name == "gpt-4"
        assert client.endpoint == "https://test.openai.azure.com"
        assert client.api_version == "2024-02-01"
    
    def test_client_property_import_error(self):
        """openaiインポートエラーのテスト"""
        client = StreamingAzureOpenAIClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment_name="gpt-4",
        )
        
        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("no openai")):
                with pytest.raises(RuntimeError, match="openai package not installed"):
                    _ = client.client
    
    @pytest.mark.asyncio
    async def test_stream_generate_with_mock(self):
        """モック化したstream_generateのテスト"""
        client = StreamingAzureOpenAIClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment_name="gpt-4",
        )
        
        # AsyncAzureOpenAIをモック
        mock_async_client = MagicMock()
        
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Azure"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " Response"
        
        async def mock_create(**kwargs):
            async def gen():
                for chunk in [mock_chunk1, mock_chunk2]:
                    yield chunk
            return gen()
        
        mock_async_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        client._client = mock_async_client
        
        tokens = []
        async for token in client.stream_generate(
            prompt="テスト",
            max_tokens=50,
        ):
            tokens.append(token)
        
        assert len(tokens) == 2
        assert "".join(tokens) == "Azure Response"


# ========== MockStreamingLLMClient追加テスト ==========


class TestMockStreamingLLMClientAdditional:
    """MockStreamingLLMClient 追加テスト"""
    
    def test_set_response(self):
        """応答パターン設定のテスト"""
        client = MockStreamingLLMClient()
        client.set_response("テスト", "カスタム応答です。")
        
        assert "テスト" in client._responses
        assert client._responses["テスト"] == "カスタム応答です。"
    
    @pytest.mark.asyncio
    async def test_stream_generate_with_custom_response(self):
        """カスタム応答パターンでのストリーミングテスト"""
        client = MockStreamingLLMClient(delay_ms=0)
        client.set_response("特定の質問", "特定の回答です。")
        
        tokens = []
        async for token in client.stream_generate(
            prompt="これは特定の質問です",
            system_prompt="システム",
        ):
            tokens.append(token)
        
        result = "".join(tokens).strip()
        assert result == "特定の回答です。"
    
    @pytest.mark.asyncio
    async def test_stream_generate_no_delay(self):
        """delay_ms=0でのストリーミングテスト"""
        client = MockStreamingLLMClient(delay_ms=0)
        
        tokens = []
        async for token in client.stream_generate(prompt="テスト"):
            tokens.append(token)
        
        assert len(tokens) > 0


# ========== create_streaming_synthesizer テスト ==========


class TestCreateStreamingSynthesizer:
    """create_streaming_synthesizer ファクトリ関数テスト"""
    
    def test_create_with_all_params(self):
        """全パラメータ指定でのテスト"""
        mock_llm = MockStreamingLLMClient()
        config = StreamConfig(chunk_size=20)
        system_prompt = "カスタムプロンプト"
        
        synthesizer = create_streaming_synthesizer(
            llm_client=mock_llm,
            config=config,
            system_prompt=system_prompt,
        )
        
        assert isinstance(synthesizer, StreamingAnswerSynthesizer)
        assert synthesizer.config.chunk_size == 20
        assert synthesizer.system_prompt == system_prompt
    
    def test_create_with_minimal_params(self):
        """最小パラメータでのテスト"""
        mock_llm = MockStreamingLLMClient()
        
        synthesizer = create_streaming_synthesizer(llm_client=mock_llm)
        
        assert isinstance(synthesizer, StreamingAnswerSynthesizer)
        assert synthesizer.config is not None
        assert synthesizer.system_prompt == StreamingAnswerSynthesizer.DEFAULT_SYSTEM_PROMPT


# ========== バッファフラッシュテスト ==========


class TestBufferFlush:
    """バッファフラッシュのテスト"""
    
    @pytest.mark.asyncio
    async def test_chunk_size_flush(self):
        """チャンクサイズによるフラッシュのテスト"""
        mock_llm = MagicMock(spec=StreamingLLMProtocol)
        mock_llm.model_name = "test-model"
        
        # 長いトークンを生成
        async def long_stream_generate(prompt: str, **kwargs):
            for _ in range(50):  # 50回繰り返し
                yield "abcdefghij"  # 10文字ずつ
        
        mock_llm.stream_generate = long_stream_generate
        
        # chunk_size=20で設定
        synthesizer = StreamingAnswerSynthesizer(
            llm_client=mock_llm,
            config=StreamConfig(chunk_size=20, include_citations=False),
        )
        
        hits = [
            SearchHit(
                text_unit_id="tu_001",
                document_id="doc_001",
                text="テスト",
                score=0.95,
            )
        ]
        
        text_chunks = []
        async for chunk in synthesizer.synthesize_stream(query="テスト", context=hits):
            if chunk.chunk_type == StreamChunkType.TEXT:
                text_chunks.append(chunk.content)
        
        # 複数のテキストチャンクが生成される
        assert len(text_chunks) > 1
        
        # 各チャンクのサイズを確認（最後以外は>=chunk_size）
        for i, content in enumerate(text_chunks[:-1]):
            assert len(content) >= 20 or i == len(text_chunks) - 1

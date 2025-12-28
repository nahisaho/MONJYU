"""Streaming Answer Synthesizer - REQ-QRY-006.

ストリーミングで回答を合成する。
LLMのトークンをリアルタイムで配信。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Optional, Protocol

from monjyu.search.base import Citation, SearchHit, SynthesizedAnswer
from monjyu.search.streaming.types import (
    StreamCallbacks,
    StreamChunk,
    StreamChunkType,
    StreamConfig,
    StreamMetadata,
    StreamState,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StreamingLLMProtocol(Protocol):
    """ストリーミングLLMプロトコル"""
    
    @property
    def model_name(self) -> str:
        """モデル名"""
        ...
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """トークンをストリーミング生成"""
        ...


class StreamingAnswerSynthesizer:
    """ストリーミング回答合成"""
    
    DEFAULT_SYSTEM_PROMPT = """
あなたは学術論文の専門家です。
与えられたコンテキスト情報に基づいて、ユーザーの質問に正確かつ簡潔に回答してください。

ルール:
1. コンテキストに含まれる情報のみを使用して回答してください
2. 情報が不十分な場合は、その旨を明示してください
3. 回答には必ず引用元（Citation）を含めてください（例: [1], [2]）
4. 学術的な正確性を最優先してください
"""
    
    USER_PROMPT_TEMPLATE = """
## コンテキスト
{context}

## 質問
{query}

## 回答形式
回答を記述した後、使用した引用元を [1], [2] のように示してください。
"""
    
    def __init__(
        self,
        llm_client: StreamingLLMProtocol,
        config: Optional[StreamConfig] = None,
        system_prompt: str | None = None,
    ):
        """
        Args:
            llm_client: ストリーミングLLMクライアント
            config: ストリーミング設定
            system_prompt: システムプロンプト
        """
        self.llm_client = llm_client
        self.config = config or StreamConfig()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._metadata: Optional[StreamMetadata] = None
        self._cancelled = False
    
    async def synthesize_stream(
        self,
        query: str,
        context: list[SearchHit],
        callbacks: Optional[StreamCallbacks] = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        コンテキストから回答をストリーミング合成
        
        Args:
            query: クエリ
            context: 検索ヒット
            callbacks: コールバック
            system_prompt: システムプロンプト
            
        Yields:
            StreamChunk
        """
        self._cancelled = False
        self._metadata = StreamMetadata(
            query=query,
            state=StreamState.SYNTHESIZING,
        )
        self._metadata.synthesis_start_time = time.time()
        
        chunk_index = 0
        
        # 合成開始通知
        start_chunk = StreamChunk(
            chunk_type=StreamChunkType.SYNTHESIS_START,
            content="",
            index=chunk_index,
        )
        chunk_index += 1
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(start_chunk)
        yield start_chunk
        
        # コンテキストが空の場合
        if not context:
            no_content_chunk = StreamChunk.text_chunk(
                "情報が見つかりませんでした。",
                chunk_index,
            )
            chunk_index += 1
            if callbacks and callbacks.on_chunk:
                await callbacks.on_chunk(no_content_chunk)
            yield no_content_chunk
            
            # 完了
            self._metadata.state = StreamState.COMPLETED
            self._metadata.synthesis_end_time = time.time()
            
            complete_chunk = StreamChunk(
                chunk_type=StreamChunkType.SYNTHESIS_COMPLETE,
                content="",
                index=chunk_index,
                metadata=self._metadata,
            )
            if callbacks and callbacks.on_chunk:
                await callbacks.on_chunk(complete_chunk)
            if callbacks and callbacks.on_complete:
                callbacks.on_complete(self._metadata)
            yield complete_chunk
            return
        
        # コンテキスト構築
        context_text = self._build_context(context)
        
        # プロンプト構築
        system = system_prompt or self.system_prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_text, query=query
        )
        
        # LLMストリーミング呼び出し
        self._metadata.state = StreamState.STREAMING
        full_response = ""
        buffer = ""
        
        try:
            async for token in self.llm_client.stream_generate(
                prompt=user_prompt,
                system_prompt=system,
            ):
                if self._cancelled:
                    self._metadata.state = StreamState.CANCELLED
                    break
                
                full_response += token
                buffer += token
                self._metadata.tokens_generated += 1
                
                # チャンクサイズに達したらフラッシュ
                if len(buffer) >= self.config.chunk_size:
                    text_chunk = StreamChunk.text_chunk(buffer, chunk_index)
                    chunk_index += 1
                    self._metadata.chunks_sent += 1
                    
                    if callbacks and callbacks.on_chunk:
                        await callbacks.on_chunk(text_chunk)
                    if callbacks and callbacks.on_chunk_sync:
                        callbacks.on_chunk_sync(text_chunk)
                    
                    yield text_chunk
                    buffer = ""
            
            # 残りのバッファをフラッシュ
            if buffer:
                text_chunk = StreamChunk.text_chunk(buffer, chunk_index)
                chunk_index += 1
                self._metadata.chunks_sent += 1
                
                if callbacks and callbacks.on_chunk:
                    await callbacks.on_chunk(text_chunk)
                yield text_chunk
            
            # 引用抽出
            if self.config.include_citations:
                citations = self._extract_citation_chunks(full_response, context)
                self._metadata.citations_found = len(citations)
                
                for citation in citations:
                    citation_chunk = StreamChunk.citation_chunk(
                        citation_id=str(citation["index"]),
                        document_title=citation["document_title"],
                        text_snippet=citation["text_snippet"],
                        index=chunk_index,
                    )
                    chunk_index += 1
                    
                    if callbacks and callbacks.on_chunk:
                        await callbacks.on_chunk(citation_chunk)
                    yield citation_chunk
            
            # 完了
            self._metadata.state = StreamState.COMPLETED
            self._metadata.synthesis_end_time = time.time()
            
        except asyncio.CancelledError:
            self._metadata.state = StreamState.CANCELLED
            raise
        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            self._metadata.state = StreamState.ERROR
            self._metadata.error_message = str(e)
            
            error_chunk = StreamChunk.error_chunk(str(e), chunk_index)
            chunk_index += 1
            
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield error_chunk
            return
        
        # 合成完了通知
        complete_chunk = StreamChunk(
            chunk_type=StreamChunkType.SYNTHESIS_COMPLETE,
            content="",
            index=chunk_index,
            metadata=self._metadata,
            data={"full_response": full_response},
        )
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(complete_chunk)
        if callbacks and callbacks.on_complete:
            callbacks.on_complete(self._metadata)
        
        yield complete_chunk
    
    async def synthesize(
        self,
        query: str,
        context: list[SearchHit],
        system_prompt: str | None = None,
    ) -> SynthesizedAnswer:
        """
        コンテキストから回答を合成（非ストリーミング版）
        
        Args:
            query: クエリ
            context: 検索ヒット
            system_prompt: システムプロンプト
            
        Returns:
            合成された回答
        """
        full_response = ""
        
        async for chunk in self.synthesize_stream(query, context, None, system_prompt):
            if chunk.chunk_type == StreamChunkType.TEXT:
                full_response += chunk.content
        
        # 引用抽出
        _, citations = self._extract_citations(full_response, context)
        confidence = self._estimate_confidence(citations, context)
        
        return SynthesizedAnswer(
            answer=full_response,
            citations=citations,
            confidence=confidence,
            tokens_used=self._metadata.tokens_generated if self._metadata else 0,
            model=self.llm_client.model_name,
        )
    
    def cancel(self) -> None:
        """ストリーミングをキャンセル"""
        self._cancelled = True
    
    def _build_context(self, hits: list[SearchHit]) -> str:
        """コンテキストテキストを構築"""
        parts = []
        for i, hit in enumerate(hits):
            parts.append(f"[{i + 1}] {hit.document_title or 'Document'}")
            parts.append(f"Score: {hit.score:.3f}")
            parts.append(hit.text)
            parts.append("")
        return "\n".join(parts)
    
    def _extract_citation_chunks(
        self,
        response: str,
        context: list[SearchHit],
    ) -> list[dict]:
        """回答から引用チャンクを抽出"""
        citation_pattern = r"\[(\d+)\]"
        cited_indices = set(int(m) for m in re.findall(citation_pattern, response))
        
        citations = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(context):
                hit = context[idx - 1]
                snippet = hit.text[:200]
                if len(hit.text) > 200:
                    snippet += "..."
                
                citations.append({
                    "index": idx,
                    "text_unit_id": hit.text_unit_id,
                    "document_id": hit.document_id,
                    "document_title": hit.document_title or "Unknown Document",
                    "text_snippet": snippet,
                    "relevance_score": hit.score,
                })
        
        return citations
    
    def _extract_citations(
        self,
        response: str,
        context: list[SearchHit],
    ) -> tuple[str, list[Citation]]:
        """回答から引用を抽出"""
        citation_pattern = r"\[(\d+)\]"
        cited_indices = set(int(m) for m in re.findall(citation_pattern, response))
        
        citations = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(context):
                hit = context[idx - 1]
                snippet = hit.text[:200]
                if len(hit.text) > 200:
                    snippet += "..."
                
                citations.append(
                    Citation(
                        text_unit_id=hit.text_unit_id,
                        document_id=hit.document_id,
                        document_title=hit.document_title or "Unknown Document",
                        text_snippet=snippet,
                        relevance_score=hit.score,
                    )
                )
        
        return response, citations
    
    def _estimate_confidence(
        self,
        citations: list[Citation],
        context: list[SearchHit],
    ) -> float:
        """信頼度を推定"""
        if not citations:
            return 0.0
        
        avg_score = sum(c.relevance_score for c in citations) / len(citations)
        coverage = min(len(citations) / max(len(context), 1), 1.0)
        confidence = avg_score * (0.5 + 0.5 * coverage)
        
        return min(confidence, 1.0)


# === Streaming LLM Client Implementations ===


class StreamingOllamaClient:
    """Ollama ストリーミングLLMクライアント"""
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
    ):
        self._model = model
        self.host = host
        self._client = None
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.AsyncClient(host=self.host)
            except ImportError:
                raise RuntimeError(
                    "ollama package not installed. Run: pip install ollama"
                )
        return self._client
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """トークンをストリーミング生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        options = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        
        stream = await self.client.chat(
            model=self._model,
            messages=messages,
            options=options if options else None,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]


class StreamingOpenAIClient:
    """OpenAI ストリーミングLLMクライアント"""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._client = None
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """トークンをストリーミング生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {"model": self._model, "messages": messages, "stream": True}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        stream = await self.client.chat.completions.create(**kwargs)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class StreamingAzureOpenAIClient:
    """Azure OpenAI ストリーミングLLMクライアント"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
    ):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self._api_key = api_key
        self._client = None
    
    @property
    def model_name(self) -> str:
        return self.deployment_name
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self._api_key,
                    api_version=self.api_version,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """トークンをストリーミング生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.deployment_name,
            "messages": messages,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        stream = await self.client.chat.completions.create(**kwargs)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockStreamingLLMClient:
    """モックストリーミングLLMクライアント（テスト用）"""
    
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        delay_ms: int = 50,
    ):
        self._responses = responses or {}
        self._default_response = "This is a mock streaming response. [1] It supports citations."
        self._delay_ms = delay_ms
    
    @property
    def model_name(self) -> str:
        return "mock-streaming-llm"
    
    def set_response(self, pattern: str, response: str) -> None:
        """応答パターンを設定"""
        self._responses[pattern] = response
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """モックストリーミング生成"""
        response = self._default_response
        
        for pattern, resp in self._responses.items():
            if pattern in prompt:
                response = resp
                break
        
        # 単語ごとにストリーミング
        words = response.split()
        for i, word in enumerate(words):
            if self._delay_ms > 0:
                await asyncio.sleep(self._delay_ms / 1000)
            
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word


def create_streaming_synthesizer(
    llm_client: StreamingLLMProtocol,
    config: Optional[StreamConfig] = None,
    system_prompt: str | None = None,
) -> StreamingAnswerSynthesizer:
    """ストリーミング回答合成器を作成
    
    Args:
        llm_client: ストリーミングLLMクライアント
        config: ストリーミング設定
        system_prompt: システムプロンプト
        
    Returns:
        StreamingAnswerSynthesizer
    """
    return StreamingAnswerSynthesizer(
        llm_client=llm_client,
        config=config,
        system_prompt=system_prompt,
    )

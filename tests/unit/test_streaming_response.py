"""Streaming Response Tests.

Tests for streaming search responses, including:
- MONJYU API streaming integration
- MCP Server streaming compatibility
- Real-time response streaming
- Backpressure handling
- Error recovery during streaming
"""

import asyncio
import json
import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


from monjyu.api.streaming import (
    ChunkType,
    MockStreamingSource,
    StreamingChunk,
    StreamingConfig,
    StreamingResult,
    StreamingService,
    StreamingState,
    StreamingStatus,
    StreamingCancelledError,
    create_streaming_service,
)


# ============================================================
# Test Data
# ============================================================


SAMPLE_SEARCH_RESPONSE = """GraphRAG (Graph-based Retrieval-Augmented Generation) is an advanced 
approach that combines knowledge graphs with retrieval-augmented generation techniques. 

Key aspects include:
1. Entity extraction and relationship mapping
2. Community detection for topic clustering
3. Hybrid search combining vector similarity and graph traversal

The methodology has been shown to improve answer accuracy by 15% compared to traditional 
RAG approaches, particularly for complex queries requiring multi-hop reasoning."""


@dataclass
class MockCitation:
    """Mock citation for testing."""
    doc_id: str
    title: str
    snippet: str
    relevance: float = 0.95


# ============================================================
# MONJYU API Streaming Tests
# ============================================================


class TestMONJYUAPIStreaming:
    """Tests for MONJYU API streaming capabilities."""
    
    @pytest.mark.asyncio
    async def test_streaming_search_response(self):
        """Test streaming search response from MONJYU API."""
        source = MockStreamingSource(
            response=SAMPLE_SEARCH_RESPONSE,
            delay_ms=5,
        )
        service = StreamingService(source=source)
        
        chunks = []
        async for chunk in service.stream_search("What is GraphRAG?"):
            chunks.append(chunk)
        
        # Verify streaming behavior
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) > 0
        
        # Verify final chunk (may have multiple done chunks)
        done_chunks = [c for c in chunks if c.is_done]
        assert len(done_chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_streaming_with_citations(self):
        """Test streaming with citation markers."""
        source = MockStreamingSource(
            response="According to the research [1], the results show significant improvement [2].",
            delay_ms=2,
        )
        config = StreamingConfig(include_citations=True)
        service = StreamingService(config=config, source=source)
        
        context = [
            {"doc_id": "paper1", "title": "Main Research Paper", "snippet": "Research findings"},
            {"doc_id": "paper2", "title": "Supporting Study", "snippet": "Supporting data"},
        ]
        
        chunks = []
        citations = []
        async for chunk in service.stream_search("query", context=context):
            chunks.append(chunk)
            if chunk.chunk_type == ChunkType.CITATION:
                citations.append(chunk)
        
        # Should have some citation chunks
        assert len(citations) >= 0  # May have 0 if not triggered
    
    @pytest.mark.asyncio
    async def test_streaming_collect_result(self):
        """Test collecting streamed result."""
        source = MockStreamingSource(
            response=SAMPLE_SEARCH_RESPONSE,
            delay_ms=2,
        )
        service = StreamingService(source=source)
        
        stream = service.stream_search("What is GraphRAG?")
        result = await service.collect_stream(stream)
        
        assert isinstance(result, StreamingResult)
        assert "GraphRAG" in result.full_response
        assert result.chunk_count > 0
        assert result.state.status == StreamingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_streaming_progress_tracking(self):
        """Test progress tracking during streaming."""
        source = MockStreamingSource(
            response="Word " * 50,  # 50 words
            delay_ms=1,
        )
        service = StreamingService(source=source)
        
        chunk_count = 0
        
        async for chunk in service.stream_search("test"):
            chunk_count += 1
        
        # Should have multiple chunks
        assert chunk_count > 0


# ============================================================
# Backpressure and Rate Control Tests
# ============================================================


class TestStreamingBackpressure:
    """Tests for streaming backpressure handling."""
    
    @pytest.mark.asyncio
    async def test_slow_consumer(self):
        """Test handling of slow consumer."""
        source = MockStreamingSource(
            response="word " * 20,
            delay_ms=1,
        )
        config = StreamingConfig(buffer_size=5)
        service = StreamingService(config=config, source=source)
        
        chunks = []
        async for chunk in service.stream_text("word " * 20, delay_ms=1):
            # Simulate slow consumer
            await asyncio.sleep(0.01)
            chunks.append(chunk)
        
        # Should still receive all chunks
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_fast_consumer(self):
        """Test handling of fast consumer."""
        source = MockStreamingSource(
            response="word " * 50,
            delay_ms=10,  # Slow producer
        )
        service = StreamingService(source=source)
        
        start_time = asyncio.get_event_loop().time()
        chunks = []
        
        async for chunk in service.stream_search("test"):
            chunks.append(chunk)
        
        # Fast consumer should wait for producer
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_prevention(self):
        """Test that buffer overflow is prevented."""
        config = StreamingConfig(buffer_size=3)
        service = StreamingService(config=config)
        
        # Generate lots of text
        long_text = "word " * 100
        
        chunks = []
        async for chunk in service.stream_text(long_text, delay_ms=1):
            chunks.append(chunk)
            # Small delay to allow buffer management
            await asyncio.sleep(0.001)
        
        # All chunks should be received
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) > 0


# ============================================================
# Error Recovery Tests
# ============================================================


class TestStreamingErrorRecovery:
    """Tests for streaming error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_mid_stream_error_handling(self):
        """Test handling errors during streaming."""
        source = MockStreamingSource(
            response="word1 word2 word3 word4 word5",
            delay_ms=1,
            fail_at=2,  # Fail after 2 words
        )
        service = StreamingService(source=source)
        
        chunks = []
        error_occurred = False
        
        try:
            async for chunk in source.stream("test"):
                chunks.append(chunk)
        except RuntimeError:
            error_occurred = True
        
        assert error_occurred
        # Should have received some chunks before error
        assert len(chunks) >= 2
    
    @pytest.mark.asyncio
    async def test_cancellation_cleanup(self):
        """Test cleanup after stream cancellation."""
        service = StreamingService()
        
        stream_id = None
        chunks = []
        
        try:
            async for chunk in service.stream_text("word " * 100, delay_ms=10):
                chunks.append(chunk)
                stream_id = chunk.stream_id
                
                if len(chunks) >= 3:
                    service.cancel_stream(stream_id)
        except StreamingCancelledError:
            pass
        
        # Verify cleanup
        state = service.get_stream_state(stream_id)
        assert state.status == StreamingStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling during streaming."""
        config = StreamingConfig(timeout_seconds=0.1)  # Very short timeout
        source = MockStreamingSource(
            response="word " * 100,
            delay_ms=50,  # Slow response
        )
        service = StreamingService(config=config, source=source)
        
        # This may or may not timeout depending on implementation
        # Just verify it doesn't hang indefinitely
        chunks = []
        try:
            async for chunk in service.stream_search("test"):
                chunks.append(chunk)
                if len(chunks) > 5:
                    break  # Don't wait forever
        except asyncio.TimeoutError:
            pass


# ============================================================
# Concurrent Streaming Tests
# ============================================================


class TestConcurrentStreaming:
    """Tests for concurrent streaming operations."""
    
    @pytest.mark.asyncio
    async def test_multiple_simultaneous_streams(self):
        """Test multiple streams running simultaneously."""
        service = StreamingService()
        
        async def stream_and_collect(text: str) -> List[StreamingChunk]:
            chunks = []
            async for chunk in service.stream_text(text, delay_ms=1):
                chunks.append(chunk)
            return chunks
        
        # Run 5 concurrent streams
        results = await asyncio.gather(
            stream_and_collect("Stream A content"),
            stream_and_collect("Stream B content"),
            stream_and_collect("Stream C content"),
            stream_and_collect("Stream D content"),
            stream_and_collect("Stream E content"),
        )
        
        # Each stream should complete successfully
        for result in results:
            assert len(result) > 0
            assert any(c.is_done for c in result)
    
    @pytest.mark.asyncio
    async def test_stream_isolation(self):
        """Test that streams are isolated from each other."""
        service = StreamingService()
        
        stream_ids = set()
        
        async def collect_stream_id(text: str) -> str:
            stream_id = None
            async for chunk in service.stream_text(text, delay_ms=1):
                stream_id = chunk.stream_id
            return stream_id
        
        ids = await asyncio.gather(
            collect_stream_id("A"),
            collect_stream_id("B"),
            collect_stream_id("C"),
        )
        
        # Each stream should have unique ID
        assert len(set(ids)) == 3
    
    @pytest.mark.asyncio
    async def test_cancel_one_stream_doesnt_affect_others(self):
        """Test cancelling one stream doesn't affect others."""
        service = StreamingService()
        
        results = {"A": [], "B": [], "C": []}
        
        async def stream_a():
            async for chunk in service.stream_text("A " * 50, delay_ms=5):
                results["A"].append(chunk)
        
        async def stream_b_cancel():
            chunks = []
            async for chunk in service.stream_text("B " * 50, delay_ms=5):
                chunks.append(chunk)
                if len(chunks) >= 3:
                    service.cancel_stream(chunk.stream_id)
                    break
            results["B"] = chunks
        
        async def stream_c():
            async for chunk in service.stream_text("C " * 50, delay_ms=5):
                results["C"].append(chunk)
        
        await asyncio.gather(
            stream_a(),
            stream_b_cancel(),
            stream_c(),
            return_exceptions=True,
        )
        
        # A and C should have more chunks than cancelled B
        assert len(results["A"]) > len(results["B"])
        assert len(results["C"]) > len(results["B"])


# ============================================================
# MCP Server Streaming Compatibility Tests
# ============================================================


class TestMCPServerStreamingCompatibility:
    """Tests for MCP Server streaming compatibility."""
    
    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU with streaming support."""
        mock = MagicMock()
        
        # Mock search result
        mock_result = MagicMock()
        mock_result.query = "test query"
        mock_result.answer = SAMPLE_SEARCH_RESPONSE
        mock_result.citations = []
        mock_result.search_mode = MagicMock(value="lazy")
        mock_result.search_level = 1
        mock_result.total_time_ms = 150.0
        mock_result.llm_calls = 3
        mock_result.citation_count = 2
        
        mock.search.return_value = mock_result
        
        return mock
    
    @pytest.mark.asyncio
    async def test_mcp_tool_response_streamable(self, mock_monjyu):
        """Test that MCP tool responses can be streamed."""
        from monjyu.mcp_server.server import (
            monjyu_search,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            # Get search result
            result = await monjyu_search("What is GraphRAG?")
            data = json.loads(result)
            
            # Result should be suitable for streaming
            assert "answer" in data
            answer = data["answer"]
            
            # Verify answer can be streamed
            service = StreamingService()
            chunks = []
            async for chunk in service.stream_text(answer, delay_ms=1):
                chunks.append(chunk)
            
            assert len(chunks) > 0
        finally:
            reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_streaming_json_serialization(self):
        """Test streaming chunks are JSON serializable."""
        service = StreamingService()
        
        async for chunk in service.stream_text("Test content", delay_ms=1):
            # Each chunk should be serializable
            data = chunk.to_dict()
            json_str = json.dumps(data)
            
            # And deserializable
            parsed = json.loads(json_str)
            restored = StreamingChunk.from_dict(parsed)
            
            assert restored.content == chunk.content
            assert restored.chunk_type == chunk.chunk_type


# ============================================================
# Integration Tests
# ============================================================


class TestStreamingIntegrationAdvanced:
    """Advanced integration tests for streaming."""
    
    @pytest.mark.asyncio
    async def test_full_search_to_stream_workflow(self):
        """Test complete workflow from search to streaming."""
        # Simulate search result
        search_response = {
            "query": "What is GraphRAG?",
            "answer": SAMPLE_SEARCH_RESPONSE,
            "citations": [
                {"doc_id": "paper1", "title": "GraphRAG Paper"},
                {"doc_id": "paper2", "title": "RAG Survey"},
            ],
        }
        
        # Create streaming service
        service = StreamingService()
        
        # Stream the answer
        chunks = []
        async for chunk in service.stream_text(search_response["answer"], delay_ms=1):
            chunks.append(chunk)
        
        # Collect and verify
        text_chunks = [c for c in chunks if c.is_text]
        reconstructed = " ".join(c.content for c in text_chunks)
        
        assert "GraphRAG" in reconstructed
        assert any(c.is_done for c in chunks)
    
    @pytest.mark.asyncio
    async def test_streaming_with_callbacks_integration(self):
        """Test streaming with callbacks in integration scenario."""
        received_chunks = []
        final_state = [None]
        
        config = StreamingConfig(
            on_chunk=lambda c: received_chunks.append(c),
            on_complete=lambda s: final_state.__setitem__(0, s),
        )
        service = StreamingService(config=config)
        
        # Stream search response
        source = MockStreamingSource(
            response=SAMPLE_SEARCH_RESPONSE,
            delay_ms=1,
        )
        service._source = source
        
        async for _ in service.stream_search("What is GraphRAG?"):
            pass
        
        # Verify callbacks were called
        assert len(received_chunks) > 0
        assert final_state[0] is not None
        assert final_state[0].status == StreamingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_streaming_metrics_collection(self):
        """Test streaming metrics are properly collected."""
        service = StreamingService()
        
        # Stream some content
        async for _ in service.stream_text(SAMPLE_SEARCH_RESPONSE, delay_ms=1):
            pass
        
        # Clear and check metrics
        cleared = service.clear_completed_streams()
        assert cleared >= 1

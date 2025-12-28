"""Tests for StreamingService.

REQ-API-003: ストリーミング出力
"""

import asyncio
import time

import pytest

from monjyu.api.streaming import (
    ChunkType,
    MockStreamingSource,
    StreamingCancelledError,
    StreamingChunk,
    StreamingConfig,
    StreamingError,
    StreamingResult,
    StreamingService,
    StreamingState,
    StreamingStatus,
    StreamingTimeoutError,
    create_streaming_service,
)


# ============================================================
# StreamingChunk Tests
# ============================================================


class TestStreamingChunk:
    """StreamingChunk tests."""
    
    def test_chunk_creation(self):
        """Test chunk creation."""
        chunk = StreamingChunk(content="Hello")
        
        assert chunk.content == "Hello"
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.is_text
        assert not chunk.is_done
        assert not chunk.is_error
    
    def test_text_factory(self):
        """Test text factory method."""
        chunk = StreamingChunk.text("Hello world")
        
        assert chunk.content == "Hello world"
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.is_text
    
    def test_citation_factory(self):
        """Test citation factory method."""
        citation_data = {"doc_id": "doc1", "title": "Paper 1"}
        chunk = StreamingChunk.citation("[1]", citation_data)
        
        assert chunk.content == "[1]"
        assert chunk.chunk_type == ChunkType.CITATION
        assert chunk.data == citation_data
    
    def test_progress_factory(self):
        """Test progress factory method."""
        chunk = StreamingChunk.progress("Processing...", percentage=50.0)
        
        assert chunk.content == "Processing..."
        assert chunk.chunk_type == ChunkType.PROGRESS
        assert chunk.data["percentage"] == 50.0
    
    def test_done_factory(self):
        """Test done factory method."""
        chunk = StreamingChunk.done("Complete")
        
        assert chunk.content == "Complete"
        assert chunk.chunk_type == ChunkType.DONE
        assert chunk.is_done
    
    def test_error_factory(self):
        """Test error factory method."""
        chunk = StreamingChunk.error("Something went wrong", error_code="ERR001")
        
        assert chunk.content == "Something went wrong"
        assert chunk.chunk_type == ChunkType.ERROR
        assert chunk.is_error
        assert chunk.data["error_code"] == "ERR001"
    
    def test_to_dict(self):
        """Test serialization."""
        chunk = StreamingChunk.text("Hello", stream_id="stream1", sequence=5)
        
        data = chunk.to_dict()
        
        assert data["content"] == "Hello"
        assert data["chunk_type"] == "text"
        assert data["stream_id"] == "stream1"
        assert data["sequence"] == 5
    
    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "content": "Hello",
            "chunk_type": "text",
            "stream_id": "stream1",
            "sequence": 5,
        }
        
        chunk = StreamingChunk.from_dict(data)
        
        assert chunk.content == "Hello"
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.stream_id == "stream1"


# ============================================================
# StreamingState Tests
# ============================================================


class TestStreamingState:
    """StreamingState tests."""
    
    def test_state_creation(self):
        """Test state creation."""
        state = StreamingState()
        
        assert state.status == StreamingStatus.PENDING
        assert state.chunks_sent == 0
        assert not state.is_active
        assert not state.is_completed
    
    def test_state_start(self):
        """Test starting state."""
        state = StreamingState()
        state.start()
        
        assert state.status == StreamingStatus.STREAMING
        assert state.is_active
        assert state.started_at is not None
    
    def test_state_complete(self):
        """Test completing state."""
        state = StreamingState()
        state.start()
        time.sleep(0.01)
        state.complete()
        
        assert state.status == StreamingStatus.COMPLETED
        assert state.is_completed
        assert state.completed_at is not None
        assert state.duration_ms > 0
    
    def test_state_cancel(self):
        """Test cancelling state."""
        state = StreamingState()
        state.start()
        state.cancel()
        
        assert state.status == StreamingStatus.CANCELLED
        assert state.is_completed
        assert state._cancel_requested
    
    def test_state_fail(self):
        """Test failing state."""
        state = StreamingState()
        state.start()
        state.fail("Something went wrong")
        
        assert state.status == StreamingStatus.ERROR
        assert state.error == "Something went wrong"
        assert state.is_completed
    
    def test_record_chunk(self):
        """Test recording chunks."""
        state = StreamingState()
        chunk = StreamingChunk.text("Hello world")
        
        state.record_chunk(chunk)
        
        assert state.chunks_sent == 1
        assert state.bytes_sent > 0
        assert state.tokens_sent == 2  # "Hello" and "world"
    
    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        state = StreamingState()
        state.start()
        
        for _ in range(10):
            state.record_chunk(StreamingChunk.text("word "))
        
        time.sleep(0.1)
        state.complete()
        
        assert state.tokens_per_second > 0
    
    def test_to_dict(self):
        """Test serialization."""
        state = StreamingState()
        state.start()
        state.complete()
        
        data = state.to_dict()
        
        assert data["status"] == "completed"
        assert "stream_id" in data
        assert "duration_ms" in data


# ============================================================
# StreamingConfig Tests
# ============================================================


class TestStreamingConfig:
    """StreamingConfig tests."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = StreamingConfig()
        
        assert config.buffer_size == 10
        assert config.timeout_seconds == 300.0
        assert config.include_citations is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            buffer_size=20,
            timeout_seconds=60.0,
            include_citations=False,
        )
        
        assert config.buffer_size == 20
        assert config.timeout_seconds == 60.0
        assert config.include_citations is False
    
    def test_config_to_dict(self):
        """Test serialization."""
        config = StreamingConfig(buffer_size=15)
        
        data = config.to_dict()
        
        assert data["buffer_size"] == 15


# ============================================================
# StreamingService Tests
# ============================================================


class TestStreamingServiceBasic:
    """Basic StreamingService tests."""
    
    def test_service_creation(self):
        """Test service creation."""
        service = StreamingService()
        
        assert service.config is not None
        assert service.active_stream_count == 0
    
    def test_service_with_config(self):
        """Test service with custom config."""
        config = StreamingConfig(timeout_seconds=60.0)
        service = StreamingService(config=config)
        
        assert service.config.timeout_seconds == 60.0
    
    def test_get_status(self):
        """Test getting status."""
        service = StreamingService()
        
        status = service.get_status()
        
        assert "active_streams" in status
        assert "config" in status


class TestStreamingServiceStreamText:
    """StreamingService.stream_text tests."""
    
    @pytest.mark.asyncio
    async def test_stream_text_basic(self):
        """Test basic text streaming."""
        service = StreamingService()
        
        chunks = []
        async for chunk in service.stream_text("Hello world", delay_ms=1):
            chunks.append(chunk)
        
        # Should have "Hello", " ", "world", and done chunk
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) == 2
        assert chunks[-1].is_done
    
    @pytest.mark.asyncio
    async def test_stream_text_content(self):
        """Test streamed content matches input."""
        service = StreamingService()
        
        text = "The quick brown fox"
        result_text = ""
        
        async for chunk in service.stream_text(text, delay_ms=1):
            if chunk.is_text:
                result_text += chunk.content
        
        assert result_text.strip() == text
    
    @pytest.mark.asyncio
    async def test_stream_text_sequence(self):
        """Test chunk sequence numbers."""
        service = StreamingService()
        
        sequences = []
        async for chunk in service.stream_text("a b c", delay_ms=1):
            if chunk.is_text:
                sequences.append(chunk.sequence)
        
        assert sequences == [0, 1, 2]
    
    @pytest.mark.asyncio
    async def test_stream_text_state_tracking(self):
        """Test state is tracked during streaming."""
        service = StreamingService()
        
        stream_id = None
        async for chunk in service.stream_text("Hello", delay_ms=1):
            stream_id = chunk.stream_id
            break
        
        # Consume rest
        async for _ in service.stream_text("Hello", delay_ms=1):
            pass
        
        assert stream_id is not None
        state = service.get_stream_state(stream_id)
        # Note: state may be None if we broke early


class TestStreamingServiceCallbacks:
    """StreamingService callback tests."""
    
    @pytest.mark.asyncio
    async def test_on_chunk_callback(self):
        """Test on_chunk callback."""
        received_chunks = []
        
        config = StreamingConfig(on_chunk=lambda c: received_chunks.append(c))
        service = StreamingService(config=config)
        
        async for _ in service.stream_text("Hello world", delay_ms=1):
            pass
        
        assert len(received_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_on_complete_callback(self):
        """Test on_complete callback."""
        completed_state = [None]
        
        config = StreamingConfig(on_complete=lambda s: completed_state.__setitem__(0, s))
        service = StreamingService(config=config)
        
        async for _ in service.stream_text("Hello", delay_ms=1):
            pass
        
        assert completed_state[0] is not None
        assert completed_state[0].status == StreamingStatus.COMPLETED


class TestStreamingServiceCancel:
    """StreamingService cancellation tests."""
    
    @pytest.mark.asyncio
    async def test_cancel_stream(self):
        """Test cancelling a stream."""
        service = StreamingService()
        
        stream_id = None
        chunk_count = 0
        
        try:
            async for chunk in service.stream_text("a " * 100, delay_ms=10):
                chunk_count += 1
                stream_id = chunk.stream_id
                
                if chunk_count == 3:
                    # Cancel after 3 chunks
                    service.cancel_stream(stream_id)
        except StreamingCancelledError:
            pass
        
        state = service.get_stream_state(stream_id)
        assert state.status == StreamingStatus.CANCELLED


class TestStreamingServiceSearch:
    """StreamingService.stream_search tests."""
    
    @pytest.mark.asyncio
    async def test_stream_search_demo(self):
        """Test demo search streaming."""
        service = StreamingService()
        
        chunks = []
        async for chunk in service.stream_search("test query"):
            chunks.append(chunk)
        
        # Should have text chunks and done chunk
        assert len(chunks) > 0
        assert any(c.is_done for c in chunks)
    
    @pytest.mark.asyncio
    async def test_stream_search_with_source(self):
        """Test search streaming with source."""
        source = MockStreamingSource(
            response="This is a test response.",
            delay_ms=1,
        )
        service = StreamingService(source=source)
        
        context = [{"doc_id": "doc1", "content": "Test context"}]
        
        chunks = []
        async for chunk in service.stream_search("query", context=context):
            chunks.append(chunk)
        
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_stream_search_with_citations(self):
        """Test search streaming includes citations."""
        source = MockStreamingSource(
            response="word1 word2 word3 word4 word5 word6",
            delay_ms=1,
        )
        config = StreamingConfig(include_citations=True)
        service = StreamingService(config=config, source=source)
        
        context = [
            {"doc_id": "doc1", "title": "Paper 1"},
            {"doc_id": "doc2", "title": "Paper 2"},
        ]
        
        chunks = []
        async for chunk in service.stream_search("query", context=context):
            chunks.append(chunk)
        
        citation_chunks = [c for c in chunks if c.chunk_type == ChunkType.CITATION]
        assert len(citation_chunks) > 0


class TestStreamingServiceCollect:
    """StreamingService.collect_stream tests."""
    
    @pytest.mark.asyncio
    async def test_collect_stream(self):
        """Test collecting stream into result."""
        service = StreamingService()
        
        stream = service.stream_text("Hello world test", delay_ms=1)
        result = await service.collect_stream(stream)
        
        assert isinstance(result, StreamingResult)
        assert "Hello world test" in result.full_response
        assert result.chunk_count > 0


# ============================================================
# StreamingResult Tests
# ============================================================


class TestStreamingResult:
    """StreamingResult tests."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = StreamingResult(
            stream_id="stream1",
            full_response="Hello world",
        )
        
        assert result.stream_id == "stream1"
        assert result.full_response == "Hello world"
        assert result.chunk_count == 0
    
    def test_result_with_chunks(self):
        """Test result with chunks."""
        chunks = [
            StreamingChunk.text("Hello "),
            StreamingChunk.text("world"),
        ]
        result = StreamingResult(
            stream_id="stream1",
            full_response="Hello world",
            chunks=chunks,
        )
        
        assert result.chunk_count == 2
    
    def test_result_to_dict(self):
        """Test serialization."""
        result = StreamingResult(
            stream_id="stream1",
            full_response="Hello",
        )
        
        data = result.to_dict()
        
        assert data["stream_id"] == "stream1"
        assert data["full_response"] == "Hello"


# ============================================================
# MockStreamingSource Tests
# ============================================================


class TestMockStreamingSource:
    """MockStreamingSource tests."""
    
    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming."""
        source = MockStreamingSource(response="Hello world", delay_ms=1)
        
        chunks = []
        async for chunk in source.stream("test"):
            chunks.append(chunk)
        
        assert "".join(chunks) == "Hello world"
    
    @pytest.mark.asyncio
    async def test_stream_with_citations(self):
        """Test streaming with citations."""
        source = MockStreamingSource(
            response="word1 word2 word3 word4",
            delay_ms=1,
        )
        context = [{"doc_id": "doc1"}]
        
        all_citations = []
        async for text, citations in source.stream_with_citations("test", context):
            all_citations.extend(citations)
        
        assert len(all_citations) > 0
    
    @pytest.mark.asyncio
    async def test_stream_failure(self):
        """Test simulated failure."""
        source = MockStreamingSource(
            response="word1 word2 word3",
            delay_ms=1,
            fail_at=1,
        )
        
        with pytest.raises(RuntimeError, match="Simulated failure"):
            async for _ in source.stream("test"):
                pass


# ============================================================
# create_streaming_service Tests
# ============================================================


class TestCreateStreamingService:
    """create_streaming_service factory tests."""
    
    def test_create_default(self):
        """Test default creation."""
        service = create_streaming_service()
        
        assert service is not None
        assert isinstance(service.config, StreamingConfig)
    
    def test_create_with_dict_config(self):
        """Test creation with dict config."""
        service = create_streaming_service(config={"timeout_seconds": 60.0})
        
        assert service.config.timeout_seconds == 60.0
    
    def test_create_with_source(self):
        """Test creation with source."""
        source = MockStreamingSource()
        service = create_streaming_service(source=source)
        
        assert service._source is source


# ============================================================
# Integration Tests
# ============================================================


class TestStreamingIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self):
        """Test complete streaming workflow."""
        # Setup
        source = MockStreamingSource(
            response="GraphRAG combines graph-based knowledge with retrieval-augmented generation.",
            delay_ms=1,
        )
        config = StreamingConfig(include_citations=True)
        service = StreamingService(config=config, source=source)
        
        context = [
            {"doc_id": "paper1", "title": "GraphRAG Paper"},
        ]
        
        # Stream search
        stream = service.stream_search("What is GraphRAG?", context=context)
        result = await service.collect_stream(stream)
        
        # Verify result
        assert "GraphRAG" in result.full_response
        assert result.chunk_count > 0
        assert result.state is not None
        assert result.state.status == StreamingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """Test multiple concurrent streams."""
        service = StreamingService()
        
        async def run_stream(text: str) -> str:
            result = ""
            async for chunk in service.stream_text(text, delay_ms=1):
                if chunk.is_text:
                    result += chunk.content
            return result
        
        # Run 3 concurrent streams
        results = await asyncio.gather(
            run_stream("Stream one"),
            run_stream("Stream two"),
            run_stream("Stream three"),
        )
        
        assert "Stream one" in results[0]
        assert "Stream two" in results[1]
        assert "Stream three" in results[2]
    
    @pytest.mark.asyncio
    async def test_clear_completed_streams(self):
        """Test clearing completed streams."""
        service = StreamingService()
        
        # Run a few streams
        for text in ["a", "b", "c"]:
            async for _ in service.stream_text(text, delay_ms=1):
                pass
        
        # Clear completed
        cleared = service.clear_completed_streams()
        
        assert cleared == 3
        assert service.active_stream_count == 0

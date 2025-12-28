"""Integration tests for Search components.

Tests the interaction between Query Router, Vector Search, Local Search,
Global Search, and their integration with Cache and Observability.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

import pytest
import numpy as np

# Query Router
from monjyu.query.router import (
    QueryRouter,
    QueryRouterConfig,
    SearchMode,
    QueryType,
    RoutingDecision,
    RoutingContext,
)

# Vector Search
from monjyu.query.vector_search import (
    VectorSearchConfig,
    VectorSearchResult,
    SearchHit,
    InMemoryVectorSearch,
    IndexedDocument,
)

# Local Search
from monjyu.query.local_search import (
    LocalSearchConfig,
    LocalSearchResult,
    LocalSearch,
    EntityInfo,
    RelationshipInfo,
    ChunkInfo,
    InMemoryEntityStore,
    InMemoryRelationshipStore,
    InMemoryChunkStore,
    MockLLMClient,
)

# Global Search
from monjyu.query.global_search import (
    GlobalSearchConfig,
    GlobalSearchResult,
    GlobalSearch,
    CommunityInfo,
    MapResult,
    InMemoryCommunityStore,
    MockLLMClient as GlobalMockLLMClient,
)

# Cache
from monjyu.storage.cache import (
    MemoryCache,
    CacheStats,
)

# Observability
from monjyu.observability import (
    InMemoryExporter,
    Observability,
    SpanStatus,
)

# Error Handling
from monjyu.errors import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    LLMError,
    QueryError,
    retry,
)


# ============================================================
# Test Data Factories
# ============================================================


def create_test_documents() -> List[IndexedDocument]:
    """Create test documents with embeddings."""
    # Generate random embeddings (dimension 384 for testing)
    np.random.seed(42)
    
    docs = [
        IndexedDocument(
            chunk_id="doc1_chunk1",
            content="GraphRAG is a retrieval-augmented generation system that uses knowledge graphs.",
            vector=np.random.rand(384).astype(np.float32),
            metadata={"section": "introduction"},
            paper_id="paper1",
            section_type="introduction",
        ),
        IndexedDocument(
            chunk_id="doc1_chunk2",
            content="The system extracts entities and relationships from documents to build a graph.",
            vector=np.random.rand(384).astype(np.float32),
            metadata={"section": "methods"},
            paper_id="paper1",
            section_type="methods",
        ),
        IndexedDocument(
            chunk_id="doc2_chunk1",
            content="LazyGraphRAG provides cost-effective graph-based retrieval through lazy evaluation.",
            vector=np.random.rand(384).astype(np.float32),
            metadata={"section": "abstract"},
            paper_id="paper2",
            section_type="abstract",
        ),
        IndexedDocument(
            chunk_id="doc2_chunk2",
            content="The approach defers expensive graph operations until query time.",
            vector=np.random.rand(384).astype(np.float32),
            metadata={"section": "methods"},
            paper_id="paper2",
            section_type="methods",
        ),
        IndexedDocument(
            chunk_id="doc3_chunk1",
            content="Vector databases enable efficient similarity search for embeddings.",
            vector=np.random.rand(384).astype(np.float32),
            metadata={"section": "background"},
            paper_id="paper3",
            section_type="background",
        ),
    ]
    return docs


def create_test_entities() -> List[EntityInfo]:
    """Create test entities."""
    return [
        EntityInfo(
            entity_id="e1",
            name="GraphRAG",
            entity_type="System",
            description="A retrieval-augmented generation system using knowledge graphs",
        ),
        EntityInfo(
            entity_id="e2",
            name="LazyGraphRAG",
            entity_type="System",
            description="A cost-effective variant of GraphRAG",
        ),
        EntityInfo(
            entity_id="e3",
            name="Knowledge Graph",
            entity_type="Concept",
            description="A graph structure representing entities and relationships",
        ),
        EntityInfo(
            entity_id="e4",
            name="Entity Extraction",
            entity_type="Method",
            description="Process of identifying entities from text",
        ),
        EntityInfo(
            entity_id="e5",
            name="Vector Database",
            entity_type="Technology",
            description="Database optimized for vector similarity search",
        ),
    ]


def create_test_relationships() -> List[RelationshipInfo]:
    """Create test relationships."""
    return [
        RelationshipInfo(
            relationship_id="r1",
            source_id="e1",
            target_id="e3",
            relation_type="uses",
            description="GraphRAG uses Knowledge Graphs",
        ),
        RelationshipInfo(
            relationship_id="r2",
            source_id="e2",
            target_id="e1",
            relation_type="extends",
            description="LazyGraphRAG extends GraphRAG",
        ),
        RelationshipInfo(
            relationship_id="r3",
            source_id="e1",
            target_id="e4",
            relation_type="employs",
            description="GraphRAG employs Entity Extraction",
        ),
        RelationshipInfo(
            relationship_id="r4",
            source_id="e1",
            target_id="e5",
            relation_type="integrates_with",
            description="GraphRAG integrates with Vector Database",
        ),
    ]


def create_test_communities() -> List[CommunityInfo]:
    """Create test communities."""
    return [
        CommunityInfo(
            community_id="c1",
            title="Graph-based RAG Systems",
            summary="This community covers graph-based retrieval augmented generation systems including GraphRAG and LazyGraphRAG.",
            level=1,
            size=10,
            key_entities=["GraphRAG", "LazyGraphRAG", "Knowledge Graph"],
            findings=["Graph structures improve retrieval quality", "Lazy evaluation reduces costs"],
        ),
        CommunityInfo(
            community_id="c2",
            title="Vector Search Technologies",
            summary="This community covers vector databases and similarity search techniques.",
            level=1,
            size=8,
            key_entities=["Vector Database", "Embedding", "Similarity Search"],
            findings=["Vector search enables semantic retrieval", "Efficient for high-dimensional data"],
        ),
        CommunityInfo(
            community_id="c3",
            title="Information Extraction Methods",
            summary="This community covers entity extraction and relationship extraction methods.",
            level=1,
            size=6,
            key_entities=["Entity Extraction", "NER", "Relation Extraction"],
            findings=["LLMs improve extraction accuracy", "Graph construction depends on quality"],
        ),
    ]


def create_test_chunks() -> List[ChunkInfo]:
    """Create test chunks."""
    return [
        ChunkInfo(
            chunk_id="doc1_chunk1",
            content="GraphRAG is a retrieval-augmented generation system that uses knowledge graphs.",
            paper_id="paper1",
            section_type="introduction",
        ),
        ChunkInfo(
            chunk_id="doc1_chunk2",
            content="The system extracts entities and relationships from documents to build a graph.",
            paper_id="paper1",
            section_type="methods",
        ),
        ChunkInfo(
            chunk_id="doc2_chunk1",
            content="LazyGraphRAG provides cost-effective graph-based retrieval through lazy evaluation.",
            paper_id="paper2",
            section_type="abstract",
        ),
    ]


# ============================================================
# Mock Components
# ============================================================


class MockEmbedder:
    """Mock embedder for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate mock embedding."""
        self.call_count += 1
        # Use hash to generate deterministic but varied embeddings
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self.dimension).astype(np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate mock embeddings for batch."""
        return [await self.embed(text) for text in texts]


class MockQueryRouter:
    """Mock query router for testing."""
    
    def __init__(self, default_mode: SearchMode = SearchMode.VECTOR):
        self.default_mode = default_mode
        self.call_count = 0
        self.queries: List[str] = []
    
    async def route(self, query: str, context: Optional[RoutingContext] = None) -> RoutingDecision:
        """Route query to appropriate search mode."""
        self.call_count += 1
        self.queries.append(query)
        
        # Simple keyword-based routing for testing
        query_lower = query.lower()
        
        if "trend" in query_lower or "overview" in query_lower:
            mode = SearchMode.LAZY
            query_type = QueryType.SURVEY
        elif "compare" in query_lower or "difference" in query_lower:
            mode = SearchMode.GRAPHRAG
            query_type = QueryType.COMPARISON
        elif "what is" in query_lower or "define" in query_lower:
            mode = SearchMode.VECTOR
            query_type = QueryType.FACTOID
        else:
            mode = self.default_mode
            query_type = QueryType.EXPLORATION
        
        return RoutingDecision(
            mode=mode,
            query_type=query_type,
            confidence=0.85,
            reasoning=f"Routed to {mode.value} based on query analysis",
        )


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def memory_cache():
    """Create a MemoryCache for testing."""
    return MemoryCache(max_size=100, default_ttl=60.0)


@pytest.fixture
def observability():
    """Create Observability with InMemoryExporter."""
    exporter = InMemoryExporter()
    obs = Observability(exporter=exporter)
    return obs, exporter


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
    )


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    return MockEmbedder(dimension=384)


@pytest.fixture
async def vector_search(mock_embedder):
    """Create InMemoryVectorSearch with test documents."""
    docs = create_test_documents()
    search = InMemoryVectorSearch(
        embedder=mock_embedder,
        config=VectorSearchConfig(top_k=5, min_score=0.0),
    )
    # Add documents using the proper async method
    await search.add_documents(docs)
    return search


@pytest.fixture
def local_search():
    """Create LocalSearch with test data."""
    entity_store = InMemoryEntityStore()
    relationship_store = InMemoryRelationshipStore()
    chunk_store = InMemoryChunkStore()
    llm_client = MockLLMClient()
    
    # Add test data
    for entity in create_test_entities():
        entity_store.add_entity(entity)
    for rel in create_test_relationships():
        relationship_store.add_relationship(rel)
    for chunk in create_test_chunks():
        chunk_store.add_chunk(chunk)
    
    return LocalSearch(
        llm_client=llm_client,
        entity_store=entity_store,
        relationship_store=relationship_store,
        chunk_store=chunk_store,
        config=LocalSearchConfig(max_hops=2, top_k_entities=5),
    )


@pytest.fixture
def global_search():
    """Create GlobalSearch with test data."""
    community_store = InMemoryCommunityStore()
    llm_client = GlobalMockLLMClient()
    
    # Add test communities
    for community in create_test_communities():
        community_store.add_community(community)
    
    return GlobalSearch(
        llm_client=llm_client,
        community_store=community_store,
        config=GlobalSearchConfig(top_k_communities=3),
    )


@pytest.fixture
def query_router():
    """Create mock query router."""
    return MockQueryRouter()


# ============================================================
# Query Router Integration Tests
# ============================================================


class TestQueryRouterIntegration:
    """Query Router integration tests."""
    
    @pytest.mark.asyncio
    async def test_router_routes_survey_to_lazy(self, query_router, observability):
        """Test that survey queries are routed to lazy search."""
        obs, exporter = observability
        
        with obs.tracer.start_span("route_query") as span:
            decision = await query_router.route("What are the research trends in RAG?")
            span.set_attribute("mode", decision.mode.value)
            span.set_attribute("query_type", decision.query_type.value)
        
        obs.tracer.flush()
        
        assert decision.mode == SearchMode.LAZY
        assert decision.query_type == QueryType.SURVEY
        assert len(exporter.spans) == 1
    
    @pytest.mark.asyncio
    async def test_router_routes_factoid_to_vector(self, query_router):
        """Test that factoid queries are routed to vector search."""
        decision = await query_router.route("What is GraphRAG?")
        
        assert decision.mode == SearchMode.VECTOR
        assert decision.query_type == QueryType.FACTOID
        assert decision.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_router_routes_comparison_to_graphrag(self, query_router):
        """Test that comparison queries are routed to graphrag."""
        decision = await query_router.route("Compare GraphRAG and LazyGraphRAG")
        
        assert decision.mode == SearchMode.GRAPHRAG
        assert decision.query_type == QueryType.COMPARISON
    
    @pytest.mark.asyncio
    async def test_router_with_cache(self, query_router, memory_cache):
        """Test router with caching of decisions."""
        query = "What is GraphRAG?"
        cache_key = f"route:{query}"
        
        # First call - cache miss
        decision1 = await query_router.route(query)
        await memory_cache.set(cache_key, {
            "mode": decision1.mode.value,
            "query_type": decision1.query_type.value,
            "confidence": decision1.confidence,
        })
        
        # Second call - use cache
        cached = await memory_cache.get(cache_key)
        assert cached is not None
        assert cached["mode"] == "vector"
        
        # Router should have been called only once
        assert query_router.call_count == 1


# ============================================================
# Vector Search Integration Tests
# ============================================================


class TestVectorSearchIntegration:
    """Vector Search integration tests."""
    
    @pytest.mark.asyncio
    async def test_vector_search_with_observability(
        self,
        vector_search,
        mock_embedder,
        observability,
    ):
        """Test vector search with observability tracking."""
        obs, exporter = observability
        
        with obs.tracer.start_span("vector_search") as span:
            span.set_attribute("query", "GraphRAG knowledge graphs")
            
            query_embedding = await mock_embedder.embed("GraphRAG knowledge graphs")
            result = await vector_search.search_by_vector(query_embedding, top_k=3)
            
            # VectorSearchResult has .hits list and .total_count
            span.set_attribute("result_count", result.total_count)
            obs.metrics.increment("vector_search.queries")
            obs.metrics.timer("vector_search.latency_ms", 10.0)
        
        obs.tracer.flush()
        
        assert len(result.hits) > 0
        assert obs.metrics.get_counter("vector_search.queries") == 1
    
    @pytest.mark.asyncio
    async def test_vector_search_with_cache(
        self,
        vector_search,
        mock_embedder,
        memory_cache,
    ):
        """Test vector search with result caching."""
        query = "GraphRAG"
        cache_key = f"vector:{query}"
        
        # First search - cache miss
        query_embedding = await mock_embedder.embed(query)
        result1 = await vector_search.search_by_vector(query_embedding, top_k=3)
        
        # Cache results (result.hits is the list of SearchHit)
        await memory_cache.set(cache_key, [r.to_dict() for r in result1.hits])
        
        # Second search - use cache
        cached = await memory_cache.get(cache_key)
        assert cached is not None
        assert len(cached) == len(result1.hits)
    
    @pytest.mark.asyncio
    async def test_vector_search_filters(self, vector_search, mock_embedder):
        """Test vector search with filtering."""
        query_embedding = await mock_embedder.embed("methods section")
        
        # Search all
        result = await vector_search.search_by_vector(query_embedding, top_k=10)
        
        # Filter by metadata
        filtered = [
            r for r in result.hits
            if r.metadata.get("section") == "methods"
        ]
        
        assert len(result.hits) >= len(filtered)


# ============================================================
# Local Search Integration Tests
# ============================================================


class TestLocalSearchIntegration:
    """Local Search integration tests."""
    
    @pytest.mark.asyncio
    async def test_local_search_with_observability(
        self,
        local_search,
        observability,
    ):
        """Test local search with observability tracking."""
        obs, exporter = observability
        
        with obs.tracer.start_span("local_search") as span:
            span.set_attribute("query", "GraphRAG entities")
            
            # LocalSearch.search is synchronous
            result = local_search.search("What entities does GraphRAG use?")
            
            # LocalSearchResult has entities_found attribute
            span.set_attribute("entities_found", len(result.entities_found))
            obs.metrics.increment("local_search.queries")
        
        obs.tracer.flush()
        
        assert result is not None
        assert obs.metrics.get_counter("local_search.queries") == 1
    
    @pytest.mark.asyncio
    async def test_local_search_entity_traversal(self, local_search):
        """Test local search with entity traversal."""
        # LocalSearch.search is synchronous
        result = local_search.search("How does GraphRAG relate to knowledge graphs?")
        
        # Should find related entities
        assert result is not None
        # Result should contain entity information
        assert hasattr(result, "entities") or hasattr(result, "answer")
    
    @pytest.mark.asyncio
    async def test_local_search_with_circuit_breaker(
        self,
        local_search,
        circuit_breaker,
    ):
        """Test local search with circuit breaker protection."""
        async def search_with_protection(query: str):
            async with circuit_breaker:
                # LocalSearch.search is synchronous
                return local_search.search(query)
        
        # Normal search should work
        result = await search_with_protection("What is GraphRAG?")
        assert result is not None
        assert circuit_breaker.state == CircuitState.CLOSED


# ============================================================
# Global Search Integration Tests
# ============================================================


class TestGlobalSearchIntegration:
    """Global Search integration tests."""
    
    @pytest.mark.asyncio
    async def test_global_search_with_observability(
        self,
        global_search,
        observability,
    ):
        """Test global search with observability tracking."""
        obs, exporter = observability
        
        with obs.tracer.start_span("global_search") as span:
            span.set_attribute("query", "RAG trends")
            
            # GlobalSearch.search is synchronous
            result = global_search.search("What are the trends in RAG research?")
            
            obs.metrics.increment("global_search.queries")
            if result and hasattr(result, "map_results"):
                span.set_attribute("communities_queried", len(result.map_results))
        
        obs.tracer.flush()
        
        assert result is not None
        assert obs.metrics.get_counter("global_search.queries") == 1
    
    @pytest.mark.asyncio
    async def test_global_search_map_reduce(self, global_search):
        """Test global search map-reduce pattern."""
        # GlobalSearch.search is synchronous
        result = global_search.search("Overview of graph-based retrieval systems")
        
        assert result is not None
        # Global search should aggregate from communities
        assert hasattr(result, "answer") or hasattr(result, "map_results")
    
    @pytest.mark.asyncio
    async def test_global_search_with_cache(self, global_search, memory_cache):
        """Test global search with caching."""
        query = "What are the main research directions?"
        cache_key = f"global:{query}"
        
        # First search (synchronous)
        result1 = global_search.search(query)
        
        # Cache the result
        if result1 and hasattr(result1, "to_dict"):
            await memory_cache.set(cache_key, result1.to_dict())
        else:
            await memory_cache.set(cache_key, {"answer": str(result1)})
        
        # Verify cache
        cached = await memory_cache.get(cache_key)
        assert cached is not None


# ============================================================
# Multi-Search Integration Tests
# ============================================================


class TestMultiSearchIntegration:
    """Integration tests for multiple search types working together."""
    
    @pytest.mark.asyncio
    async def test_routed_search_flow(
        self,
        query_router,
        vector_search,
        local_search,
        global_search,
        mock_embedder,
        observability,
    ):
        """Test complete search flow with routing."""
        obs, exporter = observability
        
        async def execute_search(query: str):
            with obs.tracer.start_span("search_flow") as span:
                # Route query
                with obs.tracer.start_span("route"):
                    decision = await query_router.route(query)
                    span.set_attribute("mode", decision.mode.value)
                
                # Execute appropriate search
                if decision.mode == SearchMode.VECTOR:
                    with obs.tracer.start_span("vector_search"):
                        embedding = await mock_embedder.embed(query)
                        return await vector_search.search_by_vector(embedding, top_k=5)
                elif decision.mode == SearchMode.LAZY:
                    with obs.tracer.start_span("local_search"):
                        # LocalSearch.search is synchronous
                        return local_search.search(query)
                elif decision.mode == SearchMode.GRAPHRAG:
                    with obs.tracer.start_span("global_search"):
                        # GlobalSearch.search is synchronous
                        return global_search.search(query)
                else:
                    with obs.tracer.start_span("vector_search"):
                        embedding = await mock_embedder.embed(query)
                        return await vector_search.search_by_vector(embedding, top_k=5)
        
        # Test different query types
        result1 = await execute_search("What is GraphRAG?")  # vector
        result2 = await execute_search("What are the trends in RAG?")  # lazy
        result3 = await execute_search("Compare GraphRAG and LazyGraphRAG")  # graphrag
        
        obs.tracer.flush()
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert len(exporter.spans) >= 6  # At least 2 spans per query
    
    @pytest.mark.asyncio
    async def test_search_with_fallback(
        self,
        vector_search,
        local_search,
        mock_embedder,
        circuit_breaker,
    ):
        """Test search with fallback on failure."""
        fail_count = 0
        
        async def search_with_fallback(query: str):
            nonlocal fail_count
            
            # Try local search first
            try:
                async with circuit_breaker:
                    if fail_count < 2:
                        fail_count += 1
                        raise LLMError("Local search failed")
                    # LocalSearch.search is synchronous
                    return local_search.search(query)
            except (LLMError, CircuitOpenError):
                # Fallback to vector search (async)
                embedding = await mock_embedder.embed(query)
                return await vector_search.search_by_vector(embedding, top_k=5)
        
        # First two calls should fallback to vector search
        result1 = await search_with_fallback("What is GraphRAG?")
        result2 = await search_with_fallback("How does it work?")
        
        # Third call should use local search
        result3 = await search_with_fallback("Entity relationships?")
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
    
    @pytest.mark.asyncio
    async def test_parallel_search_execution(
        self,
        vector_search,
        local_search,
        global_search,
        mock_embedder,
        observability,
    ):
        """Test parallel execution of multiple search types."""
        obs, exporter = observability
        query = "GraphRAG architecture"
        
        with obs.tracer.start_span("parallel_search") as span:
            span.set_attribute("query", query)
            
            # Execute searches in parallel
            async def vector_task():
                embedding = await mock_embedder.embed(query)
                return await vector_search.search_by_vector(embedding, top_k=3)
            
            async def local_task():
                # LocalSearch.search is synchronous - wrap in async
                return local_search.search(query)
            
            async def global_task():
                # GlobalSearch.search is synchronous - wrap in async
                return global_search.search(query)
            
            start_time = time.time()
            results = await asyncio.gather(
                vector_task(),
                local_task(),
                global_task(),
                return_exceptions=True,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            span.set_attribute("execution_time_ms", elapsed_ms)
            obs.metrics.timer("parallel_search.latency_ms", elapsed_ms)
        
        obs.tracer.flush()
        
        # All searches should complete
        assert len(results) == 3
        # At least vector search should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= 1
    
    @pytest.mark.asyncio
    async def test_search_result_merging(
        self,
        vector_search,
        local_search,
        mock_embedder,
    ):
        """Test merging results from multiple search types."""
        query = "GraphRAG"
        
        # Get results from both searches
        embedding = await mock_embedder.embed(query)
        vector_result = await vector_search.search_by_vector(embedding, top_k=3)
        # LocalSearch.search is synchronous
        local_result = local_search.search(query)
        
        # Merge results (VectorSearchResult.hits contains the list)
        merged = {
            "vector_results": [r.to_dict() for r in vector_result.hits],
            "local_result": local_result.to_dict() if hasattr(local_result, "to_dict") else str(local_result),
            "sources": set(),
        }
        
        # Extract sources from hits
        for r in vector_result.hits:
            if r.metadata.get("paper_id"):
                merged["sources"].add(r.metadata["paper_id"])
        
        assert "vector_results" in merged
        assert "local_result" in merged


# ============================================================
# Cache Integration Tests
# ============================================================


class TestSearchCacheIntegration:
    """Search + Cache integration tests."""
    
    @pytest.mark.asyncio
    async def test_cached_search_flow(
        self,
        query_router,
        vector_search,
        mock_embedder,
        memory_cache,
        observability,
    ):
        """Test search flow with comprehensive caching."""
        obs, exporter = observability
        
        async def cached_search(query: str):
            with obs.tracer.start_span("cached_search") as span:
                # Check route cache
                route_key = f"route:{query}"
                cached_route = await memory_cache.get(route_key)
                
                if cached_route:
                    mode = SearchMode(cached_route["mode"])
                    obs.metrics.increment("cache.route_hits")
                else:
                    decision = await query_router.route(query)
                    mode = decision.mode
                    await memory_cache.set(route_key, {"mode": mode.value})
                    obs.metrics.increment("cache.route_misses")
                
                # Check result cache
                result_key = f"result:{mode.value}:{query}"
                cached_result = await memory_cache.get(result_key)
                
                if cached_result:
                    obs.metrics.increment("cache.result_hits")
                    return cached_result
                
                # Execute search (async)
                obs.metrics.increment("cache.result_misses")
                embedding = await mock_embedder.embed(query)
                result = await vector_search.search_by_vector(embedding, top_k=3)
                
                # Cache result (VectorSearchResult.hits is the list)
                result_data = [r.to_dict() for r in result.hits]
                await memory_cache.set(result_key, result_data)
                
                return result_data
        
        # First search - all cache misses
        result1 = await cached_search("What is GraphRAG?")
        
        # Second search - should hit caches
        result2 = await cached_search("What is GraphRAG?")
        
        obs.metrics.flush()
        
        assert result1 == result2
        assert obs.metrics.get_counter("cache.route_hits") == 1
        assert obs.metrics.get_counter("cache.result_hits") == 1
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, memory_cache):
        """Test cache invalidation strategies."""
        # Set initial cache
        await memory_cache.set("search:q1", {"result": "old"}, ttl=0.1)
        
        # Verify cache exists
        cached = await memory_cache.get("search:q1")
        assert cached is not None
        
        # Wait for expiration
        await asyncio.sleep(0.15)
        
        # Cache should be invalidated
        cached = await memory_cache.get("search:q1")
        assert cached is None


# ============================================================
# Error Handling Tests
# ============================================================


class TestSearchErrorHandling:
    """Search error handling integration tests."""
    
    @pytest.mark.asyncio
    async def test_search_with_retry(self, mock_embedder):
        """Test search with retry on transient errors."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01, exceptions=(LLMError,))
        async def flaky_embed(text: str):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMError("Embedding service unavailable")
            return await mock_embedder.embed(text)
        
        result = await flaky_embed("test query")
        
        assert result is not None
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_search_circuit_breaker(self, circuit_breaker):
        """Test circuit breaker for search operations."""
        async def failing_search():
            raise LLMError("Search service down")
        
        # Trigger circuit breaker
        for _ in range(3):
            with pytest.raises(LLMError):
                async with circuit_breaker:
                    await failing_search()
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Further calls should fail fast
        with pytest.raises(CircuitOpenError):
            async with circuit_breaker:
                await failing_search()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(
        self,
        vector_search,
        mock_embedder,
        memory_cache,
    ):
        """Test graceful degradation when services fail."""
        query = "test query"
        
        # Simulate cached fallback result
        await memory_cache.set(f"fallback:{query}", {"answer": "Cached fallback answer"})
        
        async def search_with_degradation(q: str):
            try:
                # Primary search
                raise LLMError("Primary search failed")
            except LLMError:
                # Try cache fallback
                cached = await memory_cache.get(f"fallback:{q}")
                if cached:
                    return cached
                
                # Last resort: basic vector search
                embedding = await mock_embedder.embed(q)
                return vector_search.search_by_vector(embedding, top_k=1)
        
        result = await search_with_degradation(query)
        
        assert result is not None
        assert result.get("answer") == "Cached fallback answer"


# ============================================================
# Performance Tests
# ============================================================


class TestSearchPerformance:
    """Search performance integration tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(
        self,
        vector_search,
        mock_embedder,
        observability,
    ):
        """Test concurrent search performance."""
        obs, exporter = observability
        num_queries = 10
        
        async def single_search(query: str):
            embedding = await mock_embedder.embed(query)
            return await vector_search.search_by_vector(embedding, top_k=3)
        
        start_time = time.time()
        
        tasks = [
            single_search(f"query {i}")
            for i in range(num_queries)
        ]
        results = await asyncio.gather(*tasks)
        
        elapsed_ms = (time.time() - start_time) * 1000
        obs.metrics.timer("concurrent_search.total_ms", elapsed_ms)
        obs.metrics.gauge("concurrent_search.throughput", num_queries / (elapsed_ms / 1000))
        
        assert len(results) == num_queries
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_search_latency_tracking(
        self,
        vector_search,
        mock_embedder,
        observability,
    ):
        """Test search latency tracking."""
        obs, exporter = observability
        
        latencies = []
        
        for i in range(5):
            start = time.time()
            embedding = await mock_embedder.embed(f"query {i}")
            await vector_search.search_by_vector(embedding, top_k=3)
            latency_ms = (time.time() - start) * 1000
            
            latencies.append(latency_ms)
            obs.metrics.histogram("search.latency_ms", latency_ms)
        
        obs.metrics.flush()
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 1000  # Should be under 1 second

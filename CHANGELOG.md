# Changelog

All notable changes to MONJYU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2025-12-28

### Added

- **Azure AI Search VectorStore** - `monjyu.search.azure_vector_store`
  - AzureSearchVectorStoreConfig - Comprehensive configuration
  - AzureSearchVectorStore - Production vector storage
    - Vector search (HNSW algorithm)
    - Hybrid search (vector + keyword + semantic reranking)
    - Batch document upload/delete
    - Async/sync support
    - Managed Identity authentication
    - Retry logic with exponential backoff
  - AzureSearchEngine - Embedding integration wrapper
  - Factory functions: create_azure_vector_store, create_azure_search_engine
  - 30 unit tests (mocked Azure SDK)

### Changed

- Test count increased from 1461 to 1491
- Phase 3 "Azure AI Search統合" marked as complete

## [3.3.0] - 2025-12-28

### Added

- **Streaming Response** - `monjyu.search.streaming`
  - StreamConfig, StreamMetadata, StreamChunk types
  - StreamingAnswerSynthesizer - Real-time LLM token streaming
  - StreamingSearchEngine - Full pipeline streaming (search + synthesis)
  - MultiEngineStreamingSearch - Parallel multi-engine streaming
  - StreamingOllamaClient, StreamingOpenAIClient, StreamingAzureOpenAIClient
  - MockStreamingLLMClient for testing
  - StreamCallbacks for async/sync callbacks
  - 37 unit tests

### Changed

- Test count increased from 1420 to 1457
- Added "Streaming Response" to README features

## [3.2.0] - 2025-12-28

### Added

- **Hybrid GraphRAG (Query Layer)** - `monjyu.query.hybrid_search`
  - HybridSearch with 5 fusion methods (RRF, WEIGHTED, MAX, COMBSUM, COMBMNZ)
  - Support for VECTOR, LAZY, GLOBAL, LOCAL, KEYWORD search methods
  - Parallel and sequential execution modes
  - Graceful degradation when some engines fail
  - 36 unit tests

- **Hybrid GraphRAG (Search Layer)** - `monjyu.search.hybrid`
  - HybridSearchEngine with answer synthesis
  - ResultMerger for search result fusion
  - Integration with VectorSearchEngine and LazySearchEngine
  - 38 unit tests

- **External API Integration** - `monjyu.external`
  - SemanticScholarClient - Semantic Scholar API integration
  - CrossRefClient - CrossRef API integration
  - Rate limiting, caching, and error handling
  - 46 unit tests

- **MCP Server** - `monjyu.mcp_server`
  - FastMCP-based implementation
  - 7 tools: search, get_document, list_documents, citation_chain, find_related, status, get_metrics
  - Claude Desktop integration ready
  - 36 unit tests

### Changed

- Updated all architecture patterns to "✅ 実装済" status
- Test count increased from 1287 to 1420
- Component completion: 32/32 (100%)

### Fixed

- Async mock usage in hybrid search tests
- Event loop handling in synchronous search methods

## [3.1.0] - 2025-12-27

### Added

- **Progressive Controller** - `monjyu.controller.progressive`
  - Multi-level index management (Level 0-4)
  - Budget-aware query execution
  - Auto-upgrade capabilities

- **Unified Controller** - `monjyu.controller.unified`
  - Single entry point for all search operations
  - Query router integration
  - Result synthesis

- **Query Router** - `monjyu.query.router`
  - Automatic search mode selection
  - Query classification (factual, analytical, comparative, etc.)
  - Level-aware routing

- **Community Report Generator** - `monjyu.index.community_report_generator`
  - LLM-based community summary generation
  - Multi-language support (EN/JA)
  - Batch processing

- **Citation Analysis** - `monjyu.citation`
  - Citation network builder
  - Co-citation analyzer
  - Citation metrics calculation

### Changed

- Architecture upgraded to v3.1
- Specification documents updated (v3.1)

## [3.0.0] - 2025-12-26

### Added

- **MONJYU Facade API** - `monjyu.api`
  - Simple interface for index and search operations
  - State management with JSON persistence
  - Factory pattern for component creation

- **CLI** - `monjyu.cli`
  - `monjyu init` - Initialize workspace
  - `monjyu index` - Index documents
  - `monjyu search` - Search documents
  - `monjyu chat` - Interactive mode
  - `monjyu status` - Show status

- **Vector Search** - `monjyu.query.vector_search`
  - In-memory vector search
  - Cosine similarity and keyword matching
  - Configurable top-k and min score

- **Global Search** - `monjyu.query.global_search`
  - Community-based search
  - Map-Reduce pattern
  - Multi-language prompts

- **Local Search** - `monjyu.query.local_search`
  - Entity-centric graph traversal
  - Relationship extraction
  - Chunk retrieval

### Changed

- Project restructured to modular architecture
- Specification documents created (v3.0)

## [2.0.0] - 2025-12-20

### Added

- **LazySearch Engine** - `monjyu.lazy`
  - Query-time graph construction
  - Claim extraction
  - Relevance testing
  - Iterative deepening

- **Document Processing** - `monjyu.document`
  - PDF parsing with pdfplumber
  - Academic paper structure detection (IMRaD)
  - Text chunking with overlap

- **Embedding** - `monjyu.embedding`
  - Azure OpenAI embeddings
  - Ollama local embeddings
  - Batch processing

- **Index** - `monjyu.index`
  - Entity extraction
  - Relationship extraction
  - Community detection (Louvain)

### Changed

- Migrated from prototype to production-ready architecture

## [1.0.0] - 2025-12-15

### Added

- Initial prototype implementation
- Basic RAG pipeline
- PDF document loading
- Simple vector search

---

## Release Notes

### Upgrade Guide: 3.1.x → 3.2.0

No breaking changes. New features are additive.

```python
# New Hybrid Search usage
from monjyu.query.hybrid_search import HybridSearch, create_hybrid_search

search = create_hybrid_search(
    methods=["vector", "lazy"],
    fusion="rrf"
)
result = await search.search("What is GraphRAG?")
```

### Upgrade Guide: 3.0.x → 3.1.0

No breaking changes. New controllers available.

```python
# New Progressive Controller
from monjyu.controller.progressive import ProgressiveController

controller = ProgressiveController()
result = await controller.search("query", budget=100)
```

---

For more details, see the [documentation](docs/) and [specifications](specs/).

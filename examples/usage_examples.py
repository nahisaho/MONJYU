"""MONJYU Usage Examples.

MONJYUの基本的な使用例を示します。
"""

import asyncio
from pathlib import Path


async def example_1_basic_indexing():
    """例1: 基本的なドキュメントインデックス"""
    from monjyu.document import DocumentPipeline
    from monjyu.index.level0 import Level0IndexBuilder
    from monjyu.embedding import OllamaEmbedder
    
    # Embedder (Ollama使用 - ローカル)
    embedder = OllamaEmbedder(
        host="http://localhost:11434",
        model="nomic-embed-text",
    )
    
    # パイプライン作成
    pipeline = DocumentPipeline()
    builder = Level0IndexBuilder(embedder=embedder)
    
    # ドキュメント処理
    documents, text_units = await pipeline.process_directory(
        Path("./papers/"),
        patterns=["*.pdf", "*.txt"],
    )
    
    print(f"Documents: {len(documents)}")
    print(f"Text Units: {len(text_units)}")
    
    # インデックス構築
    index = await builder.build(documents, text_units)
    
    # 保存
    index.save("./output/index")
    
    return index


async def example_2_vector_search():
    """例2: ベクトル検索"""
    from monjyu.query.vector_search import VectorSearch, VectorSearchConfig
    from monjyu.embedding import OllamaEmbedder
    
    embedder = OllamaEmbedder()
    
    config = VectorSearchConfig(
        top_k=10,
        min_score=0.5,
    )
    
    search = VectorSearch(embedder=embedder, config=config)
    
    # 検索実行
    results = await search.search(
        query="What is GraphRAG?",
        # index=index,  # 事前に構築したインデックス
    )
    
    for i, hit in enumerate(results.hits, 1):
        print(f"{i}. Score: {hit.score:.3f}")
        print(f"   Content: {hit.content[:100]}...")
        print()


async def example_3_lazy_search():
    """例3: LazySearch (Query-time Graph)"""
    from monjyu.lazy import LazySearchEngine, LazySearchConfig
    
    config = LazySearchConfig(
        max_iterations=3,
        relevance_threshold=0.7,
        max_claims=50,
    )
    
    engine = LazySearchEngine(config=config)
    
    result = await engine.search(
        query="How does LazyGraphRAG compare to traditional GraphRAG?",
        # text_units=text_units,
    )
    
    print(f"Answer: {result.answer}")
    print(f"Claims: {len(result.claims)}")
    print(f"Iterations: {result.iterations}")


async def example_4_hybrid_search():
    """例4: ハイブリッド検索 (RRF Fusion)"""
    from monjyu.query.hybrid_search import (
        HybridSearchEngine,
        HybridSearchConfig,
        SearchMethod,
        FusionMethod,
    )
    
    config = HybridSearchConfig(
        methods=[SearchMethod.VECTOR, SearchMethod.LAZY],
        fusion=FusionMethod.RRF,
        rrf_k=60,
        top_k=10,
        parallel=True,
    )
    
    engine = HybridSearchEngine(config=config)
    
    results = await engine.search(
        query="What are the key innovations in recent RAG systems?",
        # index=index,
    )
    
    for hit in results.hits:
        print(f"Score: {hit.score:.3f} | Sources: {hit.sources}")


async def example_5_incremental_index():
    """例5: 差分インデックス更新"""
    from monjyu.index.incremental import (
        IncrementalIndexManager,
        IncrementalIndexConfig,
    )
    
    config = IncrementalIndexConfig(
        output_dir="./output/index",
        batch_size=50,
    )
    
    manager = IncrementalIndexManager(config)
    
    # 変更検出
    # change_set = manager.detect_changes(documents, text_units)
    # print(f"Added: {change_set.added_count}")
    # print(f"Modified: {change_set.modified_count}")
    # print(f"Deleted: {change_set.deleted_count}")
    
    # サマリー表示
    summary = manager.get_summary()
    print(f"Summary: {summary}")


async def example_6_citation_analysis():
    """例6: 引用ネットワーク分析"""
    from monjyu.citation import CitationNetworkBuilder, CoCitationAnalyzer
    
    builder = CitationNetworkBuilder()
    
    # ネットワーク構築
    # network = await builder.build(documents)
    
    # 分析
    # analyzer = CoCitationAnalyzer(network)
    # pairs = analyzer.find_co_citation_pairs(min_count=3)
    
    print("Citation analysis example")


async def example_7_unified_controller():
    """例7: Unified Controller (Auto Mode)"""
    from monjyu.controller.unified import UnifiedController, UnifiedControllerConfig
    
    config = UnifiedControllerConfig(
        default_mode="auto",
        enable_streaming=True,
    )
    
    controller = UnifiedController(config=config)
    
    # Auto mode - 最適な検索方式を自動選択
    # result = await controller.search(
    #     query="Explain the architecture of transformer models",
    # )
    
    print("Unified controller example")


async def example_8_error_handling():
    """例8: エラーハンドリング (Circuit Breaker)"""
    from monjyu.errors import CircuitBreaker, with_retry
    
    # Circuit Breaker
    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60.0,
    )
    
    # Retry デコレータ
    @with_retry(max_attempts=3, backoff_factor=2.0)
    async def call_api():
        async with circuit_breaker:
            # API呼び出し
            pass
    
    print("Error handling example")


async def main():
    """メイン関数"""
    print("=" * 60)
    print("MONJYU Usage Examples")
    print("=" * 60)
    
    # 各例を実行（実際のデータがない場合はスキップ）
    examples = [
        ("1. Basic Indexing", example_1_basic_indexing),
        ("2. Vector Search", example_2_vector_search),
        ("3. Lazy Search", example_3_lazy_search),
        ("4. Hybrid Search", example_4_hybrid_search),
        ("5. Incremental Index", example_5_incremental_index),
        ("6. Citation Analysis", example_6_citation_analysis),
        ("7. Unified Controller", example_7_unified_controller),
        ("8. Error Handling", example_8_error_handling),
    ]
    
    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            await func()
        except Exception as e:
            print(f"Skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())

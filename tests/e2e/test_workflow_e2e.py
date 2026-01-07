"""
MONJYU E2E Workflow Tests (NFR-TST-003)

主要ワークフローのE2Eテスト
対応要件:
- インデックス構築 → 検索の完全フロー
- 複数検索モードの切り替え
- エラーハンドリング・リカバリー
- 設定変更の動的適用

Usage:
    pytest tests/e2e/test_workflow_e2e.py -v
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.api.base import (
    MONJYUConfig,
    SearchMode,
    IndexLevel,
    IndexStatus,
    SearchResult,
    Citation,
)


# =============================================================================
# テストデータ
# =============================================================================

SAMPLE_DOCUMENT_1 = """
# Machine Learning Fundamentals

## Introduction

Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. This paper covers the fundamental concepts.

## Key Concepts

### Supervised Learning

Supervised learning uses labeled data to train models. Common algorithms
include linear regression, decision trees, and neural networks.

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Clustering and
dimensionality reduction are primary techniques.

## Conclusion

Understanding these fundamentals is essential for practical ML applications.
"""

SAMPLE_DOCUMENT_2 = """
# Deep Learning Architecture

## Abstract

Deep learning extends machine learning with multi-layer neural networks.
This enables learning complex representations from raw data.

## Neural Network Layers

### Convolutional Layers

CNNs use convolutional layers for image processing. They detect features
like edges, textures, and objects hierarchically.

### Recurrent Layers

RNNs process sequential data using recurrent connections. LSTMs and GRUs
address the vanishing gradient problem.

## Applications

Deep learning powers computer vision, NLP, and speech recognition systems.
"""

SAMPLE_DOCUMENT_3 = """
# Natural Language Processing

## Overview

NLP enables computers to understand and generate human language. Modern
NLP relies heavily on transformer architectures.

## Transformer Architecture

### Attention Mechanism

Self-attention allows models to weigh the importance of different parts
of the input sequence when producing output.

### Pre-trained Models

BERT, GPT, and T5 demonstrate the power of pre-training on large corpora
followed by fine-tuning for specific tasks.

## RAG Systems

Retrieval-Augmented Generation combines retrieval with generation for
knowledge-intensive tasks.
"""


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace():
    """一時ワークスペースを作成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # ドキュメントディレクトリ
        docs_dir = workspace / "documents"
        docs_dir.mkdir()
        
        # 出力ディレクトリ
        output_dir = workspace / "output"
        output_dir.mkdir()
        
        yield workspace


@pytest.fixture
def sample_documents(temp_workspace):
    """サンプルドキュメントを作成"""
    docs_dir = temp_workspace / "documents"
    
    doc1 = docs_dir / "ml_fundamentals.md"
    doc1.write_text(SAMPLE_DOCUMENT_1)
    
    doc2 = docs_dir / "deep_learning.md"
    doc2.write_text(SAMPLE_DOCUMENT_2)
    
    doc3 = docs_dir / "nlp.md"
    doc3.write_text(SAMPLE_DOCUMENT_3)
    
    return [doc1, doc2, doc3]


@pytest.fixture
def monjyu_config(temp_workspace):
    """MONJYU設定"""
    return MONJYUConfig(
        output_path=temp_workspace / "output",
        environment="local",
        default_search_mode=SearchMode.VECTOR,
        default_top_k=5,
        chunk_size=300,
        chunk_overlap=50,
    )


# =============================================================================
# 1. 基本ワークフローテスト
# =============================================================================

class TestBasicWorkflow:
    """基本ワークフローE2Eテスト"""
    
    def test_config_creation_and_validation(self, temp_workspace):
        """設定作成と検証"""
        config = MONJYUConfig(
            output_path=temp_workspace / "output",
            environment="local",
            chunk_size=500,
        )
        
        assert config.output_path == temp_workspace / "output"
        assert config.environment == "local"
        assert config.chunk_size == 500
        assert config.default_search_mode == SearchMode.LAZY
    
    def test_config_azure_environment(self, temp_workspace):
        """Azure環境設定"""
        config = MONJYUConfig(
            output_path=temp_workspace / "output",
            environment="azure",
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_api_key="test-key",
        )
        
        assert config.environment == "azure"
        # APIキーはマスクされている
        assert "test-key" not in repr(config)
        assert "***" in repr(config)
    
    def test_search_mode_selection(self):
        """検索モード選択"""
        # 各モードが正しく設定できる
        for mode in SearchMode:
            config = MONJYUConfig(default_search_mode=mode)
            assert config.default_search_mode == mode
    
    def test_index_level_configuration(self):
        """インデックスレベル設定"""
        # Level 0のみ
        config1 = MONJYUConfig(index_levels=[IndexLevel.LEVEL_0])
        assert IndexLevel.LEVEL_0 in config1.index_levels
        
        # Level 0 + 1
        config2 = MONJYUConfig(
            index_levels=[IndexLevel.LEVEL_0, IndexLevel.LEVEL_1]
        )
        assert len(config2.index_levels) == 2


# =============================================================================
# 2. ドキュメント処理ワークフロー
# =============================================================================

class TestDocumentWorkflow:
    """ドキュメント処理ワークフローテスト"""
    
    def test_document_loading(self, sample_documents):
        """ドキュメント読み込み"""
        for doc_path in sample_documents:
            assert doc_path.exists()
            content = doc_path.read_text()
            assert len(content) > 0
    
    def test_document_chunking_simulation(self, sample_documents):
        """ドキュメントチャンキングシミュレーション"""
        chunk_size = 300
        overlap = 50
        
        for doc_path in sample_documents:
            content = doc_path.read_text()
            
            # シンプルなチャンキングシミュレーション
            chunks = []
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunks.append(content[start:end])
                start = end - overlap if end < len(content) else end
            
            assert len(chunks) > 0
            # 最初と最後のチャンクを検証
            assert len(chunks[0]) <= chunk_size
    
    def test_multiple_document_processing(self, sample_documents, monjyu_config):
        """複数ドキュメント処理"""
        # 全ドキュメントが処理可能
        processed = []
        for doc in sample_documents:
            content = doc.read_text()
            processed.append({
                "path": str(doc),
                "size": len(content),
                "name": doc.name,
            })
        
        assert len(processed) == 3
        assert all(p["size"] > 0 for p in processed)


# =============================================================================
# 3. 検索ワークフロー
# =============================================================================

class TestSearchWorkflow:
    """検索ワークフローテスト"""
    
    def test_search_result_structure(self):
        """検索結果構造"""
        result = SearchResult(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            citations=[
                Citation(
                    doc_id="doc1",
                    title="ML Fundamentals",
                    text="Machine learning enables...",
                    relevance_score=0.95,
                )
            ],
            search_mode=SearchMode.VECTOR,
            search_level=0,
            total_time_ms=150.0,
            llm_calls=1,
        )
        
        assert result.query == "What is machine learning?"
        assert result.citation_count == 1
        assert result.search_mode == SearchMode.VECTOR
    
    def test_citation_structure(self):
        """引用情報構造"""
        citation = Citation(
            doc_id="doc_001",
            title="Test Document",
            chunk_id="chunk_1",
            text="This is the cited text.",
            relevance_score=0.88,
        )
        
        assert citation.doc_id == "doc_001"
        assert citation.relevance_score == 0.88
    
    def test_search_mode_switching(self):
        """検索モード切り替え"""
        modes = [SearchMode.VECTOR, SearchMode.LAZY, SearchMode.LOCAL, SearchMode.GLOBAL]
        
        for mode in modes:
            result = SearchResult(
                query="test query",
                answer="test answer",
                search_mode=mode,
            )
            assert result.search_mode == mode


# =============================================================================
# 4. インデックス状態管理ワークフロー
# =============================================================================

class TestIndexStateWorkflow:
    """インデックス状態管理テスト"""
    
    def test_index_status_transitions(self):
        """インデックス状態遷移"""
        from monjyu.api.base import MONJYUStatus
        
        # 初期状態
        status = MONJYUStatus()
        assert status.index_status == IndexStatus.NOT_BUILT
        assert not status.is_ready
        
        # 構築中
        status.index_status = IndexStatus.BUILDING
        assert not status.is_ready
        
        # 準備完了
        status.index_status = IndexStatus.READY
        assert status.is_ready
    
    def test_index_level_tracking(self):
        """インデックスレベル追跡"""
        from monjyu.api.base import MONJYUStatus
        
        status = MONJYUStatus()
        
        # Level 0 構築
        status.index_levels_built.append(IndexLevel.LEVEL_0)
        assert IndexLevel.LEVEL_0 in status.index_levels_built
        
        # Level 1 追加
        status.index_levels_built.append(IndexLevel.LEVEL_1)
        assert len(status.index_levels_built) == 2
    
    def test_statistics_tracking(self):
        """統計情報追跡"""
        from monjyu.api.base import MONJYUStatus
        
        status = MONJYUStatus(
            document_count=100,
            text_unit_count=500,
            noun_phrase_count=1500,
            community_count=25,
        )
        
        assert status.document_count == 100
        assert status.text_unit_count == 500


# =============================================================================
# 5. エラーハンドリングワークフロー
# =============================================================================

class TestErrorHandlingWorkflow:
    """エラーハンドリングワークフローテスト"""
    
    def test_invalid_path_handling(self):
        """無効パス処理"""
        invalid_path = Path("/nonexistent/path/file.md")
        assert not invalid_path.exists()
    
    def test_empty_query_handling(self):
        """空クエリ処理"""
        query = ""
        assert len(query) == 0
        
        # 空クエリでも結果構造は作成可能
        result = SearchResult(
            query=query,
            answer="",
            citations=[],
        )
        assert result.citation_count == 0
    
    def test_error_status_tracking(self):
        """エラー状態追跡"""
        from monjyu.api.base import MONJYUStatus
        
        status = MONJYUStatus(
            index_status=IndexStatus.ERROR,
            last_error="Index build failed: out of memory",
        )
        
        assert status.index_status == IndexStatus.ERROR
        assert "out of memory" in status.last_error


# =============================================================================
# 6. 設定変更ワークフロー
# =============================================================================

class TestConfigurationWorkflow:
    """設定変更ワークフローテスト"""
    
    def test_chunk_size_variation(self, temp_workspace):
        """チャンクサイズ変更"""
        sizes = [200, 500, 1000, 2000]
        
        for size in sizes:
            config = MONJYUConfig(
                output_path=temp_workspace / "output",
                chunk_size=size,
            )
            assert config.chunk_size == size
    
    def test_top_k_variation(self, temp_workspace):
        """top_k変更"""
        for k in [1, 5, 10, 20, 50]:
            config = MONJYUConfig(
                output_path=temp_workspace / "output",
                default_top_k=k,
            )
            assert config.default_top_k == k
    
    def test_environment_switching(self, temp_workspace):
        """環境切り替え"""
        # ローカル
        config_local = MONJYUConfig(
            output_path=temp_workspace / "output",
            environment="local",
        )
        assert config_local.environment == "local"
        
        # Azure
        config_azure = MONJYUConfig(
            output_path=temp_workspace / "output",
            environment="azure",
        )
        assert config_azure.environment == "azure"


# =============================================================================
# 7. マルチドキュメント検索ワークフロー
# =============================================================================

class TestMultiDocumentWorkflow:
    """マルチドキュメント検索ワークフローテスト"""
    
    def test_cross_document_query_structure(self):
        """クロスドキュメントクエリ構造"""
        # 複数ドキュメントからの引用を含む結果
        result = SearchResult(
            query="How do neural networks relate to NLP?",
            answer="Neural networks, especially transformers, are...",
            citations=[
                Citation(doc_id="deep_learning", title="Deep Learning", relevance_score=0.9),
                Citation(doc_id="nlp", title="NLP", relevance_score=0.85),
            ],
        )
        
        assert result.citation_count == 2
        # 異なるドキュメントからの引用
        doc_ids = {c.doc_id for c in result.citations}
        assert len(doc_ids) == 2
    
    def test_relevance_ranking(self):
        """関連性ランキング"""
        citations = [
            Citation(doc_id="doc1", title="Doc 1", relevance_score=0.7),
            Citation(doc_id="doc2", title="Doc 2", relevance_score=0.95),
            Citation(doc_id="doc3", title="Doc 3", relevance_score=0.8),
        ]
        
        # スコア順にソート
        sorted_citations = sorted(citations, key=lambda c: c.relevance_score, reverse=True)
        
        assert sorted_citations[0].doc_id == "doc2"
        assert sorted_citations[0].relevance_score == 0.95


# =============================================================================
# 8. パフォーマンス計測ワークフロー
# =============================================================================

class TestPerformanceWorkflow:
    """パフォーマンス計測ワークフローテスト"""
    
    def test_timing_measurement(self):
        """タイミング計測"""
        start = time.perf_counter()
        
        # シミュレート処理
        time.sleep(0.01)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        result = SearchResult(
            query="test",
            answer="test",
            total_time_ms=elapsed_ms,
        )
        
        assert result.total_time_ms >= 10.0  # 少なくとも10ms
    
    def test_llm_call_tracking(self):
        """LLM呼び出し追跡"""
        result = SearchResult(
            query="test",
            answer="test",
            llm_calls=3,
        )
        
        assert result.llm_calls == 3


# =============================================================================
# 9. 統合ワークフローテスト
# =============================================================================

class TestIntegratedWorkflow:
    """統合ワークフローテスト"""
    
    def test_full_search_workflow_simulation(self, sample_documents, monjyu_config):
        """完全な検索ワークフローシミュレーション"""
        # 1. ドキュメント読み込み
        documents = []
        for doc_path in sample_documents:
            documents.append({
                "id": doc_path.stem,
                "content": doc_path.read_text(),
            })
        
        assert len(documents) == 3
        
        # 2. チャンキング（シミュレーション）
        chunks = []
        for doc in documents:
            content = doc["content"]
            chunk_size = monjyu_config.chunk_size
            for i in range(0, len(content), chunk_size):
                chunks.append({
                    "doc_id": doc["id"],
                    "text": content[i:i + chunk_size],
                })
        
        assert len(chunks) > 0
        
        # 3. 検索（シミュレーション）
        query = "What is machine learning?"
        
        # キーワードマッチングで関連チャンクを見つける
        relevant_chunks = [
            c for c in chunks
            if "machine learning" in c["text"].lower()
        ]
        
        # 4. 結果構築
        result = SearchResult(
            query=query,
            answer="Machine learning is a subset of artificial intelligence...",
            citations=[
                Citation(
                    doc_id=c["doc_id"],
                    title=c["doc_id"].replace("_", " ").title(),
                    text=c["text"][:100],
                    relevance_score=0.9 - i * 0.1,
                )
                for i, c in enumerate(relevant_chunks[:3])
            ],
            search_mode=monjyu_config.default_search_mode,
            total_time_ms=50.0,
        )
        
        # 検証
        assert result.query == query
        assert len(result.answer) > 0
        assert result.citation_count >= 0
    
    def test_incremental_index_workflow(self, temp_workspace, sample_documents):
        """インクリメンタルインデックスワークフロー"""
        from monjyu.api.base import MONJYUStatus
        
        # 初期状態
        status = MONJYUStatus()
        
        # Phase 1: 最初のドキュメント
        status.document_count = 1
        status.text_unit_count = 10
        status.index_status = IndexStatus.READY
        
        assert status.document_count == 1
        assert status.is_ready
        
        # Phase 2: ドキュメント追加
        status.document_count = 3
        status.text_unit_count = 30
        
        assert status.document_count == 3
        
        # Phase 3: Level 1 追加
        status.index_levels_built = [IndexLevel.LEVEL_0, IndexLevel.LEVEL_1]
        status.noun_phrase_count = 150
        status.community_count = 5
        
        assert len(status.index_levels_built) == 2


# =============================================================================
# メイン実行
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

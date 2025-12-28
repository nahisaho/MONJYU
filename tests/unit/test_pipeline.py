# Unit Tests for Pipeline
"""Tests for monjyu.document.pipeline module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from monjyu.document.pipeline import (
    BatchResult,
    DocumentProcessingPipeline,
    PipelineConfig,
    ProcessingResult,
)


class TestPipelineConfig:
    """PipelineConfig のテスト"""
    
    def test_default_config(self) -> None:
        """デフォルト設定"""
        config = PipelineConfig()
        
        assert config.chunk_size == 300
        assert config.chunk_overlap == 100
        assert config.tokenizer == "cl100k_base"
        assert config.pdf_strategy == "unstructured"
        assert config.batch_size == 10
        assert config.max_concurrent == 4
    
    def test_custom_config(self) -> None:
        """カスタム設定"""
        config = PipelineConfig(
            chunk_size=500,
            chunk_overlap=50,
            pdf_strategy="azure",
            azure_endpoint="https://test.cognitiveservices.azure.com/",
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.pdf_strategy == "azure"


class TestProcessingResult:
    """ProcessingResult のテスト"""
    
    def test_success_result(self) -> None:
        """成功結果"""
        result = ProcessingResult(
            document=MagicMock(),
            units=[MagicMock(), MagicMock()],
            source_path=Path("/test/file.txt"),
            success=True,
        )
        
        assert result.success
        assert result.error is None
        assert len(result.units) == 2
    
    def test_failure_result(self) -> None:
        """失敗結果"""
        result = ProcessingResult(
            document=None,
            units=[],
            source_path=Path("/test/file.txt"),
            success=False,
            error="File not found",
        )
        
        assert not result.success
        assert result.error == "File not found"


class TestBatchResult:
    """BatchResult のテスト"""
    
    def test_empty_batch(self) -> None:
        """空のバッチ"""
        batch = BatchResult()
        
        assert len(batch.results) == 0
        assert batch.success_count == 0
        assert batch.failure_count == 0
    
    def test_add_success(self) -> None:
        """成功結果の追加"""
        batch = BatchResult()
        result = ProcessingResult(
            document=MagicMock(),
            units=[],
            source_path=Path("/test/file.txt"),
            success=True,
        )
        
        batch.add_result(result)
        
        assert batch.success_count == 1
        assert batch.failure_count == 0
    
    def test_add_failure(self) -> None:
        """失敗結果の追加"""
        batch = BatchResult()
        result = ProcessingResult(
            document=None,
            units=[],
            source_path=Path("/test/file.txt"),
            success=False,
            error="Error message",
        )
        
        batch.add_result(result)
        
        assert batch.success_count == 0
        assert batch.failure_count == 1
        assert "/test/file.txt" in batch.errors
    
    def test_mixed_results(self) -> None:
        """混合結果"""
        batch = BatchResult()
        
        batch.add_result(ProcessingResult(
            document=MagicMock(), units=[], source_path=Path("/a.txt"), success=True
        ))
        batch.add_result(ProcessingResult(
            document=None, units=[], source_path=Path("/b.txt"), success=False, error="err"
        ))
        batch.add_result(ProcessingResult(
            document=MagicMock(), units=[], source_path=Path("/c.txt"), success=True
        ))
        
        assert batch.success_count == 2
        assert batch.failure_count == 1


class TestDocumentProcessingPipeline:
    """DocumentProcessingPipeline のテスト"""
    
    @pytest.fixture
    def pipeline(self) -> DocumentProcessingPipeline:
        """デフォルトパイプライン"""
        return DocumentProcessingPipeline()
    
    @pytest.fixture
    def custom_pipeline(self) -> DocumentProcessingPipeline:
        """カスタム設定パイプライン"""
        config = PipelineConfig(
            chunk_size=200,
            chunk_overlap=30,
        )
        return DocumentProcessingPipeline(config)
    
    # --- Initialization tests ---
    
    def test_init_default(self, pipeline: DocumentProcessingPipeline) -> None:
        """デフォルト初期化"""
        assert pipeline.config.chunk_size == 300
        assert pipeline.loader is not None
        assert pipeline.parser is not None
        assert pipeline.chunker is not None
    
    def test_init_custom(self, custom_pipeline: DocumentProcessingPipeline) -> None:
        """カスタム初期化"""
        assert custom_pipeline.config.chunk_size == 200
        assert custom_pipeline.chunker.chunk_size == 200
    
    def test_init_azure_without_endpoint(self) -> None:
        """Azureエンドポイントなしでエラー"""
        config = PipelineConfig(
            pdf_strategy="azure",
            azure_endpoint=None,
        )
        
        with pytest.raises(ValueError, match="エンドポイント"):
            DocumentProcessingPipeline(config)
    
    # --- process_file tests ---
    
    def test_process_file_txt(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """テキストファイル処理"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content. " * 50)
        
        result = pipeline.process_file(test_file)
        
        assert result.success
        assert result.document is not None
        assert len(result.units) > 0
        assert result.source_path == test_file
    
    def test_process_file_md(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """Markdownファイル処理"""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Title\n\n## Introduction\n\nContent here. " * 20)
        
        result = pipeline.process_file(test_file)
        
        assert result.success
        assert result.document is not None
    
    def test_process_file_nonexistent(
        self,
        pipeline: DocumentProcessingPipeline,
    ) -> None:
        """存在しないファイル"""
        result = pipeline.process_file(Path("/nonexistent/file.txt"))
        
        assert not result.success
        assert result.error is not None
        assert result.document is None
    
    def test_process_file_unknown_extension(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """不明な拡張子のファイルも処理可能（テキストとして扱う）"""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        
        result = pipeline.process_file(test_file)
        
        # パイプラインは未知の形式もテキストとして処理可能
        assert result.success
        assert result.document is not None
        assert result.document.file_type == "xyz"
    
    # --- process_directory tests ---
    
    def test_process_directory(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """ディレクトリ処理"""
        (tmp_path / "a.txt").write_text("Content A " * 20)
        (tmp_path / "b.txt").write_text("Content B " * 20)
        (tmp_path / "c.txt").write_text("Content C " * 20)
        
        batch_result = pipeline.process_directory(tmp_path)
        
        assert batch_result.success_count == 3
        assert batch_result.failure_count == 0
    
    def test_process_directory_with_pattern(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """パターン付きディレクトリ処理"""
        (tmp_path / "a.txt").write_text("Content A")
        (tmp_path / "b.md").write_text("Content B")
        
        batch_result = pipeline.process_directory(tmp_path, pattern="*.txt")
        
        assert batch_result.success_count == 1
    
    def test_process_directory_mixed(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """混合ファイル形式のディレクトリ処理（サポート形式のみ処理）"""
        (tmp_path / "good.txt").write_text("Good content " * 20)
        (tmp_path / "other.xyz").write_text("Other content " * 20)  # サポート外、スキップ
        (tmp_path / "also_good.md").write_text("Also good " * 20)
        
        batch_result = pipeline.process_directory(tmp_path)
        
        # サポートされている.txtと.mdのみ処理（.xyzはlist_filesでフィルタリング）
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 0
    
    # --- iter_process tests ---
    
    def test_iter_process(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """イテレータ処理"""
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i} " * 20)
            files.append(f)
        
        results = list(pipeline.iter_process(files))
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    # --- get_stats tests ---
    
    def test_get_stats(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """統計取得"""
        (tmp_path / "a.txt").write_text("Content " * 100)
        (tmp_path / "b.txt").write_text("Content " * 100)
        
        batch_result = pipeline.process_directory(tmp_path)
        stats = pipeline.get_stats(batch_result)
        
        assert stats["total_files"] == 2
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["total_units"] > 0
        assert stats["total_tokens"] > 0


@pytest.mark.asyncio
class TestDocumentProcessingPipelineAsync:
    """DocumentProcessingPipeline の非同期テスト"""
    
    @pytest.fixture
    def pipeline(self) -> DocumentProcessingPipeline:
        """パイプライン"""
        return DocumentProcessingPipeline()
    
    async def test_process_file_async(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """非同期ファイル処理"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Async content " * 50)
        
        result = await pipeline.process_file_async(test_file)
        
        assert result.success
        assert result.document is not None
    
    async def test_process_directory_async(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """非同期ディレクトリ処理"""
        (tmp_path / "a.txt").write_text("Content A " * 20)
        (tmp_path / "b.txt").write_text("Content B " * 20)
        
        batch_result = await pipeline.process_directory_async(tmp_path)
        
        assert batch_result.success_count == 2
    
    async def test_iter_process_async(
        self,
        pipeline: DocumentProcessingPipeline,
        tmp_path: Path,
    ) -> None:
        """非同期イテレータ処理"""
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i} " * 20)
            files.append(f)
        
        results = []
        async for result in pipeline.iter_process_async(files):
            results.append(result)
        
        assert len(results) == 3

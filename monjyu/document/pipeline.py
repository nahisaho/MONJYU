# Document Processing Pipeline
"""
Unified document processing pipeline for MONJYU.

Provides a facade that combines FileLoader, DocumentParser, and TextChunker
for end-to-end document processing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from monjyu.document.chunker import TextChunker
from monjyu.document.loader import FileLoader
from monjyu.document.models import AcademicPaperDocument, TextUnit
from monjyu.document.parser import DocumentParser
from monjyu.document.pdf import (
    HAS_AZURE,
    PDFProcessorProtocol,
    UnstructuredPDFProcessor,
)


@dataclass
class PipelineConfig:
    """パイプライン設定
    
    Attributes:
        chunk_size: チャンクサイズ（トークン数）
        chunk_overlap: オーバーラップ（トークン数）
        tokenizer: トークナイザー名
        pdf_strategy: PDF処理戦略
        azure_endpoint: Azure Document Intelligence エンドポイント
        azure_api_key: Azure APIキー
        batch_size: バッチ処理サイズ
        max_concurrent: 最大並行処理数
    """
    chunk_size: int = 300
    chunk_overlap: int = 100
    tokenizer: str = "cl100k_base"
    pdf_strategy: Literal["unstructured", "azure"] = "unstructured"
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    batch_size: int = 10
    max_concurrent: int = 4


@dataclass
class ProcessingResult:
    """処理結果
    
    Attributes:
        document: パースされた文書
        units: 分割されたTextUnit
        source_path: 元ファイルパス
        success: 処理成功フラグ
        error: エラーメッセージ（失敗時）
    """
    document: AcademicPaperDocument | None
    units: list[TextUnit]
    source_path: Path
    success: bool
    error: str | None = None


@dataclass
class BatchResult:
    """バッチ処理結果
    
    Attributes:
        results: 処理結果リスト
        success_count: 成功数
        failure_count: 失敗数
        errors: エラーマップ（パス→エラーメッセージ）
    """
    results: list[ProcessingResult] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    
    def add_result(self, result: ProcessingResult) -> None:
        """結果を追加"""
        self.results.append(result)
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if result.error:
                self.errors[str(result.source_path)] = result.error


class DocumentProcessingPipeline:
    """文書処理パイプライン
    
    FileLoader, DocumentParser, TextChunkerを統合し、
    エンドツーエンドの文書処理を提供する。
    
    Example:
        >>> config = PipelineConfig(chunk_size=500)
        >>> pipeline = DocumentProcessingPipeline(config)
        >>> result = pipeline.process_file("paper.pdf")
        >>> print(len(result.units))
        42
        
        >>> # バッチ処理
        >>> batch_result = pipeline.process_directory("./papers")
        >>> print(f"成功: {batch_result.success_count}, 失敗: {batch_result.failure_count}")
    """
    
    def __init__(self, config: PipelineConfig | None = None) -> None:
        """初期化
        
        Args:
            config: パイプライン設定
        """
        self.config = config or PipelineConfig()
        
        # コンポーネント初期化
        self._loader = FileLoader()
        self._chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            tokenizer_name=self.config.tokenizer,
        )
        
        # PDFプロセッサー選択
        self._pdf_processor = self._create_pdf_processor()
        
        # パーサー初期化
        self._parser = DocumentParser(pdf_processor=self._pdf_processor)
    
    def _create_pdf_processor(self) -> PDFProcessorProtocol:
        """PDF処理戦略を作成"""
        if self.config.pdf_strategy == "azure":
            if not HAS_AZURE:
                msg = "Azure Document Intelligence を使用するには azure-ai-documentintelligence をインストールしてください"
                raise ImportError(msg)
            
            if not self.config.azure_endpoint:
                msg = "Azure エンドポイントが設定されていません"
                raise ValueError(msg)
            
            from monjyu.document.pdf import AzureDocIntelPDFProcessor
            return AzureDocIntelPDFProcessor(
                endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key,
            )
        
        return UnstructuredPDFProcessor()
    
    @property
    def loader(self) -> FileLoader:
        """ファイルローダーを取得"""
        return self._loader
    
    @property
    def parser(self) -> DocumentParser:
        """パーサーを取得"""
        return self._parser
    
    @property
    def chunker(self) -> TextChunker:
        """チャンカーを取得"""
        return self._chunker
    
    def process_file(self, file_path: str | Path) -> ProcessingResult:
        """単一ファイルを処理
        
        Args:
            file_path: 処理するファイルのパス
            
        Returns:
            処理結果
        """
        file_path = Path(file_path)
        
        try:
            # ファイルを読み込み
            content = self._loader.load(file_path)
            file_format = file_path.suffix.lstrip(".")
            
            # パース
            document = self._parser.parse(content, file_format)
            
            # チャンキング
            units = self._chunker.chunk(document)
            
            return ProcessingResult(
                document=document,
                units=units,
                source_path=file_path,
                success=True,
            )
            
        except Exception as e:
            return ProcessingResult(
                document=None,
                units=[],
                source_path=file_path,
                success=False,
                error=str(e),
            )
    
    def process_directory(
        self,
        directory: str | Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> BatchResult:
        """ディレクトリ内のファイルを処理
        
        Args:
            directory: 処理するディレクトリ
            pattern: ファイルパターン（glob）
            recursive: 再帰的に処理するか
            
        Returns:
            バッチ処理結果
        """
        directory = Path(directory)
        batch_result = BatchResult()
        
        # ファイル一覧を取得
        files = self._loader.list_files(directory, pattern, recursive)
        
        # 順次処理
        for file_path in files:
            result = self.process_file(file_path)
            batch_result.add_result(result)
        
        return batch_result
    
    def iter_process(
        self,
        files: list[str | Path],
    ) -> Iterator[ProcessingResult]:
        """ファイルを順次処理してイテレート
        
        Args:
            files: 処理するファイルのリスト
            
        Yields:
            処理結果
        """
        for file_path in files:
            yield self.process_file(file_path)
    
    async def process_file_async(self, file_path: str | Path) -> ProcessingResult:
        """非同期でファイルを処理
        
        Args:
            file_path: 処理するファイルのパス
            
        Returns:
            処理結果
        """
        # 同期処理をスレッドプールで実行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_file, file_path)
    
    async def process_directory_async(
        self,
        directory: str | Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> BatchResult:
        """非同期でディレクトリを処理
        
        Args:
            directory: 処理するディレクトリ
            pattern: ファイルパターン
            recursive: 再帰的に処理するか
            
        Returns:
            バッチ処理結果
        """
        directory = Path(directory)
        batch_result = BatchResult()
        
        # ファイル一覧を取得
        files = self._loader.list_files(directory, pattern, recursive)
        
        # 並行処理
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(file_path: Path) -> ProcessingResult:
            async with semaphore:
                return await self.process_file_async(file_path)
        
        tasks = [process_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for file_path, result in zip(files, results, strict=False):
            if isinstance(result, Exception):
                batch_result.add_result(ProcessingResult(
                    document=None,
                    units=[],
                    source_path=file_path,
                    success=False,
                    error=str(result),
                ))
            else:
                batch_result.add_result(result)
        
        return batch_result
    
    async def iter_process_async(
        self,
        files: list[str | Path],
    ) -> AsyncIterator[ProcessingResult]:
        """非同期でファイルを順次処理してイテレート
        
        Args:
            files: 処理するファイルのリスト
            
        Yields:
            処理結果
        """
        for file_path in files:
            yield await self.process_file_async(file_path)
    
    def get_stats(self, batch_result: BatchResult) -> dict:
        """バッチ処理の統計を取得
        
        Args:
            batch_result: バッチ処理結果
            
        Returns:
            統計情報
        """
        total_units = sum(
            len(r.units) for r in batch_result.results if r.success
        )
        total_tokens = sum(
            unit.n_tokens
            for r in batch_result.results
            if r.success
            for unit in r.units
        )
        
        return {
            "total_files": len(batch_result.results),
            "success_count": batch_result.success_count,
            "failure_count": batch_result.failure_count,
            "success_rate": batch_result.success_count / len(batch_result.results) if batch_result.results else 0,
            "total_units": total_units,
            "total_tokens": total_tokens,
            "avg_units_per_file": total_units / batch_result.success_count if batch_result.success_count else 0,
        }

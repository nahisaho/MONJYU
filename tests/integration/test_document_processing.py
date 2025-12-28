# Integration Tests for Document Processing
"""
Integration tests for monjyu.document module.

These tests verify the end-to-end functionality of the document processing pipeline,
including interaction between components.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from monjyu.document import (
    AcademicPaperDocument,
    DocumentProcessingPipeline,
    FileLoader,
    PipelineConfig,
    TextChunker,
)
from monjyu.document.parser import DocumentParser


class TestDocumentProcessingIntegration:
    """文書処理の統合テスト"""
    
    @pytest.fixture
    def sample_paper_txt(self, tmp_path: Path) -> Path:
        """サンプル学術論文（テキスト形式）"""
        content = dedent("""
            Machine Learning for Natural Language Processing: A Survey
            
            John Smith, Jane Doe
            University of Example
            
            Abstract
            
            This paper provides a comprehensive survey of machine learning techniques
            applied to natural language processing tasks. We review recent advances
            in deep learning, transformer models, and their applications to text
            classification, named entity recognition, and machine translation.
            
            1. Introduction
            
            Natural language processing (NLP) has seen tremendous progress in recent
            years, largely driven by advances in machine learning and deep learning.
            The introduction of transformer architectures has revolutionized the field,
            enabling models to capture long-range dependencies in text more effectively
            than previous approaches.
            
            In this survey, we provide an overview of the key techniques and their
            applications. We focus on supervised learning approaches, as well as
            recent developments in unsupervised and self-supervised learning.
            
            2. Methods
            
            2.1 Traditional Machine Learning
            
            Traditional machine learning approaches for NLP include support vector
            machines, naive Bayes classifiers, and hidden Markov models. These methods
            rely on hand-crafted features and have been widely used for text
            classification and sequence labeling tasks.
            
            2.2 Deep Learning
            
            Deep learning approaches, particularly recurrent neural networks (RNNs)
            and convolutional neural networks (CNNs), have shown significant
            improvements over traditional methods. Long short-term memory (LSTM)
            networks address the vanishing gradient problem in RNNs.
            
            2.3 Transformer Models
            
            The transformer architecture, introduced by Vaswani et al. (2017), has
            become the dominant approach for NLP tasks. Models like BERT, GPT, and
            T5 have achieved state-of-the-art results across numerous benchmarks.
            
            3. Results
            
            Our analysis shows that transformer-based models consistently outperform
            traditional approaches across all evaluated tasks. Table 1 summarizes
            the performance metrics on standard benchmarks.
            
            The results indicate that pre-training on large corpora followed by
            fine-tuning on task-specific data yields the best performance.
            
            4. Discussion
            
            The success of transformer models can be attributed to several factors:
            the self-attention mechanism, positional encodings, and the ability to
            scale with data and compute. However, challenges remain in terms of
            computational efficiency and interpretability.
            
            5. Conclusion
            
            This survey has provided an overview of machine learning techniques for
            NLP. Future work should focus on developing more efficient architectures
            and improving model interpretability.
            
            References
            
            [1] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
            [2] Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional
                transformers. NAACL.
            [3] Brown, T., et al. (2020). Language models are few-shot learners. NeurIPS.
        """).strip()
        
        file_path = tmp_path / "sample_paper.txt"
        file_path.write_text(content)
        return file_path
    
    @pytest.fixture
    def sample_paper_md(self, tmp_path: Path) -> Path:
        """サンプル学術論文（Markdown形式）"""
        content = dedent("""
            # Deep Learning for Computer Vision
            
            **Authors:** Alice Wang, Bob Chen
            
            ## Abstract
            
            This paper reviews deep learning methods for computer vision tasks.
            We cover convolutional neural networks, object detection, and
            image segmentation techniques.
            
            ## 1. Introduction
            
            Computer vision has experienced rapid advancement with the adoption
            of deep learning techniques. Convolutional neural networks (CNNs)
            have become the standard architecture for image-related tasks.
            
            ## 2. Methods
            
            ### 2.1 Convolutional Neural Networks
            
            CNNs use learnable filters to extract hierarchical features from
            images. Key components include convolutional layers, pooling layers,
            and fully connected layers.
            
            ### 2.2 Object Detection
            
            Object detection combines classification and localization. Popular
            approaches include YOLO, Faster R-CNN, and SSD.
            
            ## 3. Results
            
            | Model | mAP | FPS |
            |-------|-----|-----|
            | YOLO  | 45.2| 60  |
            | RCNN  | 48.1| 5   |
            
            ## 4. Conclusion
            
            Deep learning has transformed computer vision. Future directions
            include vision transformers and self-supervised learning.
            
            ## References
            
            1. Krizhevsky, A. (2012). ImageNet Classification with Deep CNNs.
            2. Redmon, J. (2016). You Only Look Once: Real-Time Object Detection.
        """).strip()
        
        file_path = tmp_path / "cv_paper.md"
        file_path.write_text(content)
        return file_path
    
    @pytest.fixture
    def sample_json_data(self, tmp_path: Path) -> Path:
        """サンプルJSONデータ"""
        data = {
            "title": "Sample Research Data",
            "description": "This is a sample JSON file containing research data.",
            "sections": [
                {
                    "name": "Introduction",
                    "content": "This section introduces the research topic."
                },
                {
                    "name": "Data",
                    "content": "This section describes the data used in the study."
                },
            ],
            "keywords": ["research", "data", "analysis"],
        }
        
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(data, indent=2))
        return file_path
    
    # --- End-to-end pipeline tests ---
    
    def test_pipeline_txt_file(self, sample_paper_txt: Path) -> None:
        """テキストファイルのパイプライン処理"""
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(sample_paper_txt)
        
        assert result.success
        assert result.document is not None
        
        # ドキュメント内容を検証
        doc = result.document
        assert "Machine Learning" in doc.full_text
        assert "NLP" in doc.full_text or "natural language" in doc.full_text.lower()
        
        # セクションが抽出されているか
        assert len(doc.sections) > 0
        
        # チャンクが生成されているか
        assert len(result.units) > 0
        assert all(unit.n_tokens > 0 for unit in result.units)
    
    def test_pipeline_md_file(self, sample_paper_md: Path) -> None:
        """Markdownファイルのパイプライン処理"""
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(sample_paper_md)
        
        assert result.success
        assert result.document is not None
        
        # Markdown内容を検証
        doc = result.document
        assert "Deep Learning" in doc.full_text or "computer vision" in doc.full_text.lower()
        
        # チャンクが生成されているか
        assert len(result.units) > 0
    
    def test_pipeline_json_file(self, sample_json_data: Path) -> None:
        """JSONファイルのパイプライン処理"""
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(sample_json_data)
        
        assert result.success
        assert result.document is not None
        assert len(result.units) > 0
    
    def test_pipeline_directory(
        self,
        sample_paper_txt: Path,
        sample_paper_md: Path,
        sample_json_data: Path,
    ) -> None:
        """ディレクトリのバッチ処理"""
        pipeline = DocumentProcessingPipeline()
        directory = sample_paper_txt.parent
        
        batch_result = pipeline.process_directory(directory)
        
        assert batch_result.success_count >= 3
        assert batch_result.failure_count == 0
        
        # 統計を確認
        stats = pipeline.get_stats(batch_result)
        assert stats["total_files"] >= 3
        assert stats["success_rate"] == 1.0
        assert stats["total_units"] > 0
    
    def test_pipeline_with_custom_config(self, sample_paper_txt: Path) -> None:
        """カスタム設定でのパイプライン処理"""
        config = PipelineConfig(
            chunk_size=100,
            chunk_overlap=20,
        )
        pipeline = DocumentProcessingPipeline(config)
        
        result = pipeline.process_file(sample_paper_txt)
        
        assert result.success
        # 小さいチャンクサイズなのでより多くのチャンクが生成される
        assert len(result.units) > 5
    
    # --- Component interaction tests ---
    
    def test_loader_parser_chunker_flow(self, sample_paper_txt: Path) -> None:
        """Loader → Parser → Chunker のフロー"""
        loader = FileLoader()
        parser = DocumentParser()
        chunker = TextChunker(chunk_size=200, overlap=50)
        
        # ロード（bytesを返す）
        content = loader.load(sample_paper_txt)
        assert isinstance(content, bytes)
        file_type = sample_paper_txt.suffix.lstrip(".")
        
        # パース
        doc = parser.parse(content, file_type)
        assert isinstance(doc, AcademicPaperDocument)
        
        # チャンキング
        units = chunker.chunk(doc)
        assert len(units) > 0
        
        # 各チャンクがdocument_idを持つ
        for unit in units:
            assert unit.document_id is not None
    
    def test_section_type_detection(self, sample_paper_txt: Path) -> None:
        """セクションタイプの検出"""
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(sample_paper_txt)
        
        assert result.success
        doc = result.document
        
        # IMRaD構造のセクションが検出されているか
        section_types = [s.section_type for s in doc.sections]
        
        # 少なくとも introduction, methods, results のいずれかが検出されているはず
        imrad_types = {"introduction", "methods", "results", "discussion", "conclusion"}
        detected_imrad = set(section_types) & imrad_types
        assert len(detected_imrad) >= 1, f"Detected types: {section_types}"
    
    def test_chunk_metadata_preserved(self, sample_paper_txt: Path) -> None:
        """チャンクメタデータの保持"""
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(sample_paper_txt)
        
        assert result.success
        
        for unit in result.units:
            # メタデータが存在する
            assert unit.metadata is not None
            
            # titleがメタデータに含まれている（新API）
            assert "title" in unit.metadata
            
            # セクションタイプが設定されている
            assert unit.section_type is not None
    
    # --- Error handling tests ---
    
    def test_graceful_error_handling(self, tmp_path: Path) -> None:
        """エラー発生時の優雅な処理"""
        pipeline = DocumentProcessingPipeline()
        
        # 存在するファイルと存在しないファイルを混在
        good_file = tmp_path / "good.txt"
        good_file.write_text("Good content " * 20)
        bad_file = tmp_path / "nonexistent.txt"
        
        results = list(pipeline.iter_process([good_file, bad_file]))
        
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success
        assert results[1].error is not None
    
    def test_empty_file_handling(self, tmp_path: Path) -> None:
        """空ファイルの処理"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(empty_file)
        
        # 空ファイルでもエラーにはならない
        assert result.success
        assert result.document is not None
        assert len(result.units) == 0


@pytest.mark.asyncio
class TestDocumentProcessingIntegrationAsync:
    """非同期統合テスト"""
    
    @pytest.fixture
    def sample_files(self, tmp_path: Path) -> list[Path]:
        """複数のサンプルファイル"""
        files = []
        for i in range(5):
            f = tmp_path / f"doc_{i}.txt"
            f.write_text(f"Document {i} content. " * 50)
            files.append(f)
        return files
    
    async def test_async_directory_processing(
        self,
        sample_files: list[Path],
    ) -> None:
        """非同期ディレクトリ処理"""
        pipeline = DocumentProcessingPipeline()
        directory = sample_files[0].parent
        
        batch_result = await pipeline.process_directory_async(directory)
        
        assert batch_result.success_count == 5
        assert batch_result.failure_count == 0
    
    async def test_async_concurrent_processing(
        self,
        sample_files: list[Path],
    ) -> None:
        """並行処理の動作確認"""
        config = PipelineConfig(max_concurrent=2)
        pipeline = DocumentProcessingPipeline(config)
        directory = sample_files[0].parent
        
        batch_result = await pipeline.process_directory_async(directory)
        
        assert batch_result.success_count == 5
    
    async def test_async_iterator(self, sample_files: list[Path]) -> None:
        """非同期イテレータ"""
        pipeline = DocumentProcessingPipeline()
        
        results = []
        async for result in pipeline.iter_process_async(sample_files):
            results.append(result)
        
        assert len(results) == 5
        assert all(r.success for r in results)


class TestDocumentProcessingPerformance:
    """パフォーマンス関連のテスト"""
    
    def test_large_document(self, tmp_path: Path) -> None:
        """大きなドキュメントの処理"""
        # 約10万文字のドキュメント
        large_content = "This is a test sentence. " * 5000
        large_file = tmp_path / "large.txt"
        large_file.write_text(large_content)
        
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(large_file)
        
        assert result.success
        assert len(result.units) > 100  # 多数のチャンクが生成される
    
    def test_many_small_files(self, tmp_path: Path) -> None:
        """多数の小さいファイルの処理"""
        for i in range(50):
            f = tmp_path / f"small_{i}.txt"
            f.write_text(f"Small document {i}")
        
        pipeline = DocumentProcessingPipeline()
        batch_result = pipeline.process_directory(tmp_path)
        
        assert batch_result.success_count == 50
        
        stats = pipeline.get_stats(batch_result)
        assert stats["total_files"] == 50

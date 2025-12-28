# Unit Tests for TextChunker
"""Tests for monjyu.document.chunker module."""

from __future__ import annotations

import pytest

from monjyu.document.chunker import TextChunker
from monjyu.document.models import AcademicPaperDocument, AcademicSection, Author


class TestTextChunker:
    """TextChunker の単体テスト"""
    
    @pytest.fixture
    def chunker(self) -> TextChunker:
        """デフォルト設定のチャンカー"""
        return TextChunker(chunk_size=100, overlap=20)
    
    @pytest.fixture
    def sample_document(self) -> AcademicPaperDocument:
        """テスト用ドキュメント"""
        return AcademicPaperDocument(
            file_name="paper.pdf",
            file_type="pdf",
            title="Test Paper",
            authors=[Author(name="Test Author")],
            abstract="This is a test abstract for the paper.",
            sections=[
                AcademicSection(
                    heading="Introduction",
                    level=1,
                    section_type="introduction",
                    content="This is the introduction section. " * 20,
                ),
                AcademicSection(
                    heading="Methods",
                    level=1,
                    section_type="methods",
                    content="This is the methods section. " * 20,
                ),
            ],
            raw_text="Full text of the document. " * 50,
        )
    
    # --- __init__ tests ---
    
    def test_init_default(self) -> None:
        """デフォルト設定"""
        chunker = TextChunker()
        assert chunker.chunk_size == 300
        assert chunker.overlap == 100
    
    def test_init_custom(self) -> None:
        """カスタム設定"""
        chunker = TextChunker(chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
    
    # --- count_tokens tests ---
    
    def test_count_tokens(self, chunker: TextChunker) -> None:
        """トークンカウント"""
        text = "Hello world!"
        tokens = chunker.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_count_tokens_empty(self, chunker: TextChunker) -> None:
        """空文字列のトークンカウント"""
        assert chunker.count_tokens("") == 0
    
    # --- chunk tests ---
    
    def test_chunk_document(
        self,
        chunker: TextChunker,
        sample_document: AcademicPaperDocument,
    ) -> None:
        """ドキュメントのチャンキング"""
        units = chunker.chunk(sample_document)
        
        assert len(units) > 0
        assert all(unit.text for unit in units)
        assert all(unit.n_tokens > 0 for unit in units)
    
    def test_chunk_preserves_metadata(
        self,
        chunker: TextChunker,
        sample_document: AcademicPaperDocument,
    ) -> None:
        """メタデータが保持される"""
        units = chunker.chunk(sample_document)
        
        for unit in units:
            assert unit.document_id is not None
            # metadataにtitleが含まれる
            assert "title" in unit.metadata
    
    def test_chunk_section_types(
        self,
        chunker: TextChunker,
        sample_document: AcademicPaperDocument,
    ) -> None:
        """セクションタイプが設定される"""
        units = chunker.chunk(sample_document)
        
        section_types = {unit.section_type for unit in units}
        # 少なくとも1つのセクションタイプがあるはず
        assert len(section_types) >= 1
    
    def test_chunk_unit_ids_unique(
        self,
        chunker: TextChunker,
        sample_document: AcademicPaperDocument,
    ) -> None:
        """各ユニットのIDがユニーク"""
        units = chunker.chunk(sample_document)
        
        ids = [unit.id for unit in units]
        assert len(ids) == len(set(ids))
    
    def test_chunk_custom_size(
        self,
        sample_document: AcademicPaperDocument,
    ) -> None:
        """カスタムチャンクサイズ"""
        small_chunker = TextChunker(chunk_size=50, overlap=10)
        large_chunker = TextChunker(chunk_size=500, overlap=50)
        
        small_units = small_chunker.chunk(sample_document)
        large_units = large_chunker.chunk(sample_document)
        
        # 小さいチャンクのほうが多くなるはず
        assert len(small_units) >= len(large_units)
    
    # --- Edge cases ---
    
    def test_chunk_empty_document(self, chunker: TextChunker) -> None:
        """空のドキュメント"""
        empty_doc = AcademicPaperDocument(
            file_name="empty.pdf",
            file_type="pdf",
            title="",
            authors=[],
            abstract="",
            sections=[],
            raw_text="",
        )
        
        units = chunker.chunk(empty_doc)
        # 空のドキュメントは空のリストを返す
        assert len(units) == 0
    
    def test_chunk_document_with_only_abstract(self, chunker: TextChunker) -> None:
        """アブストラクトのみのドキュメント"""
        doc = AcademicPaperDocument(
            file_name="abstract.pdf",
            file_type="pdf",
            title="Title",
            authors=[],
            abstract="This is a test abstract. " * 10,
            sections=[],
            raw_text="This is a test abstract. " * 10,
        )
        
        units = chunker.chunk(doc)
        assert len(units) >= 1
    
    def test_chunk_document_with_only_sections(self, chunker: TextChunker) -> None:
        """セクションのみのドキュメント"""
        doc = AcademicPaperDocument(
            file_name="sections.pdf",
            file_type="pdf",
            title="Title",
            authors=[],
            abstract="",
            sections=[
                AcademicSection(
                    heading="Section One",
                    level=1,
                    section_type="body",
                    content="Content of section one. " * 30,
                ),
            ],
            raw_text="Content of section one. " * 30,
        )
        
        units = chunker.chunk(doc)
        assert len(units) >= 1


class TestTextChunkerTokenizers:
    """トークナイザーのテスト"""
    
    def test_default_tokenizer(self) -> None:
        """デフォルトトークナイザー（cl100k_base）"""
        chunker = TextChunker(tokenizer_name="cl100k_base")
        tokens = chunker.count_tokens("Hello, world!")
        assert tokens > 0
    
    def test_different_tokenizer(self) -> None:
        """異なるトークナイザー"""
        chunker = TextChunker(tokenizer_name="p50k_base")
        tokens = chunker.count_tokens("Hello, world!")
        assert tokens > 0
    
    def test_invalid_tokenizer(self) -> None:
        """無効なトークナイザー"""
        with pytest.raises(Exception):
            TextChunker(tokenizer_name="invalid_tokenizer_name")


class TestEstimateChunks:
    """estimate_chunks のテスト"""
    
    @pytest.fixture
    def chunker(self) -> TextChunker:
        return TextChunker(chunk_size=100, overlap=20)
    
    @pytest.fixture
    def sample_document(self) -> AcademicPaperDocument:
        """テスト用ドキュメント"""
        return AcademicPaperDocument(
            file_name="test.pdf",
            file_type="pdf",
            title="Test",
            authors=[],
            abstract="Test sentence. " * 100,
            sections=[],
            raw_text="Test sentence. " * 100,
        )
    
    def test_estimate_chunks(
        self,
        chunker: TextChunker,
        sample_document: AcademicPaperDocument,
    ) -> None:
        """チャンク数の推定"""
        estimate = chunker.estimate_chunks(sample_document)
        
        assert isinstance(estimate, int)
        assert estimate > 0
    
    def test_estimate_chunks_short(self, chunker: TextChunker) -> None:
        """短いテキストの推定"""
        doc = AcademicPaperDocument(
            file_name="short.pdf",
            file_type="pdf",
            title="Short",
            authors=[],
            abstract="Short.",
            sections=[],
            raw_text="Short.",
        )
        estimate = chunker.estimate_chunks(doc)
        
        assert estimate == 1
    
    def test_estimate_chunks_empty(self, chunker: TextChunker) -> None:
        """空ドキュメントの推定"""
        doc = AcademicPaperDocument(
            file_name="empty.pdf",
            file_type="pdf",
            title="",
            authors=[],
            abstract="",
            sections=[],
            raw_text="",
        )
        estimate = chunker.estimate_chunks(doc)
        
        # 空でも最小1を返す可能性がある
        assert estimate >= 0

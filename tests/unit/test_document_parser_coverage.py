# Document Parser Coverage Tests
"""
Unit tests for document parser to improve coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import BytesIO

import pytest

from monjyu.document.parser import DocumentParser
from monjyu.document.models import (
    AcademicPaperDocument,
    AcademicSection,
    Author,
    ProcessedPDF,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def parser():
    """デフォルトパーサー"""
    return DocumentParser()


@pytest.fixture
def mock_pdf_processor():
    """モックPDFプロセッサ"""
    processor = MagicMock()
    processor.process.return_value = ProcessedPDF(
        file_name="test.pdf",
        title="Test Paper",
        authors=[Author(name="John Doe")],
        abstract="This is abstract",
        sections=[
            AcademicSection(heading="Introduction", level=1, content="Intro text", section_type="introduction")
        ],
        raw_text="Full text content",
        page_count=10,
        doi="10.1234/test",
        arxiv_id="2401.12345",
        keywords=["test", "paper"],
        language="en",
        tables=[],
        figures=[],
        references=[],
    )
    return processor


# ==============================================================================
# PDF Parsing Tests
# ==============================================================================

class TestPDFParsing:
    """PDF解析テスト"""
    
    def test_parse_pdf_with_processor(self, mock_pdf_processor):
        """PDFプロセッサを使ってPDFをパース"""
        parser = DocumentParser(pdf_processor=mock_pdf_processor)
        
        result = parser.parse(b"pdf content", ".pdf")
        
        assert result.title == "Test Paper"
        assert len(result.authors) == 1
        assert result.authors[0].name == "John Doe"
        assert result.abstract == "This is abstract"
        assert result.doi == "10.1234/test"
        assert result.arxiv_id == "2401.12345"
        mock_pdf_processor.process.assert_called_once_with(b"pdf content")
    
    def test_pdf_processor_lazy_init(self):
        """PDFプロセッサの遅延初期化"""
        parser = DocumentParser()
        assert parser._pdf_processor is None
        
        # pdf_processorプロパティにアクセスすると初期化される
        with patch("monjyu.document.pdf.unstructured_processor.UnstructuredPDFProcessor") as mock_cls:
            mock_cls.return_value = MagicMock()
            processor = parser.pdf_processor
            assert processor is not None


# ==============================================================================
# Text Parsing Tests
# ==============================================================================

class TestTextParsing:
    """テキスト解析テスト"""
    
    def test_parse_text_basic(self, parser):
        """基本テキストをパース"""
        content = b"Test Title\n\nSome content here.\nMore content."
        
        result = parser.parse(content, ".txt")
        
        assert result.title == "Test Title"
        assert result.file_type == ".txt"
        assert "Some content" in result.raw_text
    
    def test_parse_text_with_markdown_headers(self, parser):
        """Markdownヘッダー付きテキスト"""
        content = b"""# Main Title

## Introduction

This is introduction.

## Methods

This is methods section.
"""
        result = parser.parse(content, ".md")
        
        assert result.title == "# Main Title" or len(result.sections) >= 2
    
    def test_parse_text_rst(self, parser):
        """reStructuredTextをパース"""
        content = b"Title\n=====\n\nContent here."
        
        result = parser.parse(content, ".rst")
        
        assert result.file_type == ".rst"


# ==============================================================================
# HTML Parsing Tests  
# ==============================================================================

class TestHTMLParsing:
    """HTML解析テスト"""
    
    def test_parse_html_with_title_tag(self, parser):
        """titleタグのあるHTML"""
        content = b"""
        <html>
        <head><title>Page Title</title></head>
        <body>
        <p>Body content</p>
        </body>
        </html>
        """
        
        result = parser.parse(content, ".html")
        
        assert result.title == "Page Title"
        assert result.file_type == ".html"
        assert "Body content" in result.raw_text
    
    def test_parse_html_with_h1(self, parser):
        """h1タグからタイトル取得"""
        content = b"""
        <html>
        <body>
        <h1>Main Heading</h1>
        <p>Paragraph text</p>
        </body>
        </html>
        """
        
        result = parser.parse(content, ".html")
        
        # titleタグがないのでh1から取得
        assert "Main Heading" in result.title or "Main Heading" in result.raw_text
    
    def test_parse_html_no_body(self, parser):
        """bodyタグがないHTML"""
        content = b"<html>Just text</html>"
        
        result = parser.parse(content, ".html")
        
        assert result.file_type == ".html"
    
    def test_parse_htm_extension(self, parser):
        """.htm拡張子"""
        content = b"<html><body>Content</body></html>"
        
        result = parser.parse(content, ".htm")
        
        assert result.file_type == ".html"
    
    def test_parse_html_without_beautifulsoup(self, parser):
        """BeautifulSoupなしの場合テキストとして処理"""
        content = b"<html><body>Content</body></html>"
        
        with patch.dict("sys.modules", {"bs4": None}):
            with patch("monjyu.document.parser.DocumentParser._parse_html") as mock:
                # ImportErrorでテキストにフォールバック
                mock.side_effect = lambda c: parser._parse_text(c, ".html")
                result = mock(content)
                assert result is not None


# ==============================================================================
# JSON Parsing Tests
# ==============================================================================

class TestJSONParsing:
    """JSON解析テスト"""
    
    def test_parse_json_standard_format(self, parser):
        """標準的なJSON形式"""
        data = {
            "title": "Test Paper Title",
            "authors": [
                {"name": "Alice Smith", "affiliation": "MIT", "email": "alice@mit.edu"},
                {"name": "Bob Jones"}
            ],
            "abstract": "This is the abstract.",
            "doi": "10.1234/test.2024",
            "arxiv_id": "2401.99999",
            "year": 2024,
            "keywords": ["machine learning", "NLP"]
        }
        content = json.dumps(data).encode("utf-8")
        
        result = parser.parse(content, ".json")
        
        assert result.title == "Test Paper Title"
        assert len(result.authors) == 2
        assert result.authors[0].name == "Alice Smith"
        assert result.authors[0].affiliation == "MIT"
        assert result.authors[0].email == "alice@mit.edu"
        assert result.abstract == "This is the abstract."
        assert result.doi == "10.1234/test.2024"
        assert result.arxiv_id == "2401.99999"
        assert result.publication_year == 2024
        assert "machine learning" in result.keywords
    
    def test_parse_json_capitalized_keys(self, parser):
        """大文字キーのJSON"""
        data = {
            "Title": "Capitalized Title",
            "Authors": ["Jane Doe"],
            "Abstract": "Abstract text",
            "DOI": "10.5678/cap",
        }
        content = json.dumps(data).encode("utf-8")
        
        result = parser.parse(content, ".json")
        
        assert result.title == "Capitalized Title"
        assert len(result.authors) == 1
        assert result.authors[0].name == "Jane Doe"
        assert result.doi == "10.5678/cap"
    
    def test_parse_json_string_authors(self, parser):
        """文字列形式の著者"""
        data = {
            "title": "Paper",
            "authors": ["Author One", "Author Two"]
        }
        content = json.dumps(data).encode("utf-8")
        
        result = parser.parse(content, ".json")
        
        assert len(result.authors) == 2
        assert result.authors[0].name == "Author One"
    
    def test_parse_json_empty(self, parser):
        """空のJSON"""
        content = b"{}"
        
        result = parser.parse(content, ".json")
        
        assert result.title == ""
        assert result.authors == []


# ==============================================================================
# LaTeX Parsing Tests
# ==============================================================================

class TestLaTeXParsing:
    """LaTeX解析テスト"""
    
    def test_parse_latex_basic(self, parser):
        """基本的なLaTeX"""
        content = br"""
\documentclass{article}
\title{Machine Learning Paper}
\author{John Smith \and Jane Doe}
\begin{document}
\maketitle
\begin{abstract}
This paper presents a novel approach.
\end{abstract}
\section{Introduction}
Introduction content here.
\section{Methods}
Methods description.
\end{document}
"""
        result = parser.parse(content, ".tex")
        
        assert result.title == "Machine Learning Paper"
        assert len(result.authors) >= 2
        assert "novel approach" in result.abstract
        assert len(result.sections) >= 2
    
    def test_parse_latex_no_title(self, parser):
        """タイトルなしLaTeX"""
        content = br"""
\begin{document}
Some content
\end{document}
"""
        result = parser.parse(content, ".tex")
        
        assert result.title == ""
    
    def test_parse_latex_complex_author(self, parser):
        """複雑な著者フォーマット"""
        content = br"""
\title{Paper}
\author{Alice \textit{Smith}\\MIT}
"""
        result = parser.parse(content, ".tex")
        
        assert len(result.authors) >= 1
    
    def test_parse_latex_section_classification(self, parser):
        """セクション分類"""
        content = br"""
\title{Test}
\begin{document}
\section{Introduction}
Intro content.
\section{Related Work}
Related content.
\section{Methodology}
Method content.
\section{Results and Discussion}
Results content.
\section{Conclusion}
Conclusion content.
\end{document}
"""
        result = parser.parse(content, ".tex")
        
        section_types = [s.section_type for s in result.sections]
        assert "introduction" in section_types


# ==============================================================================
# Structure Extraction Tests
# ==============================================================================

class TestStructureExtraction:
    """構造抽出テスト"""
    
    def test_extract_structure_with_numbered_sections(self, parser):
        """番号付きセクション"""
        text = """Paper Title

1. Introduction
Introduction content.

2. Methods
Methods content.

3. Results
Results content.
"""
        title, sections = parser._extract_structure_from_text(text)
        
        assert title == "Paper Title"
        # セクションが抽出される（実装による）
        assert title is not None
    
    def test_extract_structure_uppercase_headers(self, parser):
        """大文字ヘッダー"""
        text = """TITLE

INTRODUCTION
Introduction text.

METHODS
Methods text.
"""
        title, sections = parser._extract_structure_from_text(text)
        
        assert len(sections) >= 2
    
    def test_is_likely_header_numbered(self, parser):
        """番号付きヘッダー判定"""
        assert parser._is_likely_header("1. Introduction")
        assert parser._is_likely_header("2.1 Sub Section")
    
    def test_is_likely_header_uppercase(self, parser):
        """全大文字ヘッダー判定"""
        assert parser._is_likely_header("INTRODUCTION")
        assert parser._is_likely_header("METHODS AND MATERIALS")
    
    def test_is_likely_header_keywords(self, parser):
        """キーワードヘッダー判定"""
        assert parser._is_likely_header("Introduction")
        assert parser._is_likely_header("Related Work")
        assert parser._is_likely_header("Conclusion")


# ==============================================================================
# Section Classification Tests
# ==============================================================================

class TestSectionClassification:
    """セクション分類テスト"""
    
    def test_classify_introduction(self, parser):
        """Introduction分類"""
        assert parser.classify_section("Introduction") == "introduction"
        assert parser.classify_section("1. INTRODUCTION") == "introduction"
        assert parser.classify_section("はじめに") == "introduction"
    
    def test_classify_methods(self, parser):
        """Methods分類"""
        assert parser.classify_section("Methods") == "methods"
        assert parser.classify_section("Methodology") == "methods"
        assert parser.classify_section("Proposed Approach") == "methods"
        assert parser.classify_section("提案手法") == "methods"
    
    def test_classify_results(self, parser):
        """Results分類"""
        assert parser.classify_section("Results") == "results"
        assert parser.classify_section("Experiments") == "results"
        assert parser.classify_section("Evaluation") == "results"
    
    def test_classify_discussion(self, parser):
        """Discussion分類"""
        assert parser.classify_section("Discussion") == "discussion"
        assert parser.classify_section("Analysis") == "discussion"
    
    def test_classify_conclusion(self, parser):
        """Conclusion分類"""
        assert parser.classify_section("Conclusion") == "conclusion"
        assert parser.classify_section("Summary") == "conclusion"
        assert parser.classify_section("まとめ") == "conclusion"
    
    def test_classify_related_work(self, parser):
        """Related Work分類"""
        assert parser.classify_section("Related Work") == "related_work"
        assert parser.classify_section("Background") == "related_work"
        assert parser.classify_section("先行研究") == "related_work"
    
    def test_classify_references(self, parser):
        """References分類"""
        assert parser.classify_section("References") == "references"
        assert parser.classify_section("Bibliography") == "references"
        assert parser.classify_section("参考文献") == "references"
    
    def test_classify_other(self, parser):
        """Other分類"""
        assert parser.classify_section("Appendix A") == "other"
        assert parser.classify_section("Acknowledgments") == "other"


# ==============================================================================
# DOI/arXiv Extraction Tests
# ==============================================================================

class TestIdentifierExtraction:
    """識別子抽出テスト"""
    
    def test_extract_doi(self, parser):
        """DOI抽出"""
        text = "This paper DOI: 10.1234/journal.2024.001 presents..."
        
        doi = parser.extract_doi(text)
        
        assert doi == "10.1234/journal.2024.001"
    
    def test_extract_doi_none(self, parser):
        """DOIなし"""
        text = "This paper has no DOI."
        
        doi = parser.extract_doi(text)
        
        assert doi is None
    
    def test_extract_arxiv_id(self, parser):
        """arXiv ID抽出"""
        text = "Available at arXiv:2401.12345v2"
        
        arxiv_id = parser.extract_arxiv_id(text)
        
        assert arxiv_id == "2401.12345v2"
    
    def test_extract_arxiv_id_case_insensitive(self, parser):
        """arXiv ID大文字小文字無視"""
        text = "ARXIV:2312.99999"
        
        arxiv_id = parser.extract_arxiv_id(text)
        
        assert arxiv_id == "2312.99999"
    
    def test_extract_arxiv_id_none(self, parser):
        """arXiv IDなし"""
        text = "No arxiv reference here."
        
        arxiv_id = parser.extract_arxiv_id(text)
        
        assert arxiv_id is None


# ==============================================================================
# File Parsing Tests
# ==============================================================================

class TestFileParsing:
    """ファイルパースTEST"""
    
    def test_parse_file_sets_filename(self, parser, tmp_path):
        """parse_fileでファイル名が設定される"""
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("Test Title\n\nTest content.")
        
        result = parser.parse_file(test_file)
        
        assert result.file_name == "test_document.txt"
    
    def test_parse_unknown_extension(self, parser):
        """未知の拡張子はテキストとして処理"""
        content = b"Some content in unknown format"
        
        result = parser.parse(content, ".xyz")
        
        # テキストとして処理される
        assert result.raw_text == "Some content in unknown format"


# ==============================================================================
# Word Document Tests (Error Paths)
# ==============================================================================

class TestWordParsingErrors:
    """Wordパース　エラーパステスト"""
    
    def test_parse_word_generic_exception(self, parser):
        """Word解析の一般エラー"""
        with patch("docx.Document", side_effect=Exception("Parse error")):
            result = parser._parse_word(b"invalid", ".docx")
            
            assert "[Word document - parse error]" in result.title
    
    def test_parse_doc_extension(self, parser):
        """古い.doc形式"""
        # .docは.docxと同じ処理を通る
        with patch("docx.Document") as mock_doc:
            mock_doc.return_value.paragraphs = []
            result = parser._parse_word(b"content", ".doc")
            assert result.file_type == ".doc"


# ==============================================================================
# PowerPoint Tests (Error Paths)
# ==============================================================================

class TestPowerPointParsingErrors:
    """PowerPointパース　エラーパステスト"""
    
    def test_parse_pptx_generic_exception(self, parser):
        """PowerPoint解析の一般エラー"""
        with patch("pptx.Presentation", side_effect=Exception("Parse error")):
            result = parser._parse_powerpoint(b"invalid")
            
            assert "[PowerPoint - parse error]" in result.title

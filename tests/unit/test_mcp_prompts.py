"""Tests for MCP Server Prompts.

FEAT-009: MCP Server Prompts
"""

import pytest


# === Prompt Tests ===

class TestLiteratureReviewPrompt:
    """Tests for literature_review prompt."""
    
    @pytest.mark.asyncio
    async def test_basic_prompt(self):
        """Test basic literature review prompt generation."""
        from monjyu.mcp_server.server import literature_review
        
        result = await literature_review(topic="machine learning")
        
        assert "machine learning" in result
        assert "monjyu_search" in result
        assert "Literature Review Structure" in result
        assert "Introduction" in result
        assert "Conclusions" in result
    
    @pytest.mark.asyncio
    async def test_with_focus_area(self):
        """Test prompt with focus area."""
        from monjyu.mcp_server.server import literature_review
        
        result = await literature_review(
            topic="deep learning",
            focus_area="transformer architectures",
        )
        
        assert "deep learning" in result
        assert "transformer architectures" in result
    
    @pytest.mark.asyncio
    async def test_custom_num_papers(self):
        """Test prompt with custom number of papers."""
        from monjyu.mcp_server.server import literature_review
        
        result = await literature_review(topic="NLP", num_papers=20)
        
        assert "20" in result


class TestPaperSummaryPrompt:
    """Tests for paper_summary prompt."""
    
    @pytest.mark.asyncio
    async def test_with_document_id(self):
        """Test prompt with document ID."""
        from monjyu.mcp_server.server import paper_summary
        
        result = await paper_summary(document_id="doc123")
        
        assert "doc123" in result
        assert "monjyu_get_document" in result
        assert "Summary Structure" in result
    
    @pytest.mark.asyncio
    async def test_with_title(self):
        """Test prompt with title search."""
        from monjyu.mcp_server.server import paper_summary
        
        result = await paper_summary(title="Attention Is All You Need")
        
        assert "Attention Is All You Need" in result
        assert "monjyu_search" in result
    
    @pytest.mark.asyncio
    async def test_without_identifier(self):
        """Test prompt without document identifier."""
        from monjyu.mcp_server.server import paper_summary
        
        result = await paper_summary()
        
        assert "Ask the user" in result


class TestComparePapersPrompt:
    """Tests for compare_papers prompt."""
    
    @pytest.mark.asyncio
    async def test_with_paper_ids(self):
        """Test prompt with specific paper IDs."""
        from monjyu.mcp_server.server import compare_papers
        
        result = await compare_papers(paper_ids="doc1, doc2, doc3")
        
        assert "doc1" in result
        assert "doc2" in result
        assert "doc3" in result
        assert "Comparison" in result
    
    @pytest.mark.asyncio
    async def test_with_topic(self):
        """Test prompt with topic search."""
        from monjyu.mcp_server.server import compare_papers
        
        result = await compare_papers(topic="graph neural networks")
        
        assert "graph neural networks" in result
        assert "monjyu_search" in result
    
    @pytest.mark.asyncio
    async def test_custom_aspects(self):
        """Test prompt with custom comparison aspects."""
        from monjyu.mcp_server.server import compare_papers
        
        result = await compare_papers(
            topic="RAG",
            comparison_aspects="architecture,performance,scalability",
        )
        
        assert "Architecture" in result
        assert "Performance" in result
        assert "Scalability" in result


class TestResearchQuestionPrompt:
    """Tests for research_question prompt."""
    
    @pytest.mark.asyncio
    async def test_basic_prompt(self):
        """Test basic research question prompt."""
        from monjyu.mcp_server.server import research_question
        
        result = await research_question(domain="natural language processing")
        
        assert "natural language processing" in result
        assert "Gap Identification" in result
        assert "Research Question Generation" in result
    
    @pytest.mark.asyncio
    async def test_with_interest(self):
        """Test prompt with specific interest."""
        from monjyu.mcp_server.server import research_question
        
        result = await research_question(
            domain="computer vision",
            current_interest="medical imaging",
        )
        
        assert "computer vision" in result
        assert "medical imaging" in result
    
    @pytest.mark.asyncio
    async def test_with_methodology(self):
        """Test prompt with methodology preference."""
        from monjyu.mcp_server.server import research_question
        
        result = await research_question(
            domain="AI",
            methodology_preference="empirical studies",
        )
        
        assert "empirical studies" in result


class TestCitationAnalysisPrompt:
    """Tests for citation_analysis prompt."""
    
    @pytest.mark.asyncio
    async def test_with_document_id(self):
        """Test citation analysis for specific document."""
        from monjyu.mcp_server.server import citation_analysis
        
        result = await citation_analysis(document_id="paper123")
        
        assert "paper123" in result
        assert "monjyu_get_document" in result
        assert "monjyu_citation_chain" in result
    
    @pytest.mark.asyncio
    async def test_full_network_analysis(self):
        """Test full network analysis."""
        from monjyu.mcp_server.server import citation_analysis
        
        result = await citation_analysis()
        
        assert "citation-network" in result
        assert "Network Position" in result
    
    @pytest.mark.asyncio
    async def test_influence_analysis_type(self):
        """Test influence analysis type."""
        from monjyu.mcp_server.server import citation_analysis
        
        result = await citation_analysis(
            document_id="doc1",
            analysis_type="influence",
        )
        
        assert "Influence Metrics" in result
        assert "Impact Assessment" in result
    
    @pytest.mark.asyncio
    async def test_trends_analysis_type(self):
        """Test trends analysis type."""
        from monjyu.mcp_server.server import citation_analysis
        
        result = await citation_analysis(analysis_type="trends")
        
        assert "Temporal Patterns" in result
        assert "Research Front Analysis" in result


# === Prompt Structure Tests ===

class TestPromptStructure:
    """Tests for prompt structure and quality."""
    
    @pytest.mark.asyncio
    async def test_all_prompts_return_strings(self):
        """Test that all prompts return strings."""
        from monjyu.mcp_server.server import (
            literature_review,
            paper_summary,
            compare_papers,
            research_question,
            citation_analysis,
        )
        
        prompts = [
            literature_review(topic="test"),
            paper_summary(document_id="test"),
            compare_papers(paper_ids="test"),
            research_question(domain="test"),
            citation_analysis(document_id="test"),
        ]
        
        for prompt in prompts:
            result = await prompt
            assert isinstance(result, str)
            assert len(result) > 100  # Prompts should be substantial
    
    @pytest.mark.asyncio
    async def test_prompts_contain_tool_references(self):
        """Test that prompts reference MCP tools."""
        from monjyu.mcp_server.server import (
            literature_review,
            paper_summary,
            compare_papers,
            research_question,
            citation_analysis,
        )
        
        # Literature review
        lit_result = await literature_review(topic="test")
        assert "monjyu_search" in lit_result
        
        # Paper summary
        sum_result = await paper_summary(document_id="test")
        assert "monjyu_get_document" in sum_result
        
        # Compare papers
        cmp_result = await compare_papers(paper_ids="test")
        assert "monjyu_get_document" in cmp_result
        
        # Research question
        rq_result = await research_question(domain="test")
        assert "monjyu_search" in rq_result
        
        # Citation analysis
        cit_result = await citation_analysis(document_id="test")
        assert "monjyu_citation_chain" in cit_result

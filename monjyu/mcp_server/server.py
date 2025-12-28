# MONJYU MCP Server - FastMCP Implementation
"""
FEAT-009: MCP Server (Model Context Protocol Server)
FastMCPを使用した MCPサーバー実装

Usage:
    monjyu-mcp                    # stdioモードで起動
    python -m monjyu.mcp_server   # 直接起動

Configuration:
    環境変数 MONJYU_CONFIG でmonjyu.yamlのパスを指定可能

MCP Tools:
    - monjyu_search: 学術論文検索 (vector/lazy/auto)
    - monjyu_get_document: ドキュメント詳細取得
    - monjyu_list_documents: ドキュメント一覧
    - monjyu_citation_chain: 引用チェーン取得
    - monjyu_find_related: 関連論文検索
    - monjyu_status: インデックスステータス取得
    - monjyu_get_metrics: 引用メトリクス取得

MCP Resources:
    - monjyu://index/status: インデックスステータス
    - monjyu://documents: ドキュメント一覧
    - monjyu://documents/{id}: 個別ドキュメント
    - monjyu://documents/{id}/content: ドキュメント全文
    - monjyu://citation-network: 引用ネットワーク概要

MCP Prompts:
    - literature_review: 文献レビュー生成
    - paper_summary: 論文要約
    - compare_papers: 論文比較
    - research_question: 研究質問の探索
    - citation_analysis: 引用分析
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

# ロギング設定（stdoutではなくstderrに出力）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("monjyu-mcp")

# === FastMCP サーバー初期化 ===

mcp = FastMCP(
    "monjyu",
    instructions="MONJYU - Progressive GraphRAG for Academic Papers. "
    "Provides semantic search, citation analysis, and document retrieval for academic papers.",
)

# === グローバル MONJYU インスタンス（遅延初期化） ===

_monjyu = None


def get_monjyu():
    """MONJYUインスタンスを取得（遅延初期化）
    
    Returns:
        MONJYU: 初期化済みインスタンス
    """
    global _monjyu
    
    if _monjyu is None:
        from monjyu.api import MONJYU
        
        # 環境変数から設定ファイルパスを取得
        config_path_str = os.environ.get("MONJYU_CONFIG")
        
        if config_path_str:
            config_path = Path(config_path_str)
            if config_path.exists():
                _monjyu = MONJYU(config_path)
                logger.info(f"MONJYU initialized with config: {config_path}")
                return _monjyu
        
        # デフォルトパスを探索
        default_paths = [
            Path("./monjyu.yaml"),
            Path("./monjyu.yml"),
            Path("./config/monjyu.yaml"),
            Path.home() / ".config" / "monjyu" / "monjyu.yaml",
        ]
        
        for path in default_paths:
            if path.exists():
                _monjyu = MONJYU(path)
                logger.info(f"MONJYU initialized with config: {path}")
                return _monjyu
        
        # デフォルト設定で初期化
        _monjyu = MONJYU()
        logger.info("MONJYU initialized with default config")
    
    return _monjyu


def reset_monjyu():
    """MONJYUインスタンスをリセット（テスト用）"""
    global _monjyu
    _monjyu = None


def set_monjyu(monjyu):
    """MONJYUインスタンスを設定（テスト用）"""
    global _monjyu
    _monjyu = monjyu


# === ヘルパー関数 ===

def json_format(data: dict) -> str:
    """JSON形式の文字列を生成"""
    return json.dumps(data, ensure_ascii=False, indent=2)


def error_format(message: str) -> str:
    """エラーメッセージをJSON形式で返す"""
    return json_format({"error": message})


# === MCP Tools (FastMCP デコレータ) ===

# === MCP Resources (FastMCP デコレータ) ===

@mcp.resource("monjyu://index/status")
async def resource_index_status() -> str:
    """Get current index status as a resource.
    
    Returns:
        JSON with index status information
    """
    try:
        monjyu = get_monjyu()
        status = monjyu.get_status()
        
        return json_format({
            "type": "index_status",
            "document_count": status.document_count,
            "text_unit_count": status.text_unit_count,
            "entity_count": getattr(status, "entity_count", 0),
            "relationship_count": getattr(status, "relationship_count", 0),
            "community_count": getattr(status, "community_count", 0),
            "index_levels": status.index_levels,
            "is_ready": status.is_ready,
            "last_updated": status.last_updated.isoformat() if status.last_updated else None,
        })
    except Exception as e:
        logger.exception("Failed to get index status resource")
        return error_format(f"Failed to get index status: {str(e)}")


@mcp.resource("monjyu://documents")
async def resource_documents_list() -> str:
    """Get list of all documents as a resource.
    
    Returns:
        JSON with document list
    """
    try:
        monjyu = get_monjyu()
        documents = monjyu.list_documents(limit=1000)
        
        return json_format({
            "type": "document_list",
            "count": len(documents),
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "authors": d.authors[:3] if d.authors else [],
                    "year": d.year,
                    "uri": f"monjyu://documents/{d.id}",
                }
                for d in documents
            ]
        })
    except Exception as e:
        logger.exception("Failed to get documents resource")
        return error_format(f"Failed to get documents: {str(e)}")


@mcp.resource("monjyu://documents/{document_id}")
async def resource_document_detail(document_id: str) -> str:
    """Get detailed document information as a resource.
    
    Args:
        document_id: Document ID
    
    Returns:
        JSON with document details
    """
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        return json_format({
            "type": "document",
            "id": doc.id,
            "title": doc.title,
            "authors": doc.authors,
            "year": doc.year,
            "doi": doc.doi,
            "abstract": doc.abstract,
            "chunk_count": doc.chunk_count,
            "citation_count": doc.citation_count,
            "reference_count": doc.reference_count,
            "influence_score": round(doc.influence_score, 4) if doc.influence_score else None,
            "content_uri": f"monjyu://documents/{doc.id}/content",
            "citations_uri": f"monjyu://documents/{doc.id}/citations",
        })
    except Exception as e:
        logger.exception("Failed to get document resource")
        return error_format(f"Failed to get document: {str(e)}")


@mcp.resource("monjyu://documents/{document_id}/content")
async def resource_document_content(document_id: str) -> str:
    """Get full document content (text units) as a resource.
    
    Args:
        document_id: Document ID
    
    Returns:
        JSON with document text content
    """
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        # テキストユニットを取得
        text_units = monjyu.get_text_units(document_id)
        
        return json_format({
            "type": "document_content",
            "document_id": document_id,
            "title": doc.title,
            "text_unit_count": len(text_units),
            "text_units": [
                {
                    "id": tu.id,
                    "text": tu.text,
                    "chunk_index": tu.chunk_index,
                }
                for tu in text_units
            ],
            "full_text": "\n\n".join([tu.text for tu in text_units]),
        })
    except Exception as e:
        logger.exception("Failed to get document content resource")
        return error_format(f"Failed to get document content: {str(e)}")


@mcp.resource("monjyu://documents/{document_id}/citations")
async def resource_document_citations(document_id: str) -> str:
    """Get document citation network as a resource.
    
    Args:
        document_id: Document ID
    
    Returns:
        JSON with citation information
    """
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        citation_manager = monjyu.get_citation_network()
        
        references = []
        cited_by = []
        
        if citation_manager:
            try:
                refs = citation_manager.get_references(document_id)
                for ref in refs[:50]:
                    ref_doc = monjyu.get_document(ref.target_id)
                    references.append({
                        "id": ref.target_id,
                        "title": ref_doc.title if ref_doc else None,
                        "year": ref_doc.year if ref_doc else None,
                        "uri": f"monjyu://documents/{ref.target_id}",
                    })
                
                citations = citation_manager.get_citations(document_id)
                for cite in citations[:50]:
                    cite_doc = monjyu.get_document(cite.source_id)
                    cited_by.append({
                        "id": cite.source_id,
                        "title": cite_doc.title if cite_doc else None,
                        "year": cite_doc.year if cite_doc else None,
                        "uri": f"monjyu://documents/{cite.source_id}",
                    })
            except Exception as e:
                logger.warning(f"Citation lookup failed: {e}")
        
        return json_format({
            "type": "document_citations",
            "document_id": document_id,
            "title": doc.title,
            "references": references,
            "references_count": len(references),
            "cited_by": cited_by,
            "cited_by_count": len(cited_by),
        })
    except Exception as e:
        logger.exception("Failed to get citations resource")
        return error_format(f"Failed to get citations: {str(e)}")


@mcp.resource("monjyu://citation-network")
async def resource_citation_network() -> str:
    """Get citation network overview as a resource.
    
    Returns:
        JSON with network statistics
    """
    try:
        monjyu = get_monjyu()
        citation_manager = monjyu.get_citation_network()
        
        if citation_manager is None:
            return json_format({
                "type": "citation_network",
                "available": False,
                "message": "Citation network not built",
            })
        
        # ネットワーク統計を取得
        stats = citation_manager.get_statistics()
        
        # 最も引用されているドキュメント
        top_cited = []
        try:
            top_docs = citation_manager.get_most_cited(limit=10)
            for doc_id, count in top_docs:
                doc = monjyu.get_document(doc_id)
                top_cited.append({
                    "id": doc_id,
                    "title": doc.title if doc else None,
                    "citation_count": count,
                    "uri": f"monjyu://documents/{doc_id}",
                })
        except Exception:
            pass
        
        return json_format({
            "type": "citation_network",
            "available": True,
            "statistics": {
                "node_count": stats.get("node_count", 0),
                "edge_count": stats.get("edge_count", 0),
                "density": round(stats.get("density", 0), 6),
                "avg_citations": round(stats.get("avg_citations", 0), 2),
            },
            "top_cited_documents": top_cited,
        })
    except Exception as e:
        logger.exception("Failed to get citation network resource")
        return error_format(f"Failed to get citation network: {str(e)}")


# === MCP Prompts (FastMCP デコレータ) ===

@mcp.prompt()
async def literature_review(
    topic: str,
    num_papers: int = 10,
    focus_area: str = "",
) -> str:
    """Generate a literature review prompt for academic research.
    
    Creates a structured prompt to help write a comprehensive literature
    review on a given topic using papers from the MONJYU index.
    
    Args:
        topic: The main research topic for the literature review
        num_papers: Number of relevant papers to include (default: 10)
        focus_area: Optional specific focus area within the topic
    
    Returns:
        A structured prompt for generating a literature review
    """
    focus_text = f" with a focus on {focus_area}" if focus_area else ""
    
    return f"""You are an academic researcher writing a literature review on "{topic}"{focus_text}.

## Instructions

1. Use the `monjyu_search` tool to find the top {num_papers} most relevant papers on this topic.
2. For each paper found, use `monjyu_get_document` to get detailed information.
3. Use `monjyu_citation_chain` to understand the citation relationships between papers.

## Literature Review Structure

Please write a comprehensive literature review following this structure:

### 1. Introduction
- Define the research topic and its importance
- State the scope and objectives of the review

### 2. Background and Context
- Provide historical context
- Define key concepts and terminology

### 3. Main Body (Thematic Analysis)
- Group papers by themes or methodological approaches
- Discuss key findings from each paper
- Compare and contrast different approaches
- Identify areas of consensus and controversy

### 4. Critical Analysis
- Evaluate the strengths and limitations of existing research
- Identify gaps in the current literature

### 5. Conclusions and Future Directions
- Summarize the main findings
- Suggest directions for future research

## Citation Format
- Use [Author, Year] format for in-text citations
- Include DOI when available

## Search Query
Topic: {topic}
Focus Area: {focus_area if focus_area else "General"}
Number of Papers: {num_papers}

Begin by searching for relevant papers using `monjyu_search`."""


@mcp.prompt()
async def paper_summary(
    document_id: str = "",
    title: str = "",
) -> str:
    """Generate a prompt for summarizing an academic paper.
    
    Creates a structured prompt to produce a comprehensive summary
    of a specific paper including its methodology, findings, and impact.
    
    Args:
        document_id: The ID of the document to summarize (optional)
        title: The title of the paper to search for (optional)
    
    Returns:
        A structured prompt for paper summarization
    """
    if document_id:
        lookup_instruction = f"Use `monjyu_get_document` with document_id='{document_id}' to retrieve the paper details."
    elif title:
        lookup_instruction = f'Use `monjyu_search` to find the paper titled "{title}", then use `monjyu_get_document` for details.'
    else:
        lookup_instruction = "Ask the user to provide either a document_id or a paper title to summarize."
    
    return f"""You are an academic research assistant tasked with summarizing a research paper.

## Instructions

1. {lookup_instruction}
2. Use `monjyu_citation_chain` to understand the paper's academic context.
3. Use `monjyu_get_metrics` to assess the paper's impact.

## Summary Structure

Please provide a comprehensive summary following this structure:

### Paper Information
- Title, Authors, Year, DOI
- Publication venue (if available)

### Abstract Summary
- One paragraph capturing the essence of the paper

### Key Contributions
- List 3-5 main contributions of the paper

### Methodology
- Research approach and methods used
- Data sources and analysis techniques

### Main Findings
- Key results and discoveries
- Statistical significance (if applicable)

### Limitations
- Acknowledged limitations by authors
- Additional limitations you identify

### Impact and Citations
- Citation count and influence score
- Key papers that cite this work
- Key papers this work builds upon

### Relevance Assessment
- Who would benefit from reading this paper?
- How does it fit into the broader research landscape?

Begin by retrieving the paper information."""


@mcp.prompt()
async def compare_papers(
    paper_ids: str = "",
    topic: str = "",
    comparison_aspects: str = "methodology,findings,limitations",
) -> str:
    """Generate a prompt for comparing multiple academic papers.
    
    Creates a structured prompt for systematic comparison of papers
    on specific aspects like methodology, findings, and conclusions.
    
    Args:
        paper_ids: Comma-separated document IDs to compare (optional)
        topic: Topic to search for papers to compare (optional)
        comparison_aspects: Aspects to compare (comma-separated)
    
    Returns:
        A structured prompt for paper comparison
    """
    aspects = [a.strip() for a in comparison_aspects.split(",")]
    aspects_list = "\n".join([f"- {aspect.title()}" for aspect in aspects])
    
    if paper_ids:
        ids = [id.strip() for id in paper_ids.split(",")]
        retrieval_instruction = f"""Retrieve the following papers using `monjyu_get_document`:
{chr(10).join([f"- Document ID: {id}" for id in ids])}"""
    elif topic:
        retrieval_instruction = f'Use `monjyu_search` with query="{topic}" to find relevant papers, then retrieve details with `monjyu_get_document`.'
    else:
        retrieval_instruction = "Ask the user to provide either document IDs or a topic to compare papers."
    
    return f"""You are an academic researcher conducting a systematic comparison of research papers.

## Instructions

1. {retrieval_instruction}
2. For each paper, use `monjyu_citation_chain` to understand citation relationships.
3. Use `monjyu_find_related` to identify conceptually similar papers.

## Comparison Aspects

Compare the papers on the following aspects:
{aspects_list}

## Comparison Structure

### 1. Overview Table
Create a summary table with:
| Paper | Year | Authors | Key Focus |
|-------|------|---------|-----------|

### 2. Detailed Comparison

For each aspect, provide:
- How each paper approaches it
- Similarities and differences
- Strengths and weaknesses

### 3. Methodology Comparison
- Research design
- Data collection methods
- Analysis techniques
- Sample sizes/datasets

### 4. Findings Comparison
- Main results from each paper
- Agreements across papers
- Contradictions or debates

### 5. Synthesis
- What do the papers collectively tell us?
- Which paper has the strongest evidence?
- Gaps not addressed by any paper

### 6. Recommendations
- Which paper to read for what purpose
- Suggested order of reading

Begin by retrieving the paper information."""


@mcp.prompt()
async def research_question(
    domain: str,
    current_interest: str = "",
    methodology_preference: str = "",
) -> str:
    """Generate a prompt for exploring research questions in a domain.
    
    Creates a structured prompt to help identify promising research
    questions based on gaps in the existing literature.
    
    Args:
        domain: The research domain to explore
        current_interest: Specific areas of interest (optional)
        methodology_preference: Preferred research methodology (optional)
    
    Returns:
        A structured prompt for research question exploration
    """
    interest_text = f"\nSpecific Interest: {current_interest}" if current_interest else ""
    method_text = f"\nMethodology Preference: {methodology_preference}" if methodology_preference else ""
    
    return f"""You are a research advisor helping to identify promising research questions in {domain}.

## Context
Domain: {domain}{interest_text}{method_text}

## Instructions

1. Use `monjyu_search` with query="{domain}" to find recent papers in this domain.
2. Use `monjyu_citation_chain` to identify influential papers and emerging trends.
3. Use `monjyu_find_related` to explore connected research areas.
4. Use the `monjyu://citation-network` resource to understand the research landscape.

## Analysis Process

### 1. Domain Mapping
- What are the main research themes in this domain?
- Who are the key researchers and groups?
- What methodologies are commonly used?

### 2. Trend Analysis
- What topics are gaining attention (highly cited recent papers)?
- What topics are declining in interest?
- What new methodologies are emerging?

### 3. Gap Identification
- What questions remain unanswered?
- What contradictions exist between studies?
- What assumptions have not been challenged?
- What populations/contexts are understudied?

### 4. Research Question Generation
For each identified gap, propose:
- A specific, answerable research question
- Why this question matters
- Potential methodology to address it
- Expected challenges

### 5. Prioritization Matrix
| Research Question | Novelty | Feasibility | Impact | Priority |
|-------------------|---------|-------------|--------|----------|

### 6. Recommended Next Steps
- Top 3 research questions to pursue
- Papers to read first
- Potential collaborators or experts to consult

Begin by searching for papers in the domain."""


@mcp.prompt()
async def citation_analysis(
    document_id: str = "",
    analysis_type: str = "full",
) -> str:
    """Generate a prompt for analyzing citation patterns.
    
    Creates a structured prompt for understanding the citation network
    and academic impact of a paper or research area.
    
    Args:
        document_id: Document ID to analyze (optional, analyzes full network if empty)
        analysis_type: Type of analysis - 'full', 'influence', 'trends' (default: full)
    
    Returns:
        A structured prompt for citation analysis
    """
    if document_id:
        target_instruction = f"""Focus on document ID: {document_id}
1. Use `monjyu_get_document` to retrieve the paper details.
2. Use `monjyu_citation_chain` to get its citation network.
3. Use `monjyu_get_metrics` for citation metrics."""
    else:
        target_instruction = """Analyze the entire citation network:
1. Use `monjyu://citation-network` resource to get network overview.
2. Use `monjyu_status` to understand the index scope.
3. Use `monjyu_search` to explore specific areas of interest."""
    
    analysis_sections = {
        "full": """### Citation Count Analysis
- Total citations received
- Citation growth over time (if temporal data available)
- Self-citation rate

### Network Position
- PageRank score interpretation
- Hub vs Authority scores
- Centrality in the network

### Citation Context
- What aspects are being cited?
- How are later papers building on this work?
- Are there critical citations?

### Co-citation Analysis
- Papers frequently cited together
- Research clusters identified

### Bibliographic Coupling
- Papers with similar reference lists
- Methodological lineages""",
        
        "influence": """### Influence Metrics
- Citation count trajectory
- Influence score breakdown
- Network centrality measures

### Impact Assessment
- Direct impact (first-order citations)
- Indirect impact (citation cascades)
- Field-specific vs cross-field impact

### Comparative Influence
- Comparison with similar papers
- Percentile ranking in field""",
        
        "trends": """### Temporal Patterns
- Publication trends over time
- Citation velocity changes
- Emerging vs declining topics

### Research Front Analysis
- Current hot topics
- Foundational vs recent papers
- Knowledge evolution paths"""
    }
    
    analysis_content = analysis_sections.get(analysis_type, analysis_sections["full"])
    
    return f"""You are a bibliometrics expert conducting citation analysis.

## Target
{target_instruction}

## Analysis Type: {analysis_type.title()}

## Analysis Structure

### 1. Overview
- Scope of analysis
- Key metrics summary

### 2. Detailed Analysis
{analysis_content}

### 3. Visualization Suggestions
- Network graph recommendations
- Timeline visualizations
- Cluster diagrams

### 4. Insights and Interpretation
- What do the citation patterns reveal?
- Anomalies or interesting patterns
- Potential biases in citations

### 5. Actionable Recommendations
- Papers to read based on analysis
- Research directions suggested by patterns
- Networking opportunities (key citing authors)

Begin by retrieving the relevant data."""


# === MCP Tools ===

@mcp.tool()
async def monjyu_search(
    query: str,
    mode: Literal["vector", "lazy", "auto"] = "lazy",
    top_k: int = 10,
) -> str:
    """Search academic papers in MONJYU index.
    
    Returns relevant papers with citations and an AI-generated answer.
    Use 'lazy' mode for complex questions requiring deep reasoning,
    'vector' mode for simple keyword searches.
    
    Args:
        query: Search query in natural language
        mode: Search mode - 'vector' (fast semantic search), 
              'lazy' (Progressive GraphRAG with deep reasoning), 
              'auto' (automatically select based on query)
        top_k: Number of results to return (1-50)
    
    Returns:
        JSON with answer, citations, and search metadata
    """
    if not query:
        return error_format("Query is required")
    
    top_k = max(1, min(top_k, 50))  # Clamp to 1-50
    
    try:
        from monjyu.api import SearchMode
        
        monjyu = get_monjyu()
        search_mode = SearchMode(mode)
        result = monjyu.search(query, mode=search_mode, top_k=top_k)
        
        response = {
            "query": result.query,
            "answer": result.answer,
            "citations": [
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "text": c.text[:200] if c.text else None,
                    "relevance_score": c.relevance_score,
                }
                for c in result.citations
            ],
            "search_info": {
                "mode": result.search_mode.value,
                "level": result.search_level,
                "time_ms": round(result.total_time_ms, 1),
                "llm_calls": result.llm_calls,
                "citation_count": result.citation_count,
            }
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Search failed")
        return error_format(f"Search failed: {str(e)}")


@mcp.tool()
async def monjyu_get_document(document_id: str) -> str:
    """Get detailed information about a specific document.
    
    Returns title, authors, abstract, and citation metrics.
    
    Args:
        document_id: Document ID to retrieve
    
    Returns:
        JSON with document details and metrics
    """
    if not document_id:
        return error_format("Document ID is required")
    
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        response = {
            "id": doc.id,
            "title": doc.title,
            "authors": doc.authors,
            "year": doc.year,
            "doi": doc.doi,
            "abstract": doc.abstract,
            "index_stats": {
                "chunk_count": doc.chunk_count,
            },
            "citation_metrics": {
                "citation_count": doc.citation_count,
                "reference_count": doc.reference_count,
                "influence_score": round(doc.influence_score, 4),
            }
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Failed to get document")
        return error_format(f"Failed to get document: {str(e)}")


@mcp.tool()
async def monjyu_list_documents(
    limit: int = 20,
    offset: int = 0,
) -> str:
    """List all documents in the MONJYU index with basic metadata.
    
    Supports pagination with limit and offset parameters.
    
    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Offset for pagination
    
    Returns:
        JSON with list of documents
    """
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    
    try:
        monjyu = get_monjyu()
        documents = monjyu.list_documents(limit=limit + offset)
        
        # オフセット適用
        if offset > 0:
            documents = documents[offset:]
        
        documents = documents[:limit]
        
        response = {
            "count": len(documents),
            "offset": offset,
            "limit": limit,
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "authors": d.authors[:3] if d.authors else [],
                    "year": d.year,
                    "chunk_count": d.chunk_count,
                }
                for d in documents
            ]
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Failed to list documents")
        return error_format(f"Failed to list documents: {str(e)}")


@mcp.tool()
async def monjyu_citation_chain(
    document_id: str,
    depth: int = 1,
) -> str:
    """Get the citation chain for a document.
    
    Shows what papers it cites (references) and what papers cite it (citations).
    Useful for understanding the academic context and influence of a paper.
    
    Args:
        document_id: Document ID to get citation chain for
        depth: Depth of citation chain to retrieve (1-3)
    
    Returns:
        JSON with references and citations
    """
    if not document_id:
        return error_format("Document ID is required")
    
    depth = max(1, min(depth, 3))
    
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        # 引用ネットワークを取得
        citation_manager = monjyu.get_citation_network()
        
        cites = []
        cited_by = []
        
        if citation_manager:
            try:
                # このドキュメントが引用している文献
                refs = citation_manager.get_references(document_id)
                for ref in refs[:10]:
                    ref_doc = monjyu.get_document(ref.target_id)
                    cites.append({
                        "id": ref.target_id,
                        "title": ref_doc.title if ref_doc else None,
                        "year": ref_doc.year if ref_doc else None,
                    })
                
                # このドキュメントを引用している文献
                citations = citation_manager.get_citations(document_id)
                for cite in citations[:10]:
                    cite_doc = monjyu.get_document(cite.source_id)
                    cited_by.append({
                        "id": cite.source_id,
                        "title": cite_doc.title if cite_doc else None,
                        "year": cite_doc.year if cite_doc else None,
                    })
            except Exception as e:
                logger.warning(f"Citation network lookup failed: {e}")
        
        response = {
            "document": {
                "id": document_id,
                "title": doc.title,
            },
            "depth": depth,
            "references": cites,
            "references_count": len(cites),
            "cited_by": cited_by,
            "cited_by_count": len(cited_by),
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Failed to get citation chain")
        return error_format(f"Failed to get citation chain: {str(e)}")


@mcp.tool()
async def monjyu_find_related(
    document_id: str,
    top_k: int = 10,
) -> str:
    """Find papers related to a given document.
    
    Based on citation patterns (co-citation, bibliographic coupling) 
    and content similarity. Returns papers that are thematically 
    or academically connected.
    
    Args:
        document_id: Document ID to find related papers for
        top_k: Number of related papers to return (1-50)
    
    Returns:
        JSON with related papers and relationship scores
    """
    if not document_id:
        return error_format("Document ID is required")
    
    top_k = max(1, min(top_k, 50))
    
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        # 引用ネットワークから関連論文を取得
        citation_manager = monjyu.get_citation_network()
        
        related_papers = []
        
        if citation_manager:
            try:
                related = citation_manager.find_co_citation_papers(document_id, top_k)
                for doc_id, score in related:
                    related_doc = monjyu.get_document(doc_id)
                    related_papers.append({
                        "id": doc_id,
                        "title": related_doc.title if related_doc else None,
                        "year": related_doc.year if related_doc else None,
                        "relationship": "co-citation",
                        "score": round(score, 4),
                    })
            except Exception as e:
                logger.warning(f"Related papers lookup failed: {e}")
        
        response = {
            "source_document": {
                "id": document_id,
                "title": doc.title,
            },
            "related_papers": related_papers,
            "count": len(related_papers),
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Failed to find related papers")
        return error_format(f"Failed to find related papers: {str(e)}")


@mcp.tool()
async def monjyu_status() -> str:
    """Get the current status of the MONJYU index.
    
    Returns document count, index levels built, and statistics.
    Useful for checking if the index is ready for queries.
    
    Returns:
        JSON with index status and statistics
    """
    try:
        monjyu = get_monjyu()
        status = monjyu.get_status()
        
        response = {
            "index_status": status.index_status.value,
            "is_ready": status.is_ready,
            "levels_built": [l.value for l in status.index_levels_built],
            "statistics": {
                "documents": status.document_count,
                "text_units": status.text_unit_count,
                "noun_phrases": status.noun_phrase_count,
                "communities": status.community_count,
                "citation_edges": status.citation_edge_count,
            },
            "last_error": status.last_error,
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Failed to get status")
        return error_format(f"Failed to get status: {str(e)}")


@mcp.tool()
async def monjyu_get_metrics(document_id: str) -> str:
    """Get citation metrics for a specific document.
    
    Returns citation count, PageRank, hub/authority scores, 
    and influence score. Useful for understanding the academic 
    impact of a paper.
    
    Args:
        document_id: Document ID to get metrics for
    
    Returns:
        JSON with citation metrics
    """
    if not document_id:
        return error_format("Document ID is required")
    
    try:
        monjyu = get_monjyu()
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_format(f"Document not found: {document_id}")
        
        citation_manager = monjyu.get_citation_network()
        
        metrics_data = {
            "citation_count": doc.citation_count,
            "reference_count": doc.reference_count,
            "pagerank": 0.0,
            "hub_score": 0.0,
            "authority_score": 0.0,
            "influence_score": doc.influence_score,
        }
        
        if citation_manager:
            try:
                metrics = citation_manager.get_metrics(document_id)
                if metrics:
                    metrics_data = {
                        "citation_count": metrics.citation_count,
                        "reference_count": metrics.reference_count,
                        "pagerank": round(metrics.pagerank, 6),
                        "hub_score": round(metrics.hub_score, 6),
                        "authority_score": round(metrics.authority_score, 6),
                        "influence_score": round(metrics.influence_score, 6),
                    }
            except Exception as e:
                logger.warning(f"Metrics lookup failed: {e}")
        
        response = {
            "document": {
                "id": document_id,
                "title": doc.title,
            },
            "metrics": metrics_data,
        }
        
        return json_format(response)
    
    except Exception as e:
        logger.exception("Failed to get metrics")
        return error_format(f"Failed to get metrics: {str(e)}")


# === サーバー起動 ===

def main():
    """MCPサーバーをstdioモードで起動"""
    logger.info("Starting MONJYU MCP Server (stdio mode)...")
    mcp.run(transport="stdio")


def main_http(host: str = "127.0.0.1", port: int = 8080):
    """MCPサーバーをStreamable HTTPモードで起動
    
    Args:
        host: ホストアドレス (default: 127.0.0.1 for security)
        port: ポート番号 (default: 8080)
    """
    logger.info(f"Starting MONJYU MCP Server (HTTP mode) on {host}:{port}...")
    mcp.run(transport="streamable-http", host=host, port=port)


def run():
    """CLIエントリーポイント（--help対応）"""
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("--help", "-h"):
            print("""MONJYU MCP Server

Usage:
    monjyu-mcp                    Start MCP server (stdio mode, default)
    monjyu-mcp --http             Start MCP server (HTTP mode on localhost:8080)
    monjyu-mcp --http --port 3000 Start MCP server (HTTP mode on custom port)
    monjyu-mcp --http --host 0.0.0.0 --port 8080
                                  Start MCP server (HTTP mode, all interfaces)
    monjyu-mcp --help             Show this help message

Transport Modes:
    stdio (default)    For local integration (Claude Desktop, etc.)
    http               For remote/web integration (Streamable HTTP + SSE)

MCP Tools:
    monjyu_search           Search academic papers (vector/lazy/auto)
    monjyu_get_document     Get document details by ID
    monjyu_list_documents   List indexed documents
    monjyu_citation_chain   Get citation chain for a document
    monjyu_find_related     Find related papers
    monjyu_status           Get index status
    monjyu_get_metrics      Get citation metrics

MCP Resources:
    monjyu://index/status           Index status
    monjyu://documents              Document list
    monjyu://documents/{id}         Document detail
    monjyu://documents/{id}/content Document content
    monjyu://documents/{id}/citations Citation info
    monjyu://citation-network       Citation network overview

MCP Prompts:
    literature_review       Generate literature review
    paper_summary           Summarize a paper
    compare_papers          Compare multiple papers
    research_question       Explore research questions
    citation_analysis       Analyze citation patterns

Configuration:
    Set MONJYU_CONFIG environment variable to specify config file path.
    Default locations:
      - ./monjyu.yaml
      - ./config/monjyu.yaml
      - ~/.config/monjyu/monjyu.yaml

Example (Claude Desktop - stdio):
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "monjyu": {
                "command": "monjyu-mcp"
            }
        }
    }

Example (HTTP mode):
    $ monjyu-mcp --http --port 8080
    
    Then connect to: http://localhost:8080/mcp

Security Note:
    When using HTTP mode:
    - Default binds to localhost (127.0.0.1) only
    - Use --host 0.0.0.0 with caution (exposes to network)
    - Consider using authentication for production
""")
            sys.exit(0)
        elif arg == "--version":
            print("monjyu-mcp 0.2.0")
            sys.exit(0)
        elif arg == "--http":
            # HTTP mode
            host = "127.0.0.1"
            port = 8080
            
            # Parse additional arguments
            i = 2
            while i < len(sys.argv):
                if sys.argv[i] == "--host" and i + 1 < len(sys.argv):
                    host = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--port" and i + 1 < len(sys.argv):
                    try:
                        port = int(sys.argv[i + 1])
                    except ValueError:
                        print(f"Invalid port: {sys.argv[i + 1]}")
                        sys.exit(1)
                    i += 2
                else:
                    print(f"Unknown option: {sys.argv[i]}")
                    sys.exit(1)
            
            main_http(host=host, port=port)
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information.")
            sys.exit(1)
    else:
        main()


# === レガシー互換（旧APIとの互換性） ===
server = mcp._mcp_server  # 古いコードとの互換性のため


if __name__ == "__main__":
    main()

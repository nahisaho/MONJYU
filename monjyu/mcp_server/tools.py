# MONJYU MCP Server - Tool Definitions
"""
FEAT-009: MCP Server - ツール定義
7つのMCPツールを定義
"""

from mcp.types import Tool

# === ツール定義 ===

MONJYU_TOOLS = [
    Tool(
        name="monjyu_search",
        description=(
            "Search academic papers in MONJYU index. "
            "Returns relevant papers with citations and an AI-generated answer. "
            "Use 'lazy' mode for complex questions requiring deep reasoning, "
            "'vector' mode for simple keyword searches."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in natural language"
                },
                "mode": {
                    "type": "string",
                    "enum": ["vector", "lazy", "auto"],
                    "default": "lazy",
                    "description": (
                        "Search mode: 'vector' (fast semantic search), "
                        "'lazy' (Progressive GraphRAG with deep reasoning), "
                        "'auto' (automatically select based on query)"
                    )
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    ),
    
    Tool(
        name="monjyu_get_document",
        description=(
            "Get detailed information about a specific document "
            "including title, authors, abstract, and citation metrics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to retrieve"
                }
            },
            "required": ["document_id"]
        }
    ),
    
    Tool(
        name="monjyu_list_documents",
        description=(
            "List all documents in the MONJYU index with basic metadata. "
            "Supports pagination with limit and offset parameters."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of documents to return"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Offset for pagination"
                }
            }
        }
    ),
    
    Tool(
        name="monjyu_citation_chain",
        description=(
            "Get the citation chain for a document, "
            "showing what papers it cites (references) and what papers cite it (citations). "
            "Useful for understanding the academic context and influence of a paper."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to get citation chain for"
                },
                "depth": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Depth of citation chain to retrieve"
                }
            },
            "required": ["document_id"]
        }
    ),
    
    Tool(
        name="monjyu_find_related",
        description=(
            "Find papers related to a given document based on citation patterns "
            "(co-citation, bibliographic coupling) and content similarity. "
            "Returns papers that are thematically or academically connected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to find related papers for"
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of related papers to return"
                }
            },
            "required": ["document_id"]
        }
    ),
    
    Tool(
        name="monjyu_status",
        description=(
            "Get the current status of the MONJYU index "
            "including document count, index levels built, and statistics. "
            "Useful for checking if the index is ready for queries."
        ),
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    
    Tool(
        name="monjyu_get_metrics",
        description=(
            "Get citation metrics for a specific document including "
            "citation count, PageRank, hub/authority scores, and influence score. "
            "Useful for understanding the academic impact of a paper."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to get metrics for"
                }
            },
            "required": ["document_id"]
        }
    ),
]


def get_tool_by_name(name: str) -> Tool | None:
    """ツール名からツール定義を取得
    
    Args:
        name: ツール名
    
    Returns:
        Tool: ツール定義、見つからない場合はNone
    """
    for tool in MONJYU_TOOLS:
        if tool.name == name:
            return tool
    return None

# MONJYU MCP Server - Tool Handlers
"""
FEAT-009: MCP Server - ツールハンドラー実装
各MCPツールの処理ロジック
"""

import json
from typing import Any

from mcp.types import TextContent

from monjyu.api import SearchMode


# === ヘルパー関数 ===

def json_response(data: dict) -> list[TextContent]:
    """JSON形式のレスポンスを生成"""
    return [TextContent(
        type="text",
        text=json.dumps(data, ensure_ascii=False, indent=2)
    )]


def error_response(message: str) -> list[TextContent]:
    """エラーレスポンスを生成"""
    return json_response({"error": message})


# === 検索ハンドラー ===

async def handle_search(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_search ツールのハンドラー
    
    学術論文を検索し、AI生成の回答と引用を返す
    """
    query = args.get("query", "")
    if not query:
        return error_response("Query is required")
    
    mode_str = args.get("mode", "lazy")
    try:
        mode = SearchMode(mode_str)
    except ValueError:
        mode = SearchMode.LAZY
    
    top_k = args.get("top_k", 10)
    
    try:
        result = monjyu.search(query, mode=mode, top_k=top_k)
        
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
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Search failed: {str(e)}")


# === ドキュメント取得ハンドラー ===

async def handle_get_document(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_get_document ツールのハンドラー
    
    指定されたドキュメントの詳細情報を返す
    """
    document_id = args.get("document_id", "")
    if not document_id:
        return error_response("Document ID is required")
    
    try:
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            return error_response(f"Document not found: {document_id}")
        
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
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Failed to get document: {str(e)}")


# === ドキュメント一覧ハンドラー ===

async def handle_list_documents(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_list_documents ツールのハンドラー
    
    インデックス内のドキュメント一覧を返す
    """
    limit = min(args.get("limit", 20), 100)
    offset = args.get("offset", 0)
    
    try:
        documents = monjyu.list_documents(limit=limit)
        
        # オフセット適用（簡易実装）
        if offset > 0:
            documents = documents[offset:]
        
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
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Failed to list documents: {str(e)}")


# === 引用チェーンハンドラー ===

async def handle_citation_chain(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_citation_chain ツールのハンドラー
    
    ドキュメントの引用チェーン（引用元・被引用）を返す
    """
    document_id = args.get("document_id", "")
    if not document_id:
        return error_response("Document ID is required")
    
    depth = min(args.get("depth", 1), 3)
    
    try:
        doc = monjyu.get_document(document_id)
        if doc is None:
            return error_response(f"Document not found: {document_id}")
        
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
            except Exception:
                pass
        
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
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Failed to get citation chain: {str(e)}")


# === 関連論文ハンドラー ===

async def handle_find_related(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_find_related ツールのハンドラー
    
    指定されたドキュメントに関連する論文を返す
    """
    document_id = args.get("document_id", "")
    if not document_id:
        return error_response("Document ID is required")
    
    top_k = min(args.get("top_k", 10), 50)
    
    try:
        doc = monjyu.get_document(document_id)
        if doc is None:
            return error_response(f"Document not found: {document_id}")
        
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
            except Exception:
                pass
        
        response = {
            "source_document": {
                "id": document_id,
                "title": doc.title,
            },
            "related_papers": related_papers,
            "count": len(related_papers),
        }
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Failed to find related papers: {str(e)}")


# === ステータスハンドラー ===

async def handle_status(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_status ツールのハンドラー
    
    インデックスの現在の状態を返す
    """
    try:
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
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Failed to get status: {str(e)}")


# === メトリクスハンドラー ===

async def handle_get_metrics(monjyu, args: dict[str, Any]) -> list[TextContent]:
    """monjyu_get_metrics ツールのハンドラー
    
    ドキュメントの引用メトリクスを返す
    """
    document_id = args.get("document_id", "")
    if not document_id:
        return error_response("Document ID is required")
    
    try:
        doc = monjyu.get_document(document_id)
        if doc is None:
            return error_response(f"Document not found: {document_id}")
        
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
            except Exception:
                pass
        
        response = {
            "document": {
                "id": document_id,
                "title": doc.title,
            },
            "metrics": metrics_data,
        }
        
        return json_response(response)
    
    except Exception as e:
        return error_response(f"Failed to get metrics: {str(e)}")


# === ツールディスパッチャー ===

TOOL_HANDLERS = {
    "monjyu_search": handle_search,
    "monjyu_get_document": handle_get_document,
    "monjyu_list_documents": handle_list_documents,
    "monjyu_citation_chain": handle_citation_chain,
    "monjyu_find_related": handle_find_related,
    "monjyu_status": handle_status,
    "monjyu_get_metrics": handle_get_metrics,
}


async def dispatch_tool(name: str, arguments: dict[str, Any], monjyu) -> list[TextContent]:
    """ツール呼び出しをディスパッチ
    
    Args:
        name: ツール名
        arguments: ツール引数
        monjyu: MONJYUインスタンス
    
    Returns:
        list[TextContent]: レスポンス
    """
    handler = TOOL_HANDLERS.get(name)
    
    if handler is None:
        return error_response(f"Unknown tool: {name}")
    
    return await handler(monjyu, arguments)

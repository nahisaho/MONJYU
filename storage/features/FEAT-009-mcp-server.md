# FEAT-009: MCP Server (Model Context Protocol Server)

**フィーチャーID**: FEAT-009  
**名称**: MCPサーバー  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

AIアシスタント（Claude、GitHub Copilot等）がMONJYUを利用できるようにするMCPサーバー。

### 1.1 スコープ

```python
# AI Assistant からの呼び出しイメージ
mcp.call("monjyu_search", {"query": "What is Transformer?"})
mcp.call("monjyu_get_document", {"document_id": "doc_001"})
mcp.call("monjyu_citation_chain", {"document_id": "doc_001"})
```

- **入力**: MCP Tool呼び出し
- **処理**: Python APIの呼び出し
- **出力**: 構造化されたJSON応答
- **特徴**: 7つのMCPツールを提供

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-EXT-MCP-001 | monjyu_search ツール | P0 |
| FR-EXT-MCP-002 | monjyu_get_document ツール | P0 |
| FR-EXT-MCP-003 | monjyu_list_documents ツール | P0 |
| FR-EXT-MCP-004 | monjyu_citation_chain ツール | P1 |
| FR-EXT-MCP-005 | monjyu_find_related ツール | P1 |
| FR-EXT-MCP-006 | monjyu_status ツール | P1 |
| FR-EXT-MCP-007 | monjyu_get_metrics ツール | P2 |

### 1.3 依存関係

- **依存**: FEAT-007 (Python API)
- **被依存**: なし

---

## 2. アーキテクチャ

### 2.1 MCPツール一覧

| Tool名 | 説明 | 主要パラメータ |
|--------|------|----------------|
| `monjyu_search` | 学術論文を検索 | query, mode, top_k |
| `monjyu_get_document` | ドキュメント詳細を取得 | document_id |
| `monjyu_list_documents` | ドキュメント一覧を取得 | limit, offset |
| `monjyu_citation_chain` | 引用チェーンを取得 | document_id, depth |
| `monjyu_find_related` | 関連論文を検索 | document_id, top_k |
| `monjyu_status` | インデックス状態を取得 | - |
| `monjyu_get_metrics` | 引用メトリクスを取得 | document_id |

### 2.2 コンポーネント図

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           MCP Server                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     MCPServer (mcp.Server)                           │ │
│  │  ┌───────────────────────────────────────────────────────────────┐  │ │
│  │  │                   Tool Registry                                │  │ │
│  │  │  ┌────────────┐ ┌──────────────┐ ┌───────────────────────┐   │  │ │
│  │  │  │ search     │ │ get_document │ │ list_documents        │   │  │ │
│  │  │  └────────────┘ └──────────────┘ └───────────────────────┘   │  │ │
│  │  │  ┌────────────┐ ┌──────────────┐ ┌───────────────────────┐   │  │ │
│  │  │  │ citation   │ │ find_related │ │ status / metrics      │   │  │ │
│  │  │  └────────────┘ └──────────────┘ └───────────────────────┘   │  │ │
│  │  └───────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Tool Handler                                      │ │
│  │  ┌───────────────────┐  ┌───────────────────────────────────────┐  │ │
│  │  │ InputValidator    │  │ ResponseFormatter                      │  │ │
│  │  │                   │  │ (to MCP format)                        │  │ │
│  │  └───────────────────┘  └───────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    MONJYU Python API                                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

                                    ▲
                                    │ stdio / SSE
                                    │
┌──────────────────────────────────────────────────────────────────────────┐
│                       AI Assistant (Claude, etc.)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.3 MCP通信フロー

```
AI Assistant                    MCP Server                    MONJYU API
     │                               │                              │
     │  1. list_tools                │                              │
     │─────────────────────────────>│                              │
     │  ← [monjyu_search, ...]      │                              │
     │                               │                              │
     │  2. call_tool                 │                              │
     │     (monjyu_search)           │                              │
     │─────────────────────────────>│                              │
     │                               │  3. monjyu.search()          │
     │                               │─────────────────────────────>│
     │                               │  ← SearchResult              │
     │                               │                              │
     │  ← tool_result (JSON)        │                              │
     │                               │                              │
```

---

## 3. 詳細設計

### 3.1 MCPサーバーメイン

```python
# monjyu/mcp_server/server.py

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from monjyu import MONJYU, SearchMode

# === サーバー初期化 ===

server = Server("monjyu")

# グローバルMONJYUインスタンス
_monjyu: MONJYU | None = None

def get_monjyu() -> MONJYU:
    """MONJYUインスタンスを取得（遅延初期化）"""
    global _monjyu
    if _monjyu is None:
        config_path = Path("./monjyu.yaml")
        if config_path.exists():
            _monjyu = MONJYU(config_path)
        else:
            _monjyu = MONJYU()
    return _monjyu

# === ツール定義 ===

@server.list_tools()
async def list_tools() -> list[Tool]:
    """利用可能なMCPツール一覧を返す"""
    return [
        Tool(
            name="monjyu_search",
            description="Search academic papers in MONJYU index. Returns relevant papers with citations and an AI-generated answer.",
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
                        "description": "Search mode: vector (fast), lazy (accurate), auto (adaptive)"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of results to return"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="monjyu_get_document",
            description="Get detailed information about a specific document including title, authors, abstract, and citation metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="monjyu_list_documents",
            description="List all documents in the MONJYU index with basic metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of documents to return"
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Offset for pagination"
                    }
                }
            }
        ),
        Tool(
            name="monjyu_citation_chain",
            description="Get the citation chain for a document, showing what it cites and what cites it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "depth": {
                        "type": "integer",
                        "default": 2,
                        "description": "Depth of citation chain to retrieve"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="monjyu_find_related",
            description="Find papers related to a given document based on citation patterns and content similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of related papers to return"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="monjyu_status",
            description="Get the current status of the MONJYU index including document count and index levels.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="monjyu_get_metrics",
            description="Get citation metrics for a document including citation count, PageRank, and influence score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID"
                    }
                },
                "required": ["document_id"]
            }
        ),
    ]
```

### 3.2 ツールハンドラー

```python
# monjyu/mcp_server/handlers.py

from mcp.types import TextContent, CallToolResult
from typing import Any

from .server import server, get_monjyu
from monjyu import SearchMode

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """MCPツール呼び出しを処理"""
    
    monjyu = get_monjyu()
    
    try:
        if name == "monjyu_search":
            return await handle_search(monjyu, arguments)
        
        elif name == "monjyu_get_document":
            return await handle_get_document(monjyu, arguments)
        
        elif name == "monjyu_list_documents":
            return await handle_list_documents(monjyu, arguments)
        
        elif name == "monjyu_citation_chain":
            return await handle_citation_chain(monjyu, arguments)
        
        elif name == "monjyu_find_related":
            return await handle_find_related(monjyu, arguments)
        
        elif name == "monjyu_status":
            return await handle_status(monjyu, arguments)
        
        elif name == "monjyu_get_metrics":
            return await handle_get_metrics(monjyu, arguments)
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


# === 各ツールのハンドラー ===

async def handle_search(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_search ツール"""
    
    query = args["query"]
    mode = SearchMode(args.get("mode", "lazy"))
    top_k = args.get("top_k", 10)
    
    result = monjyu.search(query, mode=mode, top_k=top_k)
    
    response = {
        "query": result.query,
        "answer": result.answer,
        "citations": [
            {
                "document_id": c.document_id,
                "title": c.document_title,
                "authors": c.document_authors,
                "year": c.document_year,
                "snippet": c.text_snippet
            }
            for c in result.citations
        ],
        "search_info": {
            "mode": result.search_mode.value,
            "level": result.search_level,
            "time_ms": result.total_time_ms,
            "llm_calls": result.llm_calls
        }
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]


async def handle_get_document(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_get_document ツール"""
    
    document_id = args["document_id"]
    doc = monjyu.get_document(document_id)
    
    if doc is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Document not found: {document_id}"})
        )]
    
    response = {
        "id": doc.id,
        "title": doc.title,
        "authors": doc.authors,
        "year": doc.year,
        "doi": doc.doi,
        "abstract": doc.abstract,
        "index_stats": {
            "chunk_count": doc.chunk_count
        },
        "citation_metrics": {
            "citation_count": doc.citation_count,
            "reference_count": doc.reference_count,
            "influence_score": doc.influence_score
        }
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]


async def handle_list_documents(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_list_documents ツール"""
    
    limit = args.get("limit", 20)
    offset = args.get("offset", 0)
    
    documents = monjyu.list_documents(limit=limit, offset=offset)
    
    response = {
        "count": len(documents),
        "offset": offset,
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "authors": d.authors[:3],  # 最初の3人のみ
                "year": d.year,
                "chunks": d.chunk_count
            }
            for d in documents
        ]
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]


async def handle_citation_chain(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_citation_chain ツール"""
    
    document_id = args["document_id"]
    depth = args.get("depth", 2)
    
    chain = monjyu.get_citation_chain(document_id, depth=depth)
    
    # ドキュメント情報を付加
    doc = monjyu.get_document(document_id)
    
    async def enrich_document_info(doc_ids: list[str]) -> list[dict]:
        """ドキュメントIDリストに情報を付加"""
        results = []
        for did in doc_ids[:10]:  # 最大10件
            d = monjyu.get_document(did)
            if d:
                results.append({
                    "id": did,
                    "title": d.title,
                    "year": d.year
                })
            else:
                results.append({"id": did})
        return results
    
    response = {
        "document": {
            "id": document_id,
            "title": doc.title if doc else None
        },
        "depth": depth,
        "cites": await enrich_document_info(chain.get("cites", [])),
        "cited_by": await enrich_document_info(chain.get("cited_by", []))
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]


async def handle_find_related(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_find_related ツール"""
    
    document_id = args["document_id"]
    top_k = args.get("top_k", 10)
    
    related = monjyu.find_related_papers(document_id, top_k=top_k)
    
    response = {
        "source_document_id": document_id,
        "related_papers": [
            {
                "document_id": r.document_id,
                "title": r.title,
                "authors": r.authors[:3] if hasattr(r, 'authors') else [],
                "year": r.year if hasattr(r, 'year') else None,
                "relationship": r.relationship_type,
                "similarity_score": round(r.similarity_score, 4)
            }
            for r in related
        ]
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]


async def handle_status(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_status ツール"""
    
    status = monjyu.get_status()
    
    response = {
        "index_status": status.index_status.value,
        "levels_built": [l.value for l in status.index_levels_built],
        "statistics": {
            "documents": status.document_count,
            "text_units": status.text_unit_count,
            "noun_phrases": status.noun_phrase_count,
            "communities": status.community_count,
            "citation_edges": status.citation_edge_count
        },
        "last_error": status.last_error
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]


async def handle_get_metrics(monjyu: "MONJYU", args: dict) -> list[TextContent]:
    """monjyu_get_metrics ツール"""
    
    document_id = args["document_id"]
    
    citation_manager = monjyu.get_citation_network()
    metrics = citation_manager.get_metrics(document_id)
    
    if metrics is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Metrics not found: {document_id}"})
        )]
    
    doc = monjyu.get_document(document_id)
    
    response = {
        "document": {
            "id": document_id,
            "title": doc.title if doc else None
        },
        "metrics": {
            "citation_count": metrics.citation_count,
            "reference_count": metrics.reference_count,
            "pagerank": round(metrics.pagerank, 6),
            "hub_score": round(metrics.hub_score, 6),
            "authority_score": round(metrics.authority_score, 6),
            "influence_score": round(metrics.influence_score, 6)
        }
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, ensure_ascii=False, indent=2)
    )]
```

### 3.3 サーバー起動

```python
# monjyu/mcp_server/__init__.py

import asyncio
from .server import server
from .handlers import *  # ハンドラーを登録

async def main():
    """MCPサーバーをstdioで起動"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def run():
    """エントリーポイント"""
    asyncio.run(main())

if __name__ == "__main__":
    run()
```

```toml
# pyproject.toml (抜粋)

[project.scripts]
monjyu = "monjyu.cli:main"
monjyu-mcp = "monjyu.mcp_server:run"
```

---

## 4. 設定

### 4.1 Claude Desktop設定

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "monjyu": {
      "command": "monjyu-mcp",
      "args": [],
      "env": {
        "MONJYU_CONFIG": "/path/to/monjyu.yaml"
      }
    }
  }
}
```

### 4.2 VS Code設定 (GitHub Copilot)

```json
// settings.json
{
  "github.copilot.chat.mcpServers": {
    "monjyu": {
      "command": "monjyu-mcp",
      "args": [],
      "env": {
        "MONJYU_CONFIG": "${workspaceFolder}/monjyu.yaml"
      }
    }
  }
}
```

---

## 5. 使用例

### 5.1 AIアシスタントからの利用

```
User: "Transformerについて教えて"

AI Assistant: monjyu_search を呼び出します...

{
  "query": "What is Transformer architecture?",
  "answer": "Transformerは2017年にVaswaniらによって提案された...",
  "citations": [
    {
      "document_id": "attention_is_all_you_need",
      "title": "Attention Is All You Need",
      "authors": ["Vaswani, A.", "Shazeer, N.", ...],
      "year": 2017,
      "snippet": "We propose a new simple network architecture..."
    }
  ]
}
```

### 5.2 引用チェーンの探索

```
User: "この論文の引用関係を見せて"

AI Assistant: monjyu_citation_chain を呼び出します...

{
  "document": {
    "id": "attention_is_all_you_need",
    "title": "Attention Is All You Need"
  },
  "depth": 2,
  "cites": [
    {"id": "seq2seq", "title": "Sequence to Sequence Learning", "year": 2014}
  ],
  "cited_by": [
    {"id": "bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018},
    {"id": "gpt2", "title": "Language Models are Unsupervised Multitask Learners", "year": 2019}
  ]
}
```

---

## 6. テスト計画

### 6.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_list_tools | list_tools | 7ツール返却 |
| test_search_tool | monjyu_search | 検索結果返却 |
| test_get_document | monjyu_get_document | ドキュメント情報 |
| test_invalid_tool | 不正ツール名 | エラーメッセージ |

### 6.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_mcp_protocol | stdio通信 | 正常応答 |
| test_claude_integration | Claude Desktop | ツール動作 |
| test_error_handling | エラーケース | 適切なエラー |

### 6.3 テストコード例

```python
# tests/unit/test_mcp_server.py

import pytest
import json
from monjyu.mcp_server.handlers import (
    handle_search, handle_get_document, handle_status
)

@pytest.fixture
def mock_monjyu(mocker):
    """MONJYUモック"""
    mock = mocker.Mock()
    mock.search.return_value = mocker.Mock(
        query="test query",
        answer="test answer",
        citations=[],
        search_mode=mocker.Mock(value="lazy"),
        search_level=1,
        total_time_ms=100,
        llm_calls=2
    )
    return mock


@pytest.mark.asyncio
async def test_handle_search(mock_monjyu):
    """検索ツールのテスト"""
    args = {"query": "test query", "mode": "lazy"}
    
    result = await handle_search(mock_monjyu, args)
    
    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["query"] == "test query"
    assert response["answer"] == "test answer"


@pytest.mark.asyncio
async def test_handle_status(mock_monjyu):
    """ステータスツールのテスト"""
    mock_monjyu.get_status.return_value = mocker.Mock(
        index_status=mocker.Mock(value="ready"),
        index_levels_built=[],
        document_count=10,
        text_unit_count=100,
        noun_phrase_count=500,
        community_count=5,
        citation_edge_count=20,
        last_error=None
    )
    
    result = await handle_status(mock_monjyu, {})
    
    response = json.loads(result[0].text)
    assert response["index_status"] == "ready"
    assert response["statistics"]["documents"] == 10
```

---

## 7. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-009-01 | MCPサーバー基盤 | 2h | - |
| TASK-009-02 | monjyu_search 実装 | 2h | FEAT-007 |
| TASK-009-03 | monjyu_get_document 実装 | 1h | FEAT-007 |
| TASK-009-04 | monjyu_list_documents 実装 | 1h | FEAT-007 |
| TASK-009-05 | monjyu_citation_chain 実装 | 1h | FEAT-007 |
| TASK-009-06 | monjyu_find_related 実装 | 1h | FEAT-007 |
| TASK-009-07 | monjyu_status/metrics 実装 | 1h | FEAT-007 |
| TASK-009-08 | エラーハンドリング | 1h | TASK-009-01~07 |
| TASK-009-09 | テスト作成 | 2h | TASK-009-01~08 |
| TASK-009-10 | ドキュメント・設定例 | 1h | TASK-009-01~08 |
| **合計** | | **13h** | |

---

## 8. 受入基準

- [ ] MCPサーバーが stdio で起動できる
- [ ] `monjyu_search` で検索結果が返る
- [ ] `monjyu_get_document` でドキュメント情報が返る
- [ ] `monjyu_list_documents` で一覧が返る
- [ ] `monjyu_citation_chain` で引用チェーンが返る
- [ ] `monjyu_find_related` で関連論文が返る
- [ ] `monjyu_status` でインデックス状態が返る
- [ ] `monjyu_get_metrics` でメトリクスが返る
- [ ] Claude Desktop から接続・利用できる
- [ ] エラー時に適切なエラーメッセージを返す

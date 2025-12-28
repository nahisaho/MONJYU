# MONJYU MCP Server
"""
FEAT-009: MCP Server (Model Context Protocol Server)
MCPサーバーエントリーポイント

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
    - monjyu://documents/{id}/citations: 引用情報
    - monjyu://citation-network: 引用ネットワーク概要

MCP Prompts:
    - literature_review: 文献レビュー生成
    - paper_summary: 論文要約
    - compare_papers: 論文比較
    - research_question: 研究質問の探索
    - citation_analysis: 引用分析
"""

# FastMCPベースの新しい実装をインポート
from monjyu.mcp_server.server import (
    mcp,
    main,
    main_http,
    run,
    get_monjyu,
    reset_monjyu,
    set_monjyu,
)

# レガシー互換（tools.py, handlers.py は参照用に保持）
# 実際のツール定義はserver.py内の@mcp.tool()デコレータで行われる

__all__ = [
    "mcp",
    "main",
    "main_http",
    "run",
    "get_monjyu",
    "reset_monjyu",
    "set_monjyu",
]


if __name__ == "__main__":
    run()

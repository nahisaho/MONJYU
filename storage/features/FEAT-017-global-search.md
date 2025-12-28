# FEAT-017: GlobalSearch

## 概要

グローバル検索コンポーネント（GraphRAG Global Search）。コミュニティレポートを使用したmap-reduceパターンで回答生成。

## 要件 (REQ-QRY-002)

- コミュニティレポートを使用
- map-reduce パターンで回答生成
- データセット全体の要約質問に回答可能

## コンポーネント

### 1. GlobalSearchConfig
- `community_level`: 使用するコミュニティレベル
- `top_k_communities`: 取得コミュニティ数
- `map_reduce_enabled`: map-reduce有効化
- `max_context_tokens`: 最大コンテキストトークン
- `temperature`: LLM temperature

### 2. GlobalSearchResult
- `query`: クエリ
- `answer`: 回答
- `communities_used`: 使用したコミュニティ
- `map_results`: map結果
- `processing_time_ms`: 処理時間

### 3. GlobalSearch
- `search(query, level)`: グローバル検索
- `_map_phase(query, communities)`: Map処理
- `_reduce_phase(query, map_results)`: Reduce処理

## 処理フロー

```
1. クエリ受信
2. 関連コミュニティを取得（VectorSearch）
3. Map: 各コミュニティレポートから部分回答生成
4. Reduce: 部分回答を統合して最終回答
```

## テスト項目

1. 設定テスト
   - デフォルト設定
   - カスタム設定

2. 結果テスト
   - 作成
   - 辞書変換

3. Map-Reduce テスト
   - Map処理
   - Reduce処理
   - 完全フロー

4. 統合テスト
   - 完全な検索ワークフロー

## 依存

- `monjyu.query.vector_search.VectorSearchProtocol`
- `monjyu.index.community_report_generator.CommunityReport`

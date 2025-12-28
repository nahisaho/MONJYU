# FEAT-016: VectorSearch

## 概要

ベクトル検索コンポーネント（Baseline RAG）。LanceDB/Azure AI Searchをバックエンドとして使用。

## 要件 (REQ-QRY-001)

- ベクトル類似度検索
- ハイブリッド検索（ベクトル + キーワード）
- フィルタリング対応
- レイテンシ < 1秒

## コンポーネント

### 1. SearchHit
- `chunk_id`: チャンクID
- `score`: スコア
- `content`: コンテンツ
- `metadata`: メタデータ
- `paper_id`: 論文ID
- `paper_title`: 論文タイトル
- `section_type`: セクションタイプ

### 2. VectorSearchConfig
- `top_k`: 取得件数（デフォルト10）
- `min_score`: 最小スコア閾値
- `include_metadata`: メタデータ含める
- `rerank`: リランキング有効化
- `rerank_model`: リランキングモデル

### 3. VectorSearchResult
- `hits`: 検索ヒット一覧
- `total_count`: 総件数
- `processing_time_ms`: 処理時間

### 4. EmbedderProtocol
- `embed(text)`: テキスト埋め込み
- `embed_batch(texts)`: バッチ埋め込み
- `dimension`: 埋め込み次元

### 5. VectorSearchProtocol
- `search(query, top_k, filter)`: テキスト検索
- `search_by_vector(vector, top_k, filter)`: ベクトル検索
- `hybrid_search(query, top_k, alpha, filter)`: ハイブリッド検索

### 6. InMemoryVectorSearch
- テスト・開発用インメモリ実装
- numpy cosine similarity

### 7. LanceDBVectorSearch
- ローカル環境向け
- LanceDBバックエンド

### 8. AzureAISearchVectorSearch (Future)
- 本番環境向け
- Azure AI Search バックエンド

## テスト項目

1. SearchHitテスト
   - 作成
   - デフォルト値

2. VectorSearchConfigテスト
   - デフォルト設定
   - カスタム設定

3. VectorSearchResultテスト
   - 作成
   - 辞書変換

4. EmbedderProtocolテスト
   - モックエンベッダー
   - バッチ埋め込み

5. InMemoryVectorSearchテスト
   - インデックス作成
   - ベクトル検索
   - テキスト検索
   - フィルタリング

6. 統合テスト
   - 完全な検索ワークフロー

## 依存

- `numpy`: コサイン類似度計算

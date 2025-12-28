# FEAT-018: LocalSearch

## 概要
エンティティを起点とした局所検索。指定エンティティの関連エンティティ、リレーションシップ、所属チャンクを探索して回答を生成します。

## 機能要件

### 検索フロー
1. クエリからエンティティを抽出
2. 抽出エンティティに関連するノードを探索（N-hop）
3. 関連チャンクを収集
4. コンテキストを構築してLLM回答生成

### 設定項目
- `max_hops`: 探索ホップ数（デフォルト: 2）
- `top_k_entities`: 上位エンティティ数
- `top_k_chunks`: 上位チャンク数
- `include_relationships`: リレーション情報を含めるか
- `max_context_tokens`: 最大コンテキストトークン数

### データ型
- `LocalSearchConfig`: 検索設定
- `EntityMatch`: エンティティマッチ結果
- `LocalSearchResult`: 検索結果

## 実装タスク

1. [x] types.py - データ型定義
2. [x] search.py - LocalSearch実装
3. [x] prompts.py - プロンプト定義
4. [x] __init__.py - モジュール公開
5. [x] tests/unit/test_local_search.py - テスト

## テスト項目

### LocalSearchConfig
- デフォルト値
- カスタム値
- to_dict/from_dict

### EntityMatch
- 基本作成
- to_dict

### LocalSearchResult
- 基本作成
- to_dict/from_dict

### LocalSearch
- 基本検索
- ホップ数指定
- 空グラフ検索
- 設定オーバーライド
- コンテキスト構築

### 統合テスト
- エンティティベース検索ワークフロー
- マルチホップ探索

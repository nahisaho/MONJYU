# FEAT-015: UnifiedController

## 概要

統合検索コントローラ。QueryRouterの決定に基づいて最適な検索エンジンを選択・実行する。

## 要件 (REQ-ARC-001)

- 統一インターフェースで動的に最適な検索戦略を選択
- 自動モード選択でHybrid比30%コスト削減
- async対応、型ヒント完備

## コンポーネント

### 1. UnifiedControllerConfig
- `default_mode`: デフォルト検索モード
- `enable_auto_routing`: 自動ルーティング有効化
- `fallback_enabled`: フォールバック有効化
- `timeout_seconds`: タイムアウト秒数
- `max_retries`: 最大リトライ数

### 2. SearchEngineProtocol
- 各検索エンジンの抽象インターフェース
- `search(query, context)`: 検索実行
- `is_available()`: 利用可能チェック

### 3. UnifiedController
- `search(query, mode, context)`: 統合検索
- `search_with_retry(query, mode, context)`: リトライ付き検索
- `register_engine(mode, engine)`: エンジン登録
- `get_available_modes()`: 利用可能モード取得

### 4. SearchContext
- `query_type`: クエリタイプ
- `max_results`: 最大結果数
- `include_metadata`: メタデータ含める
- `language`: 言語設定

### 5. UnifiedSearchResult
- `mode_used`: 使用したモード
- `results`: 検索結果
- `metadata`: メタデータ
- `processing_time_ms`: 処理時間
- `fallback_used`: フォールバック使用有無

## インターフェース

```python
class UnifiedController:
    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.AUTO,
        context: Optional[SearchContext] = None,
    ) -> UnifiedSearchResult:
        if mode == SearchMode.AUTO:
            routing = await self.router.route(query, routing_context)
            mode = routing.mode
        
        engine = self._get_engine(mode)
        return await engine.search(query, context)
```

## テスト項目

1. 設定テスト
   - デフォルト設定
   - カスタム設定

2. エンジン管理テスト
   - エンジン登録
   - 利用可能モード取得
   - 不明モードエラー

3. 自動ルーティングテスト
   - AUTOモード→VECTORルーティング
   - AUTOモード→LAZYルーティング
   - AUTOモード→GRAPHRAGルーティング

4. 手動モード選択テスト
   - 指定モード使用
   - ユーザー優先設定

5. フォールバックテスト
   - エンジン失敗時フォールバック
   - フォールバック無効時エラー

6. リトライテスト
   - 一時エラー時リトライ
   - 最大リトライ超過

7. 結果形式テスト
   - 結果メタデータ
   - 処理時間計測

## 依存

- `monjyu.query.router.QueryRouter`
- `monjyu.query.router.types`

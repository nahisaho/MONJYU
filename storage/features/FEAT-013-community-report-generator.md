# FEAT-013: CommunityReportGenerator

## 概要
コミュニティのエグゼクティブサマリーを LLM で生成するコンポーネント。

## 対応要件
- REQ-IDX-008: コミュニティレポート生成

## 機能
1. **CommunityReport 生成**: コミュニティのメンバーから要約を生成
2. **タイトル自動生成**: コミュニティの主題を表すタイトル
3. **Findings 抽出**: 重要な発見事項のリスト
4. **バッチ処理**: 複数コミュニティの一括処理

## データモデル

### CommunityReport
```python
@dataclass
class CommunityReport:
    community_id: str
    title: str
    summary: str
    findings: List[str]
    rating: float  # 重要度スコア (0-10)
    rating_explanation: str
    full_content: str  # 詳細コンテンツ
    metadata: Dict[str, Any]
```

## インターフェース

```python
class CommunityReportGeneratorProtocol(Protocol):
    async def generate(
        self,
        community: Community,
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> CommunityReport:
        ...
    
    async def generate_batch(
        self,
        communities: List[Community],
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> List[CommunityReport]:
        ...
```

## プロンプト設計

### 入力情報
- コミュニティ ID
- メンバーエンティティのリスト（名前、タイプ、説明）
- メンバー間の関係性

### 出力形式
```json
{
  "title": "研究テーマのタイトル",
  "summary": "このコミュニティは...",
  "findings": [
    "発見1: ...",
    "発見2: ..."
  ],
  "rating": 8.5,
  "rating_explanation": "評価理由..."
}
```

## ディレクトリ構造
```
monjyu/index/community_report_generator/
├── __init__.py
├── types.py        # CommunityReport データモデル
├── prompts.py      # EN/JA プロンプト
├── protocol.py     # Protocol 定義
└── generator.py    # LLM 実装
```

## 依存関係
- FEAT-012 CommunityDetector (Community 型)
- FEAT-010 EntityExtractor (Entity 型)
- FEAT-011 RelationshipExtractor (Relationship 型)
- monjyu/core/chat_model.py (ChatModelProtocol)

## テスト計画
1. CommunityReport 型テスト
2. プロンプトフォーマットテスト
3. レスポンスパーステスト
4. バッチ処理テスト
5. 統合テスト（モック LLM）

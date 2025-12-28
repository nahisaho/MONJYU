# FEAT-010: EntityExtractor - LLMエンティティ抽出

**フェーズ**: Phase 2  
**要件ID**: REQ-IDX-005  
**優先度**: P1  
**見積もり**: 3日  

---

## 1. 概要

### 1.1 目的

学術論文テキストからLLMを使用してエンティティ（研究者、手法、データセット等）を抽出する。
GraphRAG Level 2の基盤コンポーネント。

### 1.2 スコープ

| 含む | 含まない |
|------|----------|
| LLMベースエンティティ抽出 | NLPベース抽出（Level 1で実装済み） |
| 学術エンティティタイプ定義 | 外部API連携（後続フェーズ） |
| バッチ抽出処理 | 関係抽出（FEAT-011） |
| エンティティマージ | グラフ構築 |
| プロンプトテンプレート | |

---

## 2. 要件

### 2.1 機能要件

#### FR-010-001: 学術エンティティタイプ

```python
class AcademicEntityType(Enum):
    """学術エンティティタイプ"""
    RESEARCHER = "researcher"      # 研究者、著者
    ORGANIZATION = "organization"  # 機関、企業、研究室
    METHOD = "method"              # アルゴリズム、手法
    MODEL = "model"                # MLモデル（GPT-4, BERT等）
    DATASET = "dataset"            # データセット（ImageNet等）
    METRIC = "metric"              # 評価指標（accuracy, F1等）
    TASK = "task"                  # タスク（分類、翻訳等）
    CONCEPT = "concept"            # 概念、理論
    TOOL = "tool"                  # ツール、フレームワーク
    PAPER = "paper"                # 参照論文
```

#### FR-010-002: エンティティ抽出

- 単一チャンクからエンティティを抽出
- JSON形式でLLM出力を解析
- 失敗時はリトライ or 空リスト

#### FR-010-003: バッチ抽出

- 並列処理（Semaphore制御）
- 最大同時実行数: 5（設定可能）
- プログレス報告

#### FR-010-004: エンティティマージ

- 同一名称のエンティティを統合
- エイリアス管理
- ソースチャンクID集約

### 2.2 非機能要件

| 要件 | 基準 |
|------|------|
| 処理速度 | 100チャンク/分以上 |
| 抽出精度 | Precision > 0.8 |
| 並列処理 | 最大10並列 |
| エラー率 | < 5% |

---

## 3. 設計

### 3.1 データモデル

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class AcademicEntityType(Enum):
    """学術エンティティタイプ"""
    RESEARCHER = "researcher"
    ORGANIZATION = "organization"
    METHOD = "method"
    MODEL = "model"
    DATASET = "dataset"
    METRIC = "metric"
    TASK = "task"
    CONCEPT = "concept"
    TOOL = "tool"
    PAPER = "paper"


@dataclass
class Entity:
    """エンティティ"""
    id: str
    name: str
    type: AcademicEntityType
    description: str
    aliases: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    
    # 学術固有
    first_mentioned_year: Optional[int] = None
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    # 信頼度
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "aliases": self.aliases,
            "source_chunk_ids": self.source_chunk_ids,
            "first_mentioned_year": self.first_mentioned_year,
            "external_ids": self.external_ids,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """辞書から生成"""
        return cls(
            id=data["id"],
            name=data["name"],
            type=AcademicEntityType(data["type"]),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            source_chunk_ids=data.get("source_chunk_ids", []),
            first_mentioned_year=data.get("first_mentioned_year"),
            external_ids=data.get("external_ids", {}),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ExtractionResult:
    """抽出結果"""
    chunk_id: str
    entities: List[Entity]
    raw_response: str
    tokens_used: int = 0
    extraction_time_ms: float = 0
    error: Optional[str] = None
```

### 3.2 プロンプトテンプレート

```python
ENTITY_EXTRACTION_PROMPT = '''You are an expert at extracting academic entities from scientific papers.

## Task
Extract all significant entities from the following text. Focus on academic concepts that would be useful for understanding the research landscape.

## Entity Types
- RESEARCHER: People mentioned (researchers, authors, scientists)
- ORGANIZATION: Institutions, companies, research labs, universities
- METHOD: Algorithms, techniques, approaches, architectures
- MODEL: Specific ML/AI models (GPT-4, BERT, ResNet, Transformer, etc.)
- DATASET: Datasets used or mentioned (ImageNet, COCO, GLUE, etc.)
- METRIC: Evaluation metrics (accuracy, F1, BLEU, perplexity, etc.)
- TASK: Research tasks (classification, translation, summarization, etc.)
- CONCEPT: Abstract concepts, theories, phenomena
- TOOL: Tools, frameworks, libraries (PyTorch, TensorFlow, etc.)
- PAPER: Referenced papers or works

## Rules
1. Extract only clearly mentioned entities, not implied ones
2. Include the most specific name (e.g., "GPT-4" not just "GPT")
3. Provide concise descriptions (1-2 sentences)
4. List known aliases (e.g., ["BERT", "Bidirectional Encoder Representations from Transformers"])
5. Only include entities relevant to academic/research context

## Text
{text}

## Output Format
Return a JSON object with this exact structure:
```json
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "ENTITY_TYPE",
      "description": "brief description of the entity",
      "aliases": ["alias1", "alias2"]
    }}
  ]
}}
```

If no entities are found, return: {{"entities": []}}'''
```

### 3.3 プロトコル

```python
from abc import ABC, abstractmethod
from typing import List, AsyncIterator


class EntityExtractorProtocol(ABC):
    """エンティティ抽出プロトコル"""
    
    @abstractmethod
    async def extract(
        self,
        chunk: "TextChunk"
    ) -> ExtractionResult:
        """単一チャンクからエンティティ抽出"""
        ...
    
    @abstractmethod
    async def extract_batch(
        self,
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> List[ExtractionResult]:
        """複数チャンクから一括抽出"""
        ...
    
    @abstractmethod
    def extract_stream(
        self,
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> AsyncIterator[ExtractionResult]:
        """ストリーミング抽出"""
        ...
    
    @abstractmethod
    def merge_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """重複エンティティをマージ"""
        ...
```

### 3.4 LLM実装

```python
class LLMEntityExtractor(EntityExtractorProtocol):
    """LLMベースのエンティティ抽出"""
    
    def __init__(
        self,
        llm_client: "ChatModelProtocol",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def extract(
        self,
        chunk: "TextChunk"
    ) -> ExtractionResult:
        """単一チャンクからエンティティ抽出"""
        import json
        import time
        import re
        
        start_time = time.time()
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=chunk.content)
        
        for attempt in range(self.max_retries):
            try:
                response = await self.llm_client.chat(prompt)
                
                # JSON抽出
                json_match = re.search(r'\{[\s\S]*\}', response)
                if not json_match:
                    raise ValueError("No JSON found in response")
                
                data = json.loads(json_match.group())
                
                entities = []
                for item in data.get("entities", []):
                    entity = self._parse_entity(item, chunk.id)
                    if entity:
                        entities.append(entity)
                
                return ExtractionResult(
                    chunk_id=chunk.id,
                    entities=entities,
                    raw_response=response,
                    extraction_time_ms=(time.time() - start_time) * 1000,
                )
            
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    return ExtractionResult(
                        chunk_id=chunk.id,
                        entities=[],
                        raw_response=response if 'response' in locals() else "",
                        extraction_time_ms=(time.time() - start_time) * 1000,
                        error=str(e),
                    )
                await asyncio.sleep(self.retry_delay)
    
    def _parse_entity(
        self,
        item: Dict[str, Any],
        chunk_id: str
    ) -> Optional[Entity]:
        """エンティティ解析"""
        try:
            entity_type = AcademicEntityType[item["type"].upper()]
        except (KeyError, ValueError):
            entity_type = AcademicEntityType.CONCEPT
        
        return Entity(
            id=str(uuid.uuid4()),
            name=item["name"],
            type=entity_type,
            description=item.get("description", ""),
            aliases=item.get("aliases", []),
            source_chunk_ids=[chunk_id],
        )
    
    async def extract_batch(
        self,
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> List[ExtractionResult]:
        """バッチ抽出"""
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_limit(chunk: "TextChunk") -> ExtractionResult:
            async with semaphore:
                return await self.extract(chunk)
        
        tasks = [extract_with_limit(c) for c in chunks]
        return await asyncio.gather(*tasks)
    
    async def extract_stream(
        self,
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> AsyncIterator[ExtractionResult]:
        """ストリーミング抽出"""
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        queue: asyncio.Queue[ExtractionResult] = asyncio.Queue()
        
        async def extract_and_queue(chunk: "TextChunk"):
            async with semaphore:
                result = await self.extract(chunk)
                await queue.put(result)
        
        tasks = [asyncio.create_task(extract_and_queue(c)) for c in chunks]
        
        completed = 0
        while completed < len(chunks):
            result = await queue.get()
            yield result
            completed += 1
        
        await asyncio.gather(*tasks)
    
    def merge_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """重複エンティティをマージ"""
        from collections import defaultdict
        
        # 正規化された名前でグループ化
        groups: Dict[str, List[Entity]] = defaultdict(list)
        
        for entity in entities:
            key = self._normalize_name(entity.name)
            groups[key].append(entity)
        
        merged: List[Entity] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # マージ
                primary = group[0]
                for other in group[1:]:
                    # エイリアス統合
                    primary.aliases.extend(other.aliases)
                    if other.name not in primary.aliases and other.name != primary.name:
                        primary.aliases.append(other.name)
                    
                    # ソースチャンク統合
                    primary.source_chunk_ids.extend(other.source_chunk_ids)
                    
                    # 説明を長い方に
                    if len(other.description) > len(primary.description):
                        primary.description = other.description
                
                # 重複除去
                primary.aliases = list(set(primary.aliases))
                primary.source_chunk_ids = list(set(primary.source_chunk_ids))
                
                # 信頼度更新（出現回数に基づく）
                primary.confidence = min(1.0, len(group) * 0.2 + 0.5)
                
                merged.append(primary)
        
        return merged
    
    def _normalize_name(self, name: str) -> str:
        """名前の正規化"""
        import re
        
        # 小文字化
        name = name.lower()
        # 余分な空白除去
        name = re.sub(r'\s+', ' ', name).strip()
        # 特殊文字除去
        name = re.sub(r'[^\w\s-]', '', name)
        
        return name
```

---

## 4. ディレクトリ構造

```
monjyu/
├── index/
│   └── entity_extractor/
│       ├── __init__.py
│       ├── types.py           # AcademicEntityType, Entity
│       ├── protocol.py        # EntityExtractorProtocol
│       ├── llm_extractor.py   # LLMEntityExtractor
│       └── prompts.py         # プロンプトテンプレート
tests/
└── unit/
    └── test_entity_extractor.py
```

---

## 5. テスト計画

### 5.1 単体テスト

```python
class TestAcademicEntityType:
    """エンティティタイプのテスト"""
    
    def test_all_types_defined(self):
        """全タイプが定義されている"""
        assert len(AcademicEntityType) == 10
    
    def test_type_values(self):
        """タイプ値が正しい"""
        assert AcademicEntityType.MODEL.value == "model"
        assert AcademicEntityType.DATASET.value == "dataset"


class TestEntity:
    """エンティティのテスト"""
    
    def test_entity_creation(self):
        """エンティティ作成"""
        entity = Entity(
            id="test-001",
            name="BERT",
            type=AcademicEntityType.MODEL,
            description="A language model",
        )
        assert entity.name == "BERT"
        assert entity.type == AcademicEntityType.MODEL
    
    def test_to_dict(self):
        """辞書変換"""
        entity = Entity(...)
        data = entity.to_dict()
        assert data["name"] == entity.name
    
    def test_from_dict(self):
        """辞書から生成"""
        data = {"id": "1", "name": "GPT", "type": "model", ...}
        entity = Entity.from_dict(data)
        assert entity.name == "GPT"


class TestLLMEntityExtractor:
    """LLMエンティティ抽出のテスト"""
    
    @pytest.mark.asyncio
    async def test_extract_single_chunk(self):
        """単一チャンク抽出"""
        mock_llm = MockLLMClient()
        mock_llm.set_response('{"entities": [...]}')
        
        extractor = LLMEntityExtractor(mock_llm)
        result = await extractor.extract(chunk)
        
        assert len(result.entities) > 0
    
    @pytest.mark.asyncio
    async def test_extract_batch(self):
        """バッチ抽出"""
        extractor = LLMEntityExtractor(mock_llm)
        results = await extractor.extract_batch(chunks)
        
        assert len(results) == len(chunks)
    
    def test_merge_entities(self):
        """エンティティマージ"""
        entities = [
            Entity(id="1", name="BERT", ...),
            Entity(id="2", name="bert", ...),
        ]
        
        merged = extractor.merge_entities(entities)
        assert len(merged) == 1
        assert "bert" in merged[0].aliases or merged[0].name.lower() == "bert"
    
    @pytest.mark.asyncio
    async def test_extract_with_retry(self):
        """リトライテスト"""
        mock_llm = MockLLMClient()
        mock_llm.fail_first_n(2)
        
        extractor = LLMEntityExtractor(mock_llm, max_retries=3)
        result = await extractor.extract(chunk)
        
        assert result.error is None
```

### 5.2 テストカバレッジ目標

| モジュール | 目標 |
|-----------|------|
| types.py | 100% |
| protocol.py | 100% |
| llm_extractor.py | 90% |
| prompts.py | 100% |

---

## 6. API仕様

### 6.1 使用例

```python
from monjyu.index.entity_extractor import LLMEntityExtractor
from monjyu.core.llm import AzureOpenAIClient

# LLMクライアント
llm = AzureOpenAIClient(
    endpoint="...",
    api_key="...",
    deployment="gpt-4o-mini",
)

# エンティティ抽出
extractor = LLMEntityExtractor(llm)

# 単一チャンク
result = await extractor.extract(chunk)
print(f"Found {len(result.entities)} entities")

# バッチ処理
results = await extractor.extract_batch(chunks, max_concurrent=5)

# マージ
all_entities = []
for r in results:
    all_entities.extend(r.entities)

merged = extractor.merge_entities(all_entities)
print(f"Merged to {len(merged)} unique entities")
```

---

## 7. 実装タスク

| # | タスク | 見積もり |
|---|--------|----------|
| 1 | types.py - データモデル定義 | 2h |
| 2 | prompts.py - プロンプトテンプレート | 1h |
| 3 | protocol.py - プロトコル定義 | 1h |
| 4 | llm_extractor.py - LLM実装 | 4h |
| 5 | 単体テスト作成 | 3h |
| 6 | テスト実行・検証 | 1h |

---

## 8. 依存関係

### 8.1 内部依存

- `monjyu.core.llm.ChatModelProtocol` - LLMクライアント
- `monjyu.index.text_chunker.TextChunk` - チャンク

### 8.2 外部依存

なし（LLMクライアントは既存）

---

## 9. リスク・課題

| リスク | 対策 |
|--------|------|
| LLMコスト増加 | バッチサイズ制限、キャッシュ |
| 抽出品質ばらつき | プロンプト改善、Few-shot追加 |
| JSON解析失敗 | リトライ、フォールバック |
| レート制限 | Semaphore、バックオフ |

---

## 10. 完了条件

- [ ] 全データモデル実装
- [ ] LLMEntityExtractor実装
- [ ] 単体テスト90%以上カバレッジ
- [ ] ドキュメント更新
- [ ] コードレビュー完了

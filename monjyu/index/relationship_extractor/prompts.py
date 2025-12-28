# Relationship Extraction Prompts
"""
FEAT-011: RelationshipExtractor プロンプトテンプレート

LLM関係抽出用のプロンプト
"""


RELATIONSHIP_EXTRACTION_PROMPT = '''You are an expert at extracting relationships between academic entities from scientific papers.

## Task
Given a list of entities and the source text, extract all meaningful relationships between the entities.

## Entities
{entities}

## Relationship Types
- USES: A uses B (e.g., "BERT uses attention mechanism")
- EXTENDS: A extends B (e.g., "RoBERTa extends BERT")
- IMPROVES: A improves upon B
- IMPLEMENTS: A implements B
- BASED_ON: A is based on B
- COMPARES: A is compared with B
- OUTPERFORMS: A outperforms B
- EVALUATES_ON: A is evaluated on B (dataset/benchmark)
- TRAINED_ON: A is trained on B (dataset)
- FINE_TUNED_ON: A is fine-tuned on B
- APPLIED_TO: A is applied to B (task/domain)
- PROPOSED_BY: A is proposed by B (person/organization)
- DEVELOPED_BY: A is developed by B
- AFFILIATED_WITH: A is affiliated with B
- COLLABORATED_WITH: A collaborated with B
- CITES: A cites B
- REFERENCES: A references B
- RELATED_TO: A is related to B (general)
- SIMILAR_TO: A is similar to B
- PART_OF: A is part of B

## Rules
1. Only extract relationships explicitly stated or strongly implied in the text
2. Each relationship should have a clear source and target entity
3. Provide evidence (quote from text) for each relationship
4. Focus on academically meaningful relationships
5. Use the exact entity names provided in the entity list

## Text
{text}

## Output Format
Return a JSON object with this exact structure:
```json
{{
  "relationships": [
    {{
      "source": "exact entity name from list",
      "target": "exact entity name from list",
      "type": "RELATIONSHIP_TYPE",
      "description": "brief description of the relationship",
      "evidence": "relevant quote from the text"
    }}
  ]
}}
```

If no relationships are found, return: {{"relationships": []}}'''


RELATIONSHIP_EXTRACTION_PROMPT_JA = '''あなたは学術論文からエンティティ間の関係を抽出する専門家です。

## タスク
与えられたエンティティリストとソーステキストから、エンティティ間の意味のある関係をすべて抽出してください。

## エンティティ
{entities}

## 関係タイプ
- USES: AがBを使用（例：「BERTはアテンション機構を使用」）
- EXTENDS: AがBを拡張（例：「RoBERTaはBERTを拡張」）
- IMPROVES: AがBを改善
- IMPLEMENTS: AがBを実装
- BASED_ON: AがBに基づく
- COMPARES: AとBを比較
- OUTPERFORMS: AがBを上回る
- EVALUATES_ON: AをB（データセット/ベンチマーク）で評価
- TRAINED_ON: AをB（データセット）で訓練
- FINE_TUNED_ON: AをBでファインチューン
- APPLIED_TO: AをB（タスク/ドメイン）に適用
- PROPOSED_BY: AをB（人/組織）が提案
- DEVELOPED_BY: AをBが開発
- AFFILIATED_WITH: AがBに所属
- COLLABORATED_WITH: AがBと協力
- CITES: AがBを引用
- REFERENCES: AがBを参照
- RELATED_TO: AとBが関連（一般）
- SIMILAR_TO: AとBが類似
- PART_OF: AがBの一部

## ルール
1. テキストに明示的または強く暗示されている関係のみを抽出
2. 各関係には明確なソースとターゲットエンティティが必要
3. 各関係の根拠（テキストからの引用）を提供
4. 学術的に意味のある関係に焦点を当てる
5. エンティティリストにある正確な名前を使用

## テキスト
{text}

## 出力形式
以下の構造のJSONオブジェクトを返してください：
```json
{{
  "relationships": [
    {{
      "source": "リストにある正確なエンティティ名",
      "target": "リストにある正確なエンティティ名",
      "type": "RELATIONSHIP_TYPE",
      "description": "関係の簡潔な説明",
      "evidence": "テキストからの関連する引用"
    }}
  ]
}}
```

関係が見つからない場合: {{"relationships": []}}'''


def format_entities_for_prompt(entities: list) -> str:
    """エンティティリストをプロンプト用にフォーマット
    
    Args:
        entities: エンティティのリスト（Entity or dict）
        
    Returns:
        フォーマットされた文字列
    """
    lines = []
    for e in entities:
        if hasattr(e, 'name'):
            name = e.name
            etype = e.type.value if hasattr(e.type, 'value') else str(e.type)
        else:
            name = e.get('name', '')
            etype = e.get('type', '')
        lines.append(f"- {name} ({etype})")
    return "\n".join(lines)


def get_relationship_prompt(
    entities: list,
    text: str,
    language: str = "en"
) -> str:
    """関係抽出プロンプトを取得
    
    Args:
        entities: エンティティリスト
        text: 抽出対象テキスト
        language: 言語コード ("en" or "ja")
        
    Returns:
        フォーマット済みプロンプト
    """
    entities_str = format_entities_for_prompt(entities)
    
    if language == "ja":
        return RELATIONSHIP_EXTRACTION_PROMPT_JA.format(
            entities=entities_str,
            text=text
        )
    return RELATIONSHIP_EXTRACTION_PROMPT.format(
        entities=entities_str,
        text=text
    )

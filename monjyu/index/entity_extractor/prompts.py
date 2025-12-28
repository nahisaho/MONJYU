# Entity Extraction Prompts
"""
FEAT-010: EntityExtractor プロンプトテンプレート

LLMエンティティ抽出用のプロンプト
"""


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


ENTITY_EXTRACTION_PROMPT_JA = '''あなたは学術論文からエンティティを抽出する専門家です。

## タスク
以下のテキストから重要なエンティティを抽出してください。研究分野を理解するのに役立つ学術的な概念に焦点を当ててください。

## エンティティタイプ
- RESEARCHER: 言及されている人物（研究者、著者、科学者）
- ORGANIZATION: 機関、企業、研究室、大学
- METHOD: アルゴリズム、手法、アプローチ、アーキテクチャ
- MODEL: 具体的なML/AIモデル（GPT-4, BERT, ResNet, Transformer等）
- DATASET: 使用または言及されているデータセット（ImageNet, COCO, GLUE等）
- METRIC: 評価指標（accuracy, F1, BLEU, perplexity等）
- TASK: 研究タスク（分類、翻訳、要約等）
- CONCEPT: 抽象的な概念、理論、現象
- TOOL: ツール、フレームワーク、ライブラリ（PyTorch, TensorFlow等）
- PAPER: 参照されている論文や著作

## ルール
1. 明確に言及されているエンティティのみを抽出（暗示されたものは不可）
2. 最も具体的な名前を使用（例：「GPT」ではなく「GPT-4」）
3. 簡潔な説明を提供（1-2文）
4. 既知の別名をリスト化（例：["BERT", "Bidirectional Encoder Representations from Transformers"]）
5. 学術的・研究的文脈に関連するエンティティのみを含める

## テキスト
{text}

## 出力形式
以下の構造のJSONオブジェクトを返してください：
```json
{{
  "entities": [
    {{
      "name": "エンティティ名",
      "type": "ENTITY_TYPE",
      "description": "エンティティの簡潔な説明",
      "aliases": ["別名1", "別名2"]
    }}
  ]
}}
```

エンティティが見つからない場合: {{"entities": []}}'''


ENTITY_MERGE_PROMPT = '''You are an expert at resolving entity references in academic papers.

## Task
Determine if the following two entities refer to the same concept and should be merged.

## Entity 1
- Name: {entity1_name}
- Type: {entity1_type}
- Description: {entity1_description}
- Aliases: {entity1_aliases}

## Entity 2
- Name: {entity2_name}
- Type: {entity2_type}
- Description: {entity2_description}
- Aliases: {entity2_aliases}

## Output Format
Return a JSON object:
```json
{{
  "should_merge": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}
```'''


def get_extraction_prompt(text: str, language: str = "en") -> str:
    """抽出プロンプトを取得
    
    Args:
        text: 抽出対象テキスト
        language: 言語コード ("en" or "ja")
        
    Returns:
        フォーマット済みプロンプト
    """
    if language == "ja":
        return ENTITY_EXTRACTION_PROMPT_JA.format(text=text)
    return ENTITY_EXTRACTION_PROMPT.format(text=text)


def get_merge_prompt(
    entity1_name: str,
    entity1_type: str,
    entity1_description: str,
    entity1_aliases: list,
    entity2_name: str,
    entity2_type: str,
    entity2_description: str,
    entity2_aliases: list,
) -> str:
    """マージ判定プロンプトを取得"""
    return ENTITY_MERGE_PROMPT.format(
        entity1_name=entity1_name,
        entity1_type=entity1_type,
        entity1_description=entity1_description,
        entity1_aliases=entity1_aliases,
        entity2_name=entity2_name,
        entity2_type=entity2_type,
        entity2_description=entity2_description,
        entity2_aliases=entity2_aliases,
    )

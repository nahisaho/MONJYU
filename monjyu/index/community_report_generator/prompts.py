# Community Report Generator Prompts
"""
FEAT-013: CommunityReportGenerator プロンプト

コミュニティレポート生成用のプロンプトテンプレート
"""

from typing import List, Dict, Any


# =============================================================================
# English Prompts
# =============================================================================

COMMUNITY_REPORT_SYSTEM_PROMPT_EN = """You are an AI assistant specialized in analyzing research communities and generating executive summaries.

Your task is to analyze a group of related entities (researchers, methods, datasets, etc.) and their relationships, then generate a comprehensive report summarizing the key themes and findings of this community.

The report should:
1. Identify the main research theme or topic
2. Highlight key findings and contributions
3. Note important relationships and collaborations
4. Assess the overall significance of the community

Output your response in valid JSON format."""

COMMUNITY_REPORT_USER_PROMPT_EN = """Analyze the following research community and generate a comprehensive report.

## Community Information
- Community ID: {community_id}
- Level: {level}
- Member Count: {member_count}

## Entities in this Community
{entities_section}

## Relationships
{relationships_section}

## Output Format
Generate a JSON response with the following structure:
```json
{{
    "title": "A concise title capturing the main theme (10 words or less)",
    "summary": "A 2-3 sentence executive summary of the community",
    "full_content": "A detailed paragraph (100-200 words) describing the community",
    "findings": [
        {{
            "id": "finding-1",
            "summary": "Brief summary of finding 1",
            "explanation": "Detailed explanation",
            "evidence": ["Evidence 1", "Evidence 2"]
        }},
        {{
            "id": "finding-2",
            "summary": "Brief summary of finding 2",
            "explanation": "Detailed explanation",
            "evidence": ["Evidence 1"]
        }}
    ],
    "rating": 7.5,
    "rating_explanation": "Explanation of why this rating was assigned"
}}
```

Guidelines for rating (0-10 scale):
- 9-10: Groundbreaking research with high impact
- 7-8: Significant contributions to the field
- 5-6: Solid research with moderate impact
- 3-4: Limited or niche contributions
- 1-2: Minimal significance

Generate 3-5 findings based on the entities and relationships provided."""


# =============================================================================
# Japanese Prompts
# =============================================================================

COMMUNITY_REPORT_SYSTEM_PROMPT_JA = """あなたは研究コミュニティを分析し、エグゼクティブサマリーを生成する専門のAIアシスタントです。

タスク：関連するエンティティ（研究者、手法、データセットなど）とその関係性のグループを分析し、このコミュニティの主要なテーマと発見をまとめた包括的なレポートを生成してください。

レポートには以下を含めてください：
1. 主な研究テーマまたはトピックの特定
2. 重要な発見と貢献のハイライト
3. 重要な関係性とコラボレーションの記載
4. コミュニティの全体的な重要性の評価

回答は有効なJSON形式で出力してください。"""

COMMUNITY_REPORT_USER_PROMPT_JA = """以下の研究コミュニティを分析し、包括的なレポートを生成してください。

## コミュニティ情報
- コミュニティID: {community_id}
- レベル: {level}
- メンバー数: {member_count}

## このコミュニティのエンティティ
{entities_section}

## 関係性
{relationships_section}

## 出力形式
以下の構造でJSONレスポンスを生成してください：
```json
{{
    "title": "主要テーマを捉えた簡潔なタイトル（10語以内）",
    "summary": "コミュニティの2-3文のエグゼクティブサマリー",
    "full_content": "コミュニティを説明する詳細な段落（100-200語）",
    "findings": [
        {{
            "id": "finding-1",
            "summary": "発見1の簡潔な要約",
            "explanation": "詳細な説明",
            "evidence": ["根拠1", "根拠2"]
        }},
        {{
            "id": "finding-2",
            "summary": "発見2の簡潔な要約",
            "explanation": "詳細な説明",
            "evidence": ["根拠1"]
        }}
    ],
    "rating": 7.5,
    "rating_explanation": "この評価を付けた理由の説明"
}}
```

評価のガイドライン（0-10スケール）：
- 9-10: 高いインパクトを持つ画期的な研究
- 7-8: 分野への重要な貢献
- 5-6: 中程度のインパクトを持つ堅実な研究
- 3-4: 限定的またはニッチな貢献
- 1-2: 最小限の重要性

提供されたエンティティと関係性に基づいて3-5個の発見を生成してください。"""


def get_prompts(language: str = "en") -> Dict[str, str]:
    """言語に応じたプロンプトを取得
    
    Args:
        language: 言語コード ("en" or "ja")
        
    Returns:
        system_prompt と user_prompt の辞書
    """
    if language == "ja":
        return {
            "system": COMMUNITY_REPORT_SYSTEM_PROMPT_JA,
            "user": COMMUNITY_REPORT_USER_PROMPT_JA,
        }
    return {
        "system": COMMUNITY_REPORT_SYSTEM_PROMPT_EN,
        "user": COMMUNITY_REPORT_USER_PROMPT_EN,
    }


def format_entities_section(entities: List[Dict[str, Any]]) -> str:
    """エンティティセクションをフォーマット
    
    Args:
        entities: エンティティ情報のリスト
        
    Returns:
        フォーマットされた文字列
    """
    if not entities:
        return "(No entities)"
    
    lines = []
    for i, entity in enumerate(entities, 1):
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "UNKNOWN")
        description = entity.get("description", "")
        
        line = f"{i}. **{name}** ({entity_type})"
        if description:
            line += f"\n   Description: {description}"
        lines.append(line)
    
    return "\n".join(lines)


def format_relationships_section(relationships: List[Dict[str, Any]]) -> str:
    """関係性セクションをフォーマット
    
    Args:
        relationships: 関係性情報のリスト
        
    Returns:
        フォーマットされた文字列
    """
    if not relationships:
        return "(No relationships)"
    
    lines = []
    for i, rel in enumerate(relationships, 1):
        source = rel.get("source", "Unknown")
        target = rel.get("target", "Unknown")
        rel_type = rel.get("type", "RELATED_TO")
        description = rel.get("description", "")
        
        line = f"{i}. {source} --[{rel_type}]--> {target}"
        if description:
            line += f"\n   Description: {description}"
        lines.append(line)
    
    return "\n".join(lines)


def build_report_prompt(
    community_id: str,
    level: int,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    language: str = "en",
) -> Dict[str, str]:
    """レポート生成プロンプトを構築
    
    Args:
        community_id: コミュニティID
        level: コミュニティレベル
        entities: エンティティリスト
        relationships: 関係性リスト
        language: 言語コード
        
    Returns:
        system_prompt と user_prompt の辞書
    """
    prompts = get_prompts(language)
    
    entities_section = format_entities_section(entities)
    relationships_section = format_relationships_section(relationships)
    
    user_prompt = prompts["user"].format(
        community_id=community_id,
        level=level,
        member_count=len(entities),
        entities_section=entities_section,
        relationships_section=relationships_section,
    )
    
    return {
        "system": prompts["system"],
        "user": user_prompt,
    }

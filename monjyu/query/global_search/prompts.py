"""GlobalSearch prompts."""

# Map プロンプト（部分回答生成）
MAP_PROMPT_EN = """You are a helpful assistant analyzing research data.
Based on the following community report, provide information relevant to the question.
Focus only on information present in the report. If no relevant information exists, respond with "No relevant information found."

---Community Report---
Title: {title}
Summary: {summary}

Key Findings:
{findings}

Key Entities: {entities}
---End Report---

Question: {query}

Provide a concise answer based ONLY on the above community report:"""

MAP_PROMPT_JA = """あなたは研究データを分析するアシスタントです。
以下のコミュニティレポートに基づいて、質問に関連する情報を提供してください。
レポートに含まれる情報のみに焦点を当ててください。関連情報がない場合は「関連情報なし」と回答してください。

---コミュニティレポート---
タイトル: {title}
要約: {summary}

主な発見:
{findings}

主要エンティティ: {entities}
---レポート終了---

質問: {query}

上記のコミュニティレポートのみに基づいて簡潔に回答してください:"""


# Reduce プロンプト（統合回答生成）
REDUCE_PROMPT_EN = """You are a helpful assistant synthesizing research findings.
Based on the following partial answers from different research communities, provide a comprehensive answer to the question.

Question: {query}

---Partial Answers---
{partial_answers}
---End Partial Answers---

Synthesize the above information into a coherent, comprehensive answer. If there are conflicting findings, note them. If no useful information was found, indicate that."""

REDUCE_PROMPT_JA = """あなたは研究成果を統合するアシスタントです。
異なる研究コミュニティからの部分回答に基づいて、質問に対する包括的な回答を提供してください。

質問: {query}

---部分回答---
{partial_answers}
---部分回答終了---

上記の情報を統合して、一貫性のある包括的な回答を作成してください。矛盾する発見がある場合は指摘してください。有用な情報が見つからなかった場合はその旨を示してください。"""


def get_map_prompt(language: str = "auto") -> str:
    """Mapプロンプトを取得"""
    if language.lower() in ("ja", "japanese", "日本語"):
        return MAP_PROMPT_JA
    return MAP_PROMPT_EN


def get_reduce_prompt(language: str = "auto") -> str:
    """Reduceプロンプトを取得"""
    if language.lower() in ("ja", "japanese", "日本語"):
        return REDUCE_PROMPT_JA
    return REDUCE_PROMPT_EN

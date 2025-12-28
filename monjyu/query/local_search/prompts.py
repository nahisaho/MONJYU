"""LocalSearch prompts."""

# ローカル検索プロンプト（英語）
LOCAL_SEARCH_PROMPT_EN = """You are a helpful assistant answering questions based on specific entities and their relationships from a knowledge graph.

Based on the following entity and relationship information, provide a detailed answer to the question.

---Entities---
{entities}
---End Entities---

---Relationships---
{relationships}
---End Relationships---

---Supporting Text Chunks---
{chunks}
---End Text Chunks---

Question: {query}

Provide a comprehensive answer using ONLY the information above. If the information is insufficient, state what is known and what cannot be determined."""

# ローカル検索プロンプト（日本語）
LOCAL_SEARCH_PROMPT_JA = """あなたは知識グラフから得られた特定のエンティティとその関係性に基づいて質問に回答するアシスタントです。

以下のエンティティと関係性の情報に基づいて、質問に詳細に回答してください。

---エンティティ---
{entities}
---エンティティ終了---

---関係性---
{relationships}
---関係性終了---

---関連テキストチャンク---
{chunks}
---テキストチャンク終了---

質問: {query}

上記の情報のみを使用して包括的に回答してください。情報が不十分な場合は、判明していることと判断できないことを明示してください。"""


def get_local_search_prompt(language: str = "auto") -> str:
    """ローカル検索プロンプトを取得"""
    if language.lower() in ("ja", "japanese", "日本語"):
        return LOCAL_SEARCH_PROMPT_JA
    return LOCAL_SEARCH_PROMPT_EN

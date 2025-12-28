# Answer Synthesizer
"""
回答合成 - コンテキストからLLMで回答を生成

TASK-004-05: AnswerSynthesizer実装
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from monjyu.search.base import Citation, SearchHit, SynthesizedAnswer

if TYPE_CHECKING:
    pass


class LLMClientProtocol(Protocol):
    """LLMクライアントプロトコル"""

    @property
    def model_name(self) -> str:
        """モデル名"""
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """テキスト生成"""
        ...


class AnswerSynthesizer:
    """回答合成"""

    DEFAULT_SYSTEM_PROMPT = """
あなたは学術論文の専門家です。
与えられたコンテキスト情報に基づいて、ユーザーの質問に正確かつ簡潔に回答してください。

ルール:
1. コンテキストに含まれる情報のみを使用して回答してください
2. 情報が不十分な場合は、その旨を明示してください
3. 回答には必ず引用元（Citation）を含めてください（例: [1], [2]）
4. 学術的な正確性を最優先してください
"""

    USER_PROMPT_TEMPLATE = """
## コンテキスト
{context}

## 質問
{query}

## 回答形式
回答を記述した後、使用した引用元を [1], [2] のように示してください。
"""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        system_prompt: str | None = None,
    ):
        """
        Args:
            llm_client: LLMクライアント
            system_prompt: システムプロンプト（省略時はデフォルト使用）
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def synthesize(
        self,
        query: str,
        context: list[SearchHit],
        system_prompt: str | None = None,
    ) -> SynthesizedAnswer:
        """
        コンテキストから回答を合成

        Args:
            query: クエリ
            context: 検索ヒット（コンテキスト）
            system_prompt: システムプロンプト（オーバーライド用）

        Returns:
            合成された回答
        """
        if not context:
            return SynthesizedAnswer(
                answer="情報が見つかりませんでした。",
                citations=[],
                confidence=0.0,
                model=self.llm_client.model_name,
            )

        # コンテキスト構築
        context_text = self._build_context(context)

        # プロンプト構築
        system = system_prompt or self.system_prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_text, query=query
        )

        # LLM呼び出し
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system,
        )

        # 引用抽出
        answer_text, citations = self._extract_citations(response, context)

        # 信頼度推定（引用数とコンテキストスコアに基づく簡易推定）
        confidence = self._estimate_confidence(citations, context)

        return SynthesizedAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            model=self.llm_client.model_name,
        )

    def _build_context(self, hits: list[SearchHit]) -> str:
        """コンテキストテキストを構築"""
        parts = []
        for i, hit in enumerate(hits):
            parts.append(f"[{i + 1}] {hit.document_title or 'Document'}")
            parts.append(f"Score: {hit.score:.3f}")
            parts.append(hit.text)
            parts.append("")
        return "\n".join(parts)

    def _extract_citations(
        self, response: str, context: list[SearchHit]
    ) -> tuple[str, list[Citation]]:
        """回答から引用を抽出"""
        # [1], [2] などのパターンを検出
        citation_pattern = r"\[(\d+)\]"
        cited_indices = set(int(m) for m in re.findall(citation_pattern, response))

        citations = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(context):
                hit = context[idx - 1]
                # テキストスニペットを作成（最大200文字）
                snippet = hit.text[:200]
                if len(hit.text) > 200:
                    snippet += "..."

                citations.append(
                    Citation(
                        text_unit_id=hit.text_unit_id,
                        document_id=hit.document_id,
                        document_title=hit.document_title or "Unknown Document",
                        text_snippet=snippet,
                        relevance_score=hit.score,
                    )
                )

        return response, citations

    def _estimate_confidence(
        self, citations: list[Citation], context: list[SearchHit]
    ) -> float:
        """信頼度を推定"""
        if not citations:
            return 0.0

        # 引用されたコンテキストの平均スコア
        avg_score = sum(c.relevance_score for c in citations) / len(citations)

        # 引用カバレッジ（引用数 / コンテキスト数）
        coverage = min(len(citations) / max(len(context), 1), 1.0)

        # 信頼度 = 平均スコア * カバレッジ調整
        confidence = avg_score * (0.5 + 0.5 * coverage)

        return min(confidence, 1.0)


# === LLM Client Implementations ===


class OllamaLLMClient:
    """Ollama LLMクライアント"""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
    ):
        """
        Args:
            model: Ollamaモデル名
            host: Ollamaホスト
        """
        self._model = model
        self.host = host
        self._client = None

    @property
    def model_name(self) -> str:
        """モデル名"""
        return self._model

    @property
    def client(self):
        """Ollamaクライアント（遅延初期化）"""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise RuntimeError(
                    "ollama package not installed. Run: pip install ollama"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """テキスト生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = {}
        if max_tokens:
            options["num_predict"] = max_tokens

        response = self.client.chat(
            model=self._model,
            messages=messages,
            options=options if options else None,
        )

        return response["message"]["content"]


class OpenAILLMClient:
    """OpenAI LLMクライアント"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        """
        Args:
            model: OpenAIモデル名
            api_key: APIキー（省略時は環境変数から取得）
        """
        self._model = model
        self._api_key = api_key
        self._client = None

    @property
    def model_name(self) -> str:
        """モデル名"""
        return self._model

    @property
    def client(self):
        """OpenAIクライアント（遅延初期化）"""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """テキスト生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self._model, "messages": messages}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content


class AzureOpenAILLMClient:
    """Azure OpenAI LLMクライアント"""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
    ):
        """
        Args:
            endpoint: Azure OpenAIエンドポイント
            api_key: APIキー
            deployment_name: デプロイメント名
            api_version: APIバージョン
        """
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self._api_key = api_key
        self._client = None

    @property
    def model_name(self) -> str:
        """モデル名"""
        return self.deployment_name

    @property
    def client(self):
        """AzureOpenAIクライアント（遅延初期化）"""
        if self._client is None:
            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self._api_key,
                    api_version=self.api_version,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """テキスト生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self.deployment_name, "messages": messages}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content


class MockLLMClient:
    """モックLLMクライアント（テスト用）"""

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Args:
            responses: クエリパターンに対する応答マッピング
        """
        self._responses = responses or {}
        self._default_response = "This is a mock response. [1]"

    @property
    def model_name(self) -> str:
        """モデル名"""
        return "mock-llm"

    def set_response(self, pattern: str, response: str) -> None:
        """応答パターンを設定"""
        self._responses[pattern] = response

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """テキスト生成（モック）"""
        # パターンマッチング
        for pattern, response in self._responses.items():
            if pattern in prompt:
                return response

        return self._default_response

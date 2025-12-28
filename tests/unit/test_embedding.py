# Embedding Client Unit Tests
"""
Unit tests for embedding clients.
"""

from __future__ import annotations

import dataclasses
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from monjyu.embedding.base import EmbeddingClient, EmbeddingRecord
from monjyu.embedding.ollama import OllamaEmbeddingClient


class TestEmbeddingRecord:
    """EmbeddingRecordのテスト"""
    
    def test_create_record(self):
        """レコード作成"""
        record = EmbeddingRecord(
            id="emb_001",
            text_unit_id="tu_001",
            vector=[0.1, 0.2, 0.3],
            model="nomic-embed-text",
            dimensions=3,
        )
        
        assert record.id == "emb_001"
        assert record.text_unit_id == "tu_001"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.model == "nomic-embed-text"
        assert record.dimensions == 3
    
    def test_is_dataclass(self):
        """dataclassであること"""
        assert dataclasses.is_dataclass(EmbeddingRecord)
    
    def test_asdict(self):
        """辞書変換（dataclasses.asdict使用）"""
        record = EmbeddingRecord(
            id="emb_001",
            text_unit_id="tu_001",
            vector=[0.1, 0.2, 0.3],
            model="nomic-embed-text",
            dimensions=3,
        )
        
        data = dataclasses.asdict(record)
        
        assert data["id"] == "emb_001"
        assert data["text_unit_id"] == "tu_001"
        assert data["vector"] == [0.1, 0.2, 0.3]
        assert data["model"] == "nomic-embed-text"
        assert data["dimensions"] == 3


class TestOllamaEmbeddingClient:
    """OllamaEmbeddingClientのテスト"""
    
    def test_init_defaults(self):
        """デフォルト初期化"""
        client = OllamaEmbeddingClient()
        
        assert client.model_name == "nomic-embed-text"
        assert client.base_url == "http://localhost:11434"
    
    def test_init_custom_model(self):
        """カスタムモデル指定"""
        client = OllamaEmbeddingClient(
            model="mxbai-embed-large",
            base_url="http://custom:11434",
        )
        
        assert client.model_name == "mxbai-embed-large"
        assert client.base_url == "http://custom:11434"
    
    def test_init_timeout(self):
        """タイムアウト設定"""
        client = OllamaEmbeddingClient(timeout=120.0)
        assert client.timeout == 120.0
    
    def test_init_max_retries(self):
        """リトライ回数設定"""
        client = OllamaEmbeddingClient(max_retries=5)
        assert client.max_retries == 5


class TestAzureOpenAIEmbeddingClient:
    """AzureOpenAIEmbeddingClientのテスト"""
    
    def test_import(self):
        """インポート可能"""
        try:
            from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
            assert AzureOpenAIEmbeddingClient is not None
        except ImportError:
            pytest.skip("Azure OpenAI SDK not installed")

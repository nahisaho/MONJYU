# MONJYU Config Manager
"""
monjyu.api.config - 設定マネージャー

FEAT-007: Python API (MONJYU Facade)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from monjyu.api.base import (
    MONJYUConfig,
    SearchMode,
    IndexLevel,
)


class ConfigManager:
    """設定マネージャー"""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: MONJYUConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConfigManager:
        """YAMLファイルから読み込み"""
        manager = cls(path)
        manager.load()
        return manager

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ConfigManager:
        """辞書から作成"""
        manager = cls()
        manager._config = cls._parse_config(config_dict)
        return manager

    @classmethod
    def from_config(cls, config: MONJYUConfig) -> ConfigManager:
        """MONJYUConfigから作成"""
        manager = cls()
        manager._config = config
        return manager

    def load(self) -> MONJYUConfig:
        """設定を読み込み"""
        if not self.config_path or not self.config_path.exists():
            self._config = MONJYUConfig()
            return self._config

        with open(self.config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._config = self._parse_config(data)
        return self._config

    def save(self, path: str | Path | None = None) -> None:
        """設定を保存"""
        if self._config is None:
            return

        save_path = Path(path) if path else self.config_path
        if save_path is None:
            raise ValueError("No path specified for saving config")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._config_to_dict(self._config)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def _parse_config(data: dict[str, Any]) -> MONJYUConfig:
        """設定をパース"""
        # index_levels の変換
        index_levels_raw = data.get("index_levels", [0, 1])
        index_levels = []
        for level in index_levels_raw:
            if isinstance(level, IndexLevel):
                index_levels.append(level)
            elif isinstance(level, int):
                index_levels.append(IndexLevel(level))
            elif isinstance(level, str):
                index_levels.append(IndexLevel[level.upper()])

        # search_mode の変換
        search_mode_raw = data.get("default_search_mode", "lazy")
        if isinstance(search_mode_raw, SearchMode):
            search_mode = search_mode_raw
        else:
            search_mode = SearchMode(search_mode_raw)

        return MONJYUConfig(
            output_path=Path(data.get("output_path", "./output")),
            environment=data.get("environment", "local"),
            index_levels=index_levels,
            default_search_mode=search_mode,
            default_top_k=data.get("default_top_k", 10),
            chunk_size=data.get("chunk_size", 1200),
            chunk_overlap=data.get("chunk_overlap", 100),
            llm_model=data.get("llm_model", "llama3:8b-instruct-q4_K_M"),
            embedding_model=data.get("embedding_model", "nomic-embed-text"),
            ollama_base_url=data.get("ollama_base_url", "http://192.168.224.1:11434"),
            azure_openai_endpoint=data.get("azure_openai_endpoint"),
            azure_openai_api_key=data.get("azure_openai_api_key"),
            azure_search_endpoint=data.get("azure_search_endpoint"),
            azure_search_api_key=data.get("azure_search_api_key"),
        )

    @staticmethod
    def _config_to_dict(config: MONJYUConfig) -> dict[str, Any]:
        """MONJYUConfigを辞書に変換"""
        return {
            "output_path": str(config.output_path),
            "environment": config.environment,
            "index_levels": [level.value for level in config.index_levels],
            "default_search_mode": config.default_search_mode.value,
            "default_top_k": config.default_top_k,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "ollama_base_url": config.ollama_base_url,
            "azure_openai_endpoint": config.azure_openai_endpoint,
            "azure_openai_api_key": config.azure_openai_api_key,
            "azure_search_endpoint": config.azure_search_endpoint,
            "azure_search_api_key": config.azure_search_api_key,
        }

    @property
    def config(self) -> MONJYUConfig:
        """設定を取得"""
        if self._config is None:
            self._config = self.load()
        return self._config


def load_config(path: str | Path | None = None) -> MONJYUConfig:
    """設定を読み込むヘルパー関数"""
    manager = ConfigManager(path)
    return manager.load()

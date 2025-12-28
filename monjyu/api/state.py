# MONJYU State Manager
"""
monjyu.api.state - 状態マネージャー

FEAT-007: Python API (MONJYU Facade)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monjyu.api.base import (
    MONJYUStatus,
    IndexStatus,
    IndexLevel,
)


class StateManager:
    """状態マネージャー"""

    STATE_FILE = "monjyu_state.json"

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.state_file = output_path / self.STATE_FILE
        self._status: MONJYUStatus | None = None

    def load(self) -> MONJYUStatus:
        """状態を読み込み"""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._status = self._parse_status(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # 破損したファイルの場合はデフォルト値
                self._status = MONJYUStatus()
        else:
            self._status = MONJYUStatus()

        return self._status

    def save(self) -> None:
        """状態を保存"""
        if self._status is None:
            return

        self.output_path.mkdir(parents=True, exist_ok=True)

        data = self._status_to_dict(self._status)

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def update(self, **kwargs: Any) -> MONJYUStatus:
        """状態を更新"""
        if self._status is None:
            self.load()

        for key, value in kwargs.items():
            if hasattr(self._status, key):
                setattr(self._status, key, value)

        self.save()
        return self._status

    def reset(self) -> MONJYUStatus:
        """状態をリセット"""
        self._status = MONJYUStatus()
        self.save()
        return self._status

    @staticmethod
    def _parse_status(data: dict[str, Any]) -> MONJYUStatus:
        """辞書から状態をパース"""
        # index_levels_built の変換
        levels_raw = data.get("index_levels_built", [])
        levels = []
        for level in levels_raw:
            if isinstance(level, IndexLevel):
                levels.append(level)
            elif isinstance(level, int):
                levels.append(IndexLevel(level))
            elif isinstance(level, str):
                levels.append(IndexLevel[level.upper()])

        # index_status の変換
        status_raw = data.get("index_status", "not_built")
        if isinstance(status_raw, IndexStatus):
            index_status = status_raw
        else:
            index_status = IndexStatus(status_raw)

        return MONJYUStatus(
            index_status=index_status,
            index_levels_built=levels,
            document_count=data.get("document_count", 0),
            text_unit_count=data.get("text_unit_count", 0),
            noun_phrase_count=data.get("noun_phrase_count", 0),
            community_count=data.get("community_count", 0),
            citation_edge_count=data.get("citation_edge_count", 0),
            last_error=data.get("last_error"),
        )

    @staticmethod
    def _status_to_dict(status: MONJYUStatus) -> dict[str, Any]:
        """状態を辞書に変換"""
        return {
            "index_status": status.index_status.value,
            "index_levels_built": [level.value for level in status.index_levels_built],
            "document_count": status.document_count,
            "text_unit_count": status.text_unit_count,
            "noun_phrase_count": status.noun_phrase_count,
            "community_count": status.community_count,
            "citation_edge_count": status.citation_edge_count,
            "last_error": status.last_error,
        }

    @property
    def status(self) -> MONJYUStatus:
        """状態を取得"""
        if self._status is None:
            self.load()
        return self._status

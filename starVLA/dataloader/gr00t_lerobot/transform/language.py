import json
import random
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from .base import ModalityTransform


class LanguageParaphraseTransform(ModalityTransform):
    """Train-only instruction paraphrasing with reviewed, exact-match variants."""

    paraphrase_json: str = Field(..., description="Path to the paraphrase JSON file.")
    p: float = Field(default=0.5, ge=0.0, le=1.0, description="Replacement probability.")

    _enabled: bool = PrivateAttr(default=False)
    _paraphrases: dict[str, list[str]] | None = PrivateAttr(default=None)

    def _resolve_config_path(self) -> Path:
        path = Path(self.paraphrase_json)
        if path.is_absolute() and path.exists():
            return path
        if path.exists():
            return path

        repo_root = Path(__file__).resolve().parents[4]
        repo_path = repo_root / path
        if repo_path.exists():
            return repo_path

        raise FileNotFoundError(f"Paraphrase config not found: {self.paraphrase_json}")

    def _load_paraphrases(self) -> None:
        if self._paraphrases is not None:
            return

        config_path = self._resolve_config_path()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._enabled = bool(config.get("enabled", False))

        flattened: dict[str, list[str]] = {}
        for dataset_map in config.get("task_paraphrases", {}).values():
            for original, candidates in dataset_map.items():
                if not isinstance(original, str) or not isinstance(candidates, list):
                    continue
                valid_candidates = [
                    str(candidate)
                    for candidate in candidates
                    if isinstance(candidate, str) and candidate.strip() and candidate != original
                ]
                if valid_candidates:
                    flattened[original] = valid_candidates

        self._paraphrases = flattened

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training or self.p <= 0:
            return data

        self._load_paraphrases()
        if not self._enabled or not self._paraphrases:
            return data

        for key in self.apply_to:
            if key not in data:
                continue
            data[key] = self._paraphrase_value(data[key])
        return data

    def _paraphrase_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._maybe_paraphrase_text(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._maybe_paraphrase_text(item) for item in value)
        if isinstance(value, str):
            return self._maybe_paraphrase_text(value)
        return value

    def _maybe_paraphrase_text(self, text: Any) -> Any:
        if not isinstance(text, str):
            return text
        candidates = self._paraphrases.get(text) if self._paraphrases else None
        if not candidates or random.random() >= self.p:
            return text
        return random.choice(candidates)

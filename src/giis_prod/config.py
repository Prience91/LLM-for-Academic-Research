from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CANONICAL_PDF_DIR = "RAG Corpus"
_LEGACY_PDF_DIRS = {"Scientific papers", "data/pdfs", "data\\pdfs", "Papers_RAG", "Papers_RAG_V17_Turbo"}
_CANONICAL_TRAINING_PDF_DIR = "Training Corpus"
_LEGACY_TRAINING_PDF_DIRS = {"training set", "Papers_Training", "Papers_Training_V17_Turbo"}


def _coerce_like(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    root: Path
    data: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        path = Path(config_path) if config_path else project_root / "config" / "settings.toml"
        path = path.resolve()
        with path.open("rb") as stream:
            data = tomllib.load(stream)

        for env_name, raw_value in os.environ.items():
            if not env_name.startswith("GIIS__"):
                continue
            parts = env_name.removeprefix("GIIS__").lower().split("__")
            if len(parts) != 2:
                raise ValueError(f"Invalid GIIS override name: {env_name}")
            section, key = parts
            if section not in data or key not in data[section]:
                raise KeyError(f"Unknown configuration override: {env_name}")
            data[section][key] = _coerce_like(raw_value, data[section][key])

        # Normalize copied configurations to the canonical operator-facing source directories.
        # Papers_Imported is deliberately absent: it is a dormant vault, never an ingestion source.
        configured_pdf_dir = str(data.get("paths", {}).get("pdf_dir", ""))
        if configured_pdf_dir in _LEGACY_PDF_DIRS:
            data["paths"]["pdf_dir"] = _CANONICAL_PDF_DIR
        configured_training_dir = str(data.get("paths", {}).get("training_pdf_dir", ""))
        if configured_training_dir in _LEGACY_TRAINING_PDF_DIRS:
            data["paths"]["training_pdf_dir"] = _CANONICAL_TRAINING_PDF_DIR

        return cls(root=project_root, data=data)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data[name])

    def get(self, dotted_key: str) -> Any:
        section, key = dotted_key.split(".", maxsplit=1)
        return self.data[section][key]

    def path(self, dotted_key: str) -> Path:
        value = Path(str(self.get(dotted_key)))
        return value if value.is_absolute() else (self.root / value).resolve()

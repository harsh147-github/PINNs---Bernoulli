"""Load ``configs/nozzle.yaml`` into a plain nested dictionary.

Every number used anywhere in the package comes from this file, so a run is
fully described by (config file, random seed, code version).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "nozzle.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read the YAML config (defaults to ``configs/nozzle.yaml``)."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

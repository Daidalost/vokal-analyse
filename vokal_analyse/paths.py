"""Where models, the separation cache and reports live (per user)."""
from __future__ import annotations

import os
from pathlib import Path


def base_dir() -> Path:
    env = os.environ.get("VOKAL_ANALYSE_HOME")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VokalAnalyse"
    return Path.home() / ".vokal_analyse"


def models_dir() -> Path:
    return base_dir() / "models"


def cache_dir() -> Path:
    return base_dir() / "stems"


def reports_dir() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.exists() else Path.home()) / "Vokal Analyse"

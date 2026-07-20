from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@lru_cache(maxsize=1)
def get_model_name() -> str:
    return os.getenv("GROUNDTRUTH_MODEL", "gemini-1.5-flash")


@lru_cache(maxsize=1)
def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

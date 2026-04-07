"""helpers for guided-start catalog loading."""

import json
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

from fastapi import HTTPException

from backbone.models.prompt_artifact import GuidedStartCatalog

GUIDED_START_DATA_DIR = Path(__file__).parent / "data" / "guided_start"
GUIDED_START_CATALOG_PATH = GUIDED_START_DATA_DIR / "catalog.json"


@lru_cache(maxsize=1)
def load_guided_start_catalog() -> GuidedStartCatalog:
    """Load and validate the guided-start catalog."""
    if not GUIDED_START_CATALOG_PATH.exists():
        raise HTTPException(status_code=500, detail="Guided start catalog is missing.")
    try:
        return GuidedStartCatalog.model_validate(json.loads(GUIDED_START_CATALOG_PATH.read_text()))
    except JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Guided start catalog is invalid: {exc.msg}")

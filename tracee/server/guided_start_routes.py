"""api routes for guided-start catalog."""

from fastapi import APIRouter

from backbone.models.prompt_artifact import GuidedStartCatalog
from server.guided_start_service import load_guided_start_catalog

router = APIRouter()


@router.get("/guided-start/catalog", response_model=GuidedStartCatalog)
def get_guided_start_catalog() -> GuidedStartCatalog:
    """Return the guided-start reference catalog."""
    return load_guided_start_catalog()

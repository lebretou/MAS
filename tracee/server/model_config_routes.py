"""API routes for saved model configurations.

Provides CRUD endpoints for managing reusable model configurations.
"""

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backbone.models.saved_model_config import (
    SavedModelConfig,
    SavedModelConfigCreate,
    SavedModelConfigUpdate,
)
from backbone.utils.identifiers import generate_config_id, utc_timestamp

router = APIRouter()

# Storage directory
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
MODEL_CONFIGS_DIR = Path(os.getenv("MODEL_CONFIGS_DIR", str(DEFAULT_DATA_DIR / "model_configs")))


# --- Helper Functions ---


def _ensure_dir() -> None:
    """Ensure the model configs directory exists."""
    MODEL_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def _get_config_file(config_id: str) -> Path:
    """Get path to a config file."""
    return MODEL_CONFIGS_DIR / f"{config_id}.json"


def _load_config(config_id: str) -> SavedModelConfig:
    """Load a model configuration from disk."""
    config_file = _get_config_file(config_id)
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {config_id}")
    return SavedModelConfig.model_validate_json(config_file.read_text())


def _save_config(config: SavedModelConfig) -> None:
    """Save a model configuration to disk."""
    _ensure_dir()
    config_file = _get_config_file(config.config_id)
    config_file.write_text(config.model_dump_json(indent=2))


def _delete_config_file(config_id: str) -> None:
    """Delete a model configuration file."""
    config_file = _get_config_file(config_id)
    if config_file.exists():
        config_file.unlink()


def _list_all_configs() -> list[SavedModelConfig]:
    """List all saved model configurations."""
    _ensure_dir()
    configs = []
    
    for config_file in MODEL_CONFIGS_DIR.glob("*.json"):
        try:
            config = SavedModelConfig.model_validate_json(config_file.read_text())
            configs.append(config)
        except Exception:
            # Skip invalid config files
            continue
    
    # Sort by updated_at descending (newest first), with default configs first
    configs.sort(key=lambda c: (not c.is_default, c.updated_at), reverse=True)
    return configs


# --- API Endpoints ---


@router.get("/model-configs")
def list_model_configs() -> list[SavedModelConfig]:
    """List all saved model configurations."""
    return _list_all_configs()


@router.post("/model-configs")
def create_model_config(request: SavedModelConfigCreate) -> SavedModelConfig:
    """Create a new model configuration."""
    now = utc_timestamp()
    
    # If this is marked as default, unmark any existing defaults
    if request.is_default:
        for existing in _list_all_configs():
            if existing.is_default:
                existing.is_default = False
                existing.updated_at = now
                _save_config(existing)
    
    config = SavedModelConfig(
        config_id=generate_config_id(),
        name=request.name,
        description=request.description,
        provider=request.provider,
        model_name=request.model_name,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        extra_params=request.extra_params,
        is_default=request.is_default,
        tags=request.tags,
        created_at=now,
        updated_at=now,
    )
    
    _save_config(config)
    return config


@router.get("/model-configs/{config_id}")
def get_model_config(config_id: str) -> SavedModelConfig:
    """Get a specific model configuration."""
    return _load_config(config_id)


@router.patch("/model-configs/{config_id}")
def update_model_config(config_id: str, request: SavedModelConfigUpdate) -> SavedModelConfig:
    """Update an existing model configuration."""
    config = _load_config(config_id)
    now = utc_timestamp()
    
    # If this is being marked as default, unmark any existing defaults
    if request.is_default and not config.is_default:
        for existing in _list_all_configs():
            if existing.is_default and existing.config_id != config_id:
                existing.is_default = False
                existing.updated_at = now
                _save_config(existing)
    
    # Update fields that were provided
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)
    
    config.updated_at = now
    _save_config(config)
    return config


@router.delete("/model-configs/{config_id}")
def delete_model_config(config_id: str) -> dict:
    """Delete a model configuration."""
    # Verify it exists
    _load_config(config_id)
    
    _delete_config_file(config_id)
    return {"deleted": config_id}


@router.get("/model-configs/default")
def get_default_model_config() -> SavedModelConfig:
    """Get the default model configuration, if one is set."""
    for config in _list_all_configs():
        if config.is_default:
            return config
    
    raise HTTPException(status_code=404, detail="No default model configuration set")


@router.post("/model-configs/{config_id}/set-default")
def set_default_model_config(config_id: str) -> SavedModelConfig:
    """Set a model configuration as the default."""
    config = _load_config(config_id)
    now = utc_timestamp()
    
    # Unmark any existing defaults
    for existing in _list_all_configs():
        if existing.is_default and existing.config_id != config_id:
            existing.is_default = False
            existing.updated_at = now
            _save_config(existing)
    
    # Mark this one as default
    config.is_default = True
    config.updated_at = now
    _save_config(config)
    
    return config

"""FastAPI application for serving trace and prompt data."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.agent_routes import router as agent_router
from server.graph_routes import router as graph_router
from server.guided_start_routes import router as guided_start_router
from server.trace_routes import router as trace_router
from server.prompt_routes import router as prompt_router
from server.playground_routes import router as playground_router
from server.db import init_all
from server.model_config_routes import router as model_config_router
from server.trace_db import TRACE_DB_PATH

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

# CORS origins - configurable via environment variable
# Use comma-separated values for multiple origins, or "*" for all (development only)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on startup."""
    init_all()
    yield


app = FastAPI(
    title="Tracee API",
    description="API server for MAS trace data, prompt management, and playground",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routes
app.include_router(trace_router, prefix="/api")
app.include_router(prompt_router, prefix="/api")
app.include_router(playground_router, prefix="/api")
app.include_router(guided_start_router, prefix="/api")
app.include_router(model_config_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(graph_router, prefix="/api")


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.2.0",
        "trace_db": str(TRACE_DB_PATH),
        "endpoints": {
            "traces": "/api/traces",
            "prompts": "/api/prompts",
            "playground": "/api/playground/run",
            "guided_start": "/api/guided-start/catalog",
            "model_configs": "/api/model-configs",
            "agents": "/api/agents",
            "graphs": "/api/graphs",
        },
    }


ui_build = Path(__file__).parent.parent / "playground-ui" / "dist"
ui_root = ui_build.resolve()
if ui_build.exists():
    assets_dir = ui_build / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")


@app.get("/")
def root():
    """Serve the UI root when built, otherwise return health."""
    if ui_build.exists():
        return FileResponse(ui_build / "index.html")
    return health()


@app.get("/{path:path}")
def spa(path: str):
    """Serve built UI assets and SPA routes."""
    if not ui_build.exists():
        return health()
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    requested = (ui_build / path).resolve()
    if path and not requested.is_relative_to(ui_root):
        raise HTTPException(status_code=404, detail="Not Found")
    if path and requested.is_file():
        return FileResponse(requested)
    return FileResponse(ui_build / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

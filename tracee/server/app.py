"""FastAPI application for serving trace and prompt data."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import router as trace_router
from server.prompt_routes import router as prompt_router
from server.playground_routes import router as playground_router
from server.db import init_all
from server.model_config_routes import router as model_config_router
from server.trace_db import TRACE_DB_PATH

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

app = FastAPI(
    title="Tracee API",
    description="API server for MAS trace data, prompt management, and playground",
    version="0.2.0",
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routes
app.include_router(trace_router, prefix="/api")
app.include_router(prompt_router, prefix="/api")
app.include_router(playground_router, prefix="/api")
app.include_router(model_config_router, prefix="/api")


@app.on_event("startup")
def startup():
    """initialize database tables."""
    init_all()


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.2.0",
        "trace_db": str(TRACE_DB_PATH),
        "endpoints": {
            "traces": "/api/traces",
            "prompts": "/api/prompts",
            "playground": "/api/playground/run",
            "model_configs": "/api/model-configs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

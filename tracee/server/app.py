"""FastAPI application for serving trace and prompt data."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import router as trace_router
from server.prompt_routes import router as prompt_router, PROMPTS_DIR

# configurable traces directory via environment variable (stored alongside prompts in server/data/)
from server.routes import TRACES_DIR

app = FastAPI(
    title="Tracee API",
    description="API server for serving MAS trace data and prompt management",
    version="0.1.0",
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


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "traces_dir": str(TRACES_DIR),
        "prompts_dir": str(PROMPTS_DIR),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

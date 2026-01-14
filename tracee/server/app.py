"""FastAPI application for serving trace data."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import router

# configurable traces directory via environment variable
DEFAULT_TRACES_DIR = Path(__file__).parent.parent / "sample_mas" / "backend" / "outputs" / "traces"
TRACES_DIR = Path(os.getenv("TRACES_DIR", str(DEFAULT_TRACES_DIR)))

app = FastAPI(
    title="Tracee API",
    description="API server for serving MAS trace data",
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
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "traces_dir": str(TRACES_DIR)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

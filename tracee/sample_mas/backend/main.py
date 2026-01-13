from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import os
import uuid
import json
from typing import Optional
from backend.graph.workflow import run_analysis_workflow_async
from backend.telemetry.config import setup_telemetry

# initialize FastAPI app
app = FastAPI(title="Data Analysis Multi-Agent System", version="1.0.0")

# setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# setup directories
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# store session data in memory (in production, use a database)
sessions = {}

# set up basic startup logging and telemetry
@app.on_event("startup")
async def startup_event():
    """initialize startup output."""
    print("=" * 60)
    print("Starting Data Analysis Multi-Agent System")
    print("=" * 60)
    setup_telemetry()


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    session_id: str
    query: str


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    summary: str
    plots: list[str] = []
    code: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Data Analysis Multi-Agent System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "analyze": "POST /analyze",
            "results": "GET /results/{session_id}",
            "outputs": "GET /outputs/{filename}"
        }
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file (CSV, Excel, or JSON).
    
    Args:
        file: The uploaded file
        
    Returns:
        Session information with dataset metadata
    """
    # generate session ID
    session_id = str(uuid.uuid4())
    
    # save uploaded file
    file_extension = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}{file_extension}")
    
    # read file content
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # load dataset based on file type
    try:
        if file_extension == ".csv":
            dataset = pd.read_csv(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            dataset = pd.read_excel(file_path)
        elif file_extension == ".json":
            dataset = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # store in session
        sessions[session_id] = {
            "dataset": dataset,
            "file_path": file_path,
            "filename": file.filename,
            "uploaded_at": pd.Timestamp.now().isoformat()
        }
        
        # return dataset info
        return {
            "session_id": session_id,
            "filename": file.filename,
            "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
            "columns": list(dataset.columns),
            "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()}
        }
        
    except Exception as e:
        # clean up file if loading failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Run analysis workflow on uploaded dataset.
    
    Args:
        request: Analysis request with session_id and query
        
    Returns:
        Analysis results including summary and plots
    """
    # check if session exists
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a dataset first.")
    
    session = sessions[request.session_id]
    dataset = session["dataset"]
    
    try:
        # run workflow
        result = await run_analysis_workflow_async(
            dataset=dataset,
            query=request.query,
            dataset_path=session["filename"],
            session_id=request.session_id
        )
        
        if result.get("success"):
            # extract plot filenames
            plots = result.get("execution_result", {}).get("plots", [])
            
            return AnalysisResponse(
                success=True,
                summary=result.get("final_summary", ""),
                plots=plots,
                code=result.get("generated_code", None)
            )
        else:
            return AnalysisResponse(
                success=False,
                summary=result.get("final_summary", ""),
                error=result.get("error", "Unknown error"),
                plots=[],
                code=result.get("generated_code", None)
            )
            
    except Exception as e:
        return AnalysisResponse(
            success=False,
            summary=f"Analysis failed: {str(e)}",
            error=str(e),
            plots=[],
            code=None
        )


@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get information about a session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Session information
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    dataset = session["dataset"]
    
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "uploaded_at": session["uploaded_at"],
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "columns": list(dataset.columns)
    }


@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """Serve generated output files (plots).
    
    Args:
        filename: The output filename
        
    Returns:
        The file as a response
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated files.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Confirmation message
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # delete uploaded file
    if os.path.exists(session["file_path"]):
        os.remove(session["file_path"])
    
    # remove from sessions
    del sessions[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(sessions)
    }


# mount frontend static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

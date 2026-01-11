import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

# Load environment variables
load_dotenv()


def get_langsmith_config():
    """Configure LangSmith tracing."""
    return {
        "project_name": os.getenv("LANGSMITH_PROJECT", "data-analysis-agents"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
    }


def get_langfuse_handler():
    """Get Langfuse callback handler for tracing.
    
    The CallbackHandler reads credentials from environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not public_key or not secret_key:
        print("Warning: Langfuse credentials not found. Telemetry will be disabled.")
        return None
    
    try:
        # CallbackHandler reads from environment variables automatically
        return LangfuseCallbackHandler()
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse handler: {e}")
        return None


def get_callbacks():
    """Get all configured callbacks for LangChain."""
    callbacks = []
    
    # Add Langfuse handler if configured
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        callbacks.append(langfuse_handler)
    
    return callbacks


def setup_telemetry():
    """Setup telemetry for the application."""
    # Set LangSmith environment variables
    config = get_langsmith_config()
    if config["tracing_enabled"] and config["api_key"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config["project_name"]
        os.environ["LANGCHAIN_API_KEY"] = config["api_key"]
        print(f"✓ LangSmith tracing enabled for project: {config['project_name']}")
    else:
        print("⚠ LangSmith tracing disabled - API key not found")
    
    # Test Langfuse connection
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        print("✓ Langfuse callback handler configured")
    else:
        print("⚠ Langfuse callback handler not configured")
    
    return {
        "langsmith": config,
        "langfuse_handler": langfuse_handler,
    }

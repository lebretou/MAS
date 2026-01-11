import io
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr
import traceback
import os
import uuid
from typing import Any

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')


def execute_code_safely(code: str, dataset: pd.DataFrame, output_dir: str) -> dict:
    """Execute Python code in a sandboxed environment.
    
    Args:
        code: The Python code to execute
        dataset: The pandas DataFrame to make available
        output_dir: Directory to save plot outputs
        
    Returns:
        A dictionary with execution results
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "error": None,
        "plots": [],
        "variables": {}
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Prepare restricted globals
    safe_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
        },
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'df': dataset.copy(),  # Provide a copy to avoid modifying original
        'dataset': dataset.copy(),
    }
    
    # Import sklearn and scipy if needed
    try:
        import sklearn
        safe_globals['sklearn'] = sklearn
    except ImportError:
        pass
    
    try:
        import scipy
        safe_globals['scipy'] = scipy
    except ImportError:
        pass
    
    # Local variables to capture
    local_vars = {}
    
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Execute the code
            exec(code, safe_globals, local_vars)
            
            # Save any matplotlib figures
            figures = [plt.figure(n) for n in plt.get_fignums()]
            for i, fig in enumerate(figures):
                plot_filename = f"plot_{uuid.uuid4().hex[:8]}_{i}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                result["plots"].append(plot_filename)
            
            # Close all figures to free memory
            plt.close('all')
            
            result["success"] = True
            
    except Exception as e:
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        result["stderr"] += traceback.format_exc()
    
    # Capture output
    result["stdout"] = stdout_buffer.getvalue()
    result["stderr"] += stderr_buffer.getvalue()
    
    # Capture interesting variables (avoid large objects)
    for key, value in local_vars.items():
        if not key.startswith('_'):
            try:
                # Only capture simple types and small summaries
                if isinstance(value, (int, float, str, bool)):
                    result["variables"][key] = value
                elif isinstance(value, (list, tuple)) and len(value) < 10:
                    result["variables"][key] = str(value)
                elif isinstance(value, dict) and len(value) < 10:
                    result["variables"][key] = str(value)
                elif isinstance(value, pd.DataFrame):
                    result["variables"][key] = f"DataFrame(shape={value.shape})"
                elif isinstance(value, pd.Series):
                    result["variables"][key] = f"Series(length={len(value)})"
                elif isinstance(value, np.ndarray):
                    result["variables"][key] = f"ndarray(shape={value.shape})"
            except:
                pass
    
    return result

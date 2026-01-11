import ast
import pandas as pd
from langchain_core.tools import tool
from typing import Any


@tool
def validate_python_code(code: str) -> dict:
    """Validate Python code syntax and check for potentially dangerous operations.
    
    Args:
        code: The Python code string to validate
        
    Returns:
        A dictionary with validation results
    """
    result = {
        "is_valid": False,
        "errors": [],
        "warnings": []
    }
    
    # Check for syntax errors
    try:
        ast.parse(code)
        result["is_valid"] = True
    except SyntaxError as e:
        result["errors"].append(f"Syntax error: {str(e)}")
        return result
    
    # Check for dangerous operations
    dangerous_keywords = ['__import__', 'eval', 'exec', 'compile', 'open', 'file', 
                          'input', 'raw_input', 'execfile', '__builtins__']
    
    for keyword in dangerous_keywords:
        if keyword in code:
            result["warnings"].append(f"Warning: potentially dangerous operation '{keyword}' detected")
    
    # Check for allowed imports
    allowed_modules = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy', 'pd', 'np', 'plt', 'sns']
    
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_base = alias.name.split('.')[0]
                if module_base not in allowed_modules:
                    result["warnings"].append(f"Warning: import of '{alias.name}' may not be allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_base = node.module.split('.')[0]
                if module_base not in allowed_modules:
                    result["warnings"].append(f"Warning: import from '{node.module}' may not be allowed")
    
    return result


@tool
def check_dataset_columns(dataset: pd.DataFrame, columns: list[str]) -> dict:
    """Check if specified columns exist in the dataset.
    
    Args:
        dataset: The pandas DataFrame
        columns: List of column names to check
        
    Returns:
        A dictionary indicating which columns exist
    """
    dataset_columns = set(dataset.columns)
    result = {
        "all_exist": all(col in dataset_columns for col in columns),
        "existing_columns": [col for col in columns if col in dataset_columns],
        "missing_columns": [col for col in columns if col not in dataset_columns],
        "available_columns": list(dataset_columns)
    }
    return result


@tool
def validate_analysis_plan(plan: str) -> dict:
    """Validate an analysis plan to ensure it's well-structured.
    
    Args:
        plan: The analysis plan text
        
    Returns:
        A dictionary with validation results
    """
    result = {
        "is_valid": True,
        "issues": [],
        "suggestions": []
    }
    
    # Check if plan is not empty
    if not plan or len(plan.strip()) < 20:
        result["is_valid"] = False
        result["issues"].append("Plan is too short or empty")
        return result
    
    # Check for key elements
    plan_lower = plan.lower()
    
    key_elements = {
        "data loading": ["load", "read", "data", "dataset"],
        "analysis": ["analyze", "calculate", "compute", "regression", "correlation"],
        "visualization": ["plot", "visualize", "chart", "graph", "figure"]
    }
    
    for element_name, keywords in key_elements.items():
        if not any(keyword in plan_lower for keyword in keywords):
            result["suggestions"].append(f"Consider adding {element_name} steps")
    
    return result


@tool
def suggest_visualizations(dataset: pd.DataFrame, query: str) -> dict:
    """Suggest appropriate visualizations based on dataset and query.
    
    Args:
        dataset: The pandas DataFrame
        query: The user's query
        
    Returns:
        A dictionary with visualization suggestions
    """
    numeric_cols = list(dataset.select_dtypes(include=['number']).columns)
    categorical_cols = list(dataset.select_dtypes(include=['object', 'category']).columns)
    
    suggestions = {
        "recommended_plots": [],
        "reasoning": []
    }
    
    query_lower = query.lower()
    
    # Suggest based on query keywords
    if any(word in query_lower for word in ['correlation', 'relationship', 'compare']):
        if len(numeric_cols) >= 2:
            suggestions["recommended_plots"].append("scatter plot")
            suggestions["recommended_plots"].append("correlation heatmap")
            suggestions["reasoning"].append("Query mentions relationships between variables")
    
    if any(word in query_lower for word in ['distribution', 'spread', 'histogram']):
        suggestions["recommended_plots"].append("histogram")
        suggestions["recommended_plots"].append("box plot")
        suggestions["reasoning"].append("Query asks about data distribution")
    
    if any(word in query_lower for word in ['trend', 'time', 'series']):
        suggestions["recommended_plots"].append("line plot")
        suggestions["reasoning"].append("Query mentions temporal patterns")
    
    if any(word in query_lower for word in ['regression', 'predict', 'model']):
        suggestions["recommended_plots"].append("scatter plot with regression line")
        suggestions["recommended_plots"].append("residual plot")
        suggestions["reasoning"].append("Query involves predictive modeling")
    
    # Default suggestions based on data types
    if not suggestions["recommended_plots"]:
        if len(numeric_cols) >= 2:
            suggestions["recommended_plots"].append("scatter plot")
        if len(numeric_cols) >= 1:
            suggestions["recommended_plots"].append("histogram")
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions["recommended_plots"].append("bar plot")
    
    return suggestions


@tool
def list_available_libraries() -> dict:
    """List available libraries for data analysis.
    
    Returns:
        A dictionary with available libraries and their usage
    """
    return {
        "available_libraries": {
            "pandas": "Data manipulation and analysis (imported as pd)",
            "numpy": "Numerical computing (imported as np)",
            "matplotlib.pyplot": "Basic plotting (imported as plt)",
            "seaborn": "Statistical visualizations (imported as sns)",
            "sklearn": "Machine learning (various submodules)",
            "scipy": "Scientific computing and statistics"
        },
        "common_operations": [
            "df.describe() - Get summary statistics",
            "df.corr() - Calculate correlations",
            "pd.plotting.scatter_matrix() - Scatter plot matrix",
            "sns.heatmap() - Correlation heatmap",
            "plt.savefig() - Save plots"
        ]
    }


def create_validation_tools(dataset: pd.DataFrame = None) -> list:
    """Create validation tool instances.
    
    Args:
        dataset: Optional pandas DataFrame to bind to dataset-specific tools
        
    Returns:
        List of tool instances
    """
    from langchain_core.tools import StructuredTool
    
    tools = []
    
    # Add dataset-independent tools
    tools.append(validate_python_code)
    tools.append(validate_analysis_plan)
    tools.append(list_available_libraries)
    
    # Add dataset-dependent tools if dataset is provided
    if dataset is not None:
        # Create wrapper functions that capture the dataset in closure
        def _check_dataset_columns(columns: list[str]) -> dict:
            """Check if specified columns exist in the dataset.
            
            Args:
                columns: List of column names to check
            """
            return check_dataset_columns.func(dataset, columns)
        
        def _suggest_visualizations(query: str) -> dict:
            """Suggest appropriate visualizations based on dataset and query.
            
            Args:
                query: The user's query
            """
            return suggest_visualizations.func(dataset, query)
        
        tools.append(StructuredTool.from_function(
            func=_check_dataset_columns,
            name="check_dataset_columns",
            description="Check if specified columns exist in the dataset. Takes 'columns' parameter (list of strings)."
        ))
        
        tools.append(StructuredTool.from_function(
            func=_suggest_visualizations,
            name="suggest_visualizations",
            description="Suggest appropriate visualizations based on dataset and query. Takes 'query' parameter."
        ))
    
    return tools

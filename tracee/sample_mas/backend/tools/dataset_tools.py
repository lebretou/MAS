"""
tools for our sample mas
"""
import pandas as pd
from langchain_core.tools import tool 
from typing import Any


@tool
def get_dataset_info(dataset: pd.DataFrame) -> dict:
    """Get dataset information including columns, types, shape, and basic statistics.
    """
    return {
        "columns": list(dataset.columns),
        "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "missing_values": dataset.isnull().sum().to_dict(),
        "numeric_columns": list(dataset.select_dtypes(include=['number']).columns),
        "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
    }


@tool
def get_sample_rows(dataset: pd.DataFrame, n: int = 5) -> dict:
    """Get sample rows from the dataset.
    """
    sample = dataset.head(n)
    return {
        "sample_rows": sample.to_dict(orient='records'),
        "count": len(sample)
    }


@tool
def search_dataset_columns(dataset: pd.DataFrame, keyword: str) -> dict:
    """Search for columns containing a specific keyword.
    """
    matching_columns = [col for col in dataset.columns if keyword.lower() in col.lower()]
    return {
        "matching_columns": matching_columns,
        "count": len(matching_columns)
    }


@tool
def get_column_statistics(dataset: pd.DataFrame, column: str) -> dict:
    """Get detailed statistics for a specific column.
    
    Args:
        dataset: The pandas DataFrame
        column: The column name to analyze
        
    Returns:
        A dictionary with column statistics
    """
    if column not in dataset.columns:
        return {"error": f"Column '{column}' not found in dataset"}
    
    col_data = dataset[column]
    stats = {
        "column": column,
        "dtype": str(col_data.dtype),
        "non_null_count": int(col_data.count()),
        "null_count": int(col_data.isnull().sum()),
    }
    
    if pd.api.types.is_numeric_dtype(col_data):
        stats.update({
            "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
            "median": float(col_data.median()) if not col_data.isnull().all() else None,
            "std": float(col_data.std()) if not col_data.isnull().all() else None,
            "min": float(col_data.min()) if not col_data.isnull().all() else None,
            "max": float(col_data.max()) if not col_data.isnull().all() else None,
        })
    else:
        stats.update({
            "unique_values": int(col_data.nunique()),
            "top_values": col_data.value_counts().head(5).to_dict() if not col_data.empty else {}
        })
    
    return stats


def create_dataset_tools_for_agent(dataset: pd.DataFrame) -> list:
    """Create tool instances bound to a specific dataset.
    
    Args:
        dataset: The pandas DataFrame to bind to the tools
        
    Returns:
        List of tool instances that can be used with agents
    """
    from langchain_core.tools import StructuredTool
    
    # create wrapper functions that capture the dataset in closure
    def _get_dataset_info() -> dict:
        """Get comprehensive dataset information including columns, types, shape, and basic statistics."""
        return get_dataset_info.func(dataset)
    
    def _get_sample_rows(n: int = 5) -> dict:
        """Get sample rows from the dataset.
        
        Args:
            n: Number of rows to sample (default: 5)
        """
        return get_sample_rows.func(dataset, n)
    
    def _search_dataset_columns(keyword: str) -> dict:
        """Search for columns containing a specific keyword.
        
        Args:
            keyword: The keyword to search for in column names
        """
        return search_dataset_columns.func(dataset, keyword)
    
    def _get_column_statistics(column: str) -> dict:
        """Get detailed statistics for a specific column.
        
        Args:
            column: The column name to analyze
        """
        return get_column_statistics.func(dataset, column)
    
    # convert to LangChain tools
    tools = [
        StructuredTool.from_function(
            func=_get_dataset_info,
            name="get_dataset_info",
            description="Get comprehensive dataset information including columns, types, shape, and basic statistics."
        ),
        StructuredTool.from_function(
            func=_get_sample_rows,
            name="get_sample_rows",
            description="Get sample rows from the dataset. Takes optional parameter 'n' (default: 5)."
        ),
        StructuredTool.from_function(
            func=_search_dataset_columns,
            name="search_dataset_columns",
            description="Search for columns containing a specific keyword. Takes 'keyword' parameter."
        ),
        StructuredTool.from_function(
            func=_get_column_statistics,
            name="get_column_statistics",
            description="Get detailed statistics for a specific column. Takes 'column' parameter."
        ),
    ]
    
    return tools

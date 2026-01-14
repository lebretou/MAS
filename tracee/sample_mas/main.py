#!/usr/bin/env python
"""
CLI entry point for the Data Analysis Multi-Agent System.
Run this script to analyze datasets from the terminal.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

from backend.graph.workflow import run_analysis_workflow
from backend.telemetry.config import setup_telemetry


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from various file formats."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif suffix == ".json":
        return pd.read_json(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        print(f"Error: Unsupported file format: {suffix}")
        print("Supported formats: .csv, .xlsx, .xls, .json, .parquet")
        sys.exit(1)


def interactive_mode(dataset: pd.DataFrame, dataset_path: str, session_id: str):
    """Run interactive query loop."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    print(f"Columns: {', '.join(dataset.columns[:10])}" + ("..." if len(dataset.columns) > 10 else ""))
    print()
    
    query_count = 0
    while True:
        query = input("\n🔍 Enter your query: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        query_count += 1
        current_session = f"{session_id}_q{query_count}"
        
        print("\n" + "-" * 40)
        print("Running analysis workflow...")
        print("-" * 40)
        
        result = run_analysis_workflow(
            dataset=dataset,
            query=query,
            dataset_path=dataset_path,
            session_id=current_session
        )
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        
        if result.get("success"):
            print(f"\n✅ Analysis completed successfully")
            print(f"\n📋 Summary:\n{result.get('final_summary', 'No summary available')}")
            
            if result.get("generated_code"):
                print(f"\n💻 Generated Code:\n{'-' * 40}")
                print(result.get("generated_code"))
            
            plots = result.get("execution_result", {}).get("plots", [])
            if plots:
                print(f"\n📊 Generated plots: {plots}")
        else:
            print(f"\n❌ Analysis failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Summary: {result.get('final_summary', '')}")


def single_query_mode(dataset: pd.DataFrame, query: str, dataset_path: str, session_id: str):
    """Run a single query and exit."""
    print("\n" + "=" * 60)
    print("Data Analysis Multi-Agent System")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Query: {query}")
    print("\n" + "-" * 40)
    print("Running analysis workflow...")
    print("-" * 40)
    
    result = run_analysis_workflow(
        dataset=dataset,
        query=query,
        dataset_path=dataset_path,
        session_id=session_id
    )
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    
    if result.get("success"):
        print(f"\n✅ Analysis completed successfully")
        print(f"\n📋 Summary:\n{result.get('final_summary', 'No summary available')}")
        
        if result.get("generated_code"):
            print(f"\n💻 Generated Code:\n{'-' * 40}")
            print(result.get("generated_code"))
        
        plots = result.get("execution_result", {}).get("plots", [])
        if plots:
            print(f"\n📊 Generated plots saved to outputs/: {plots}")
    else:
        print(f"\n❌ Analysis failed")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Summary: {result.get('final_summary', '')}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Data Analysis Multi-Agent System - Analyze datasets with natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with a dataset
  python main.py --dataset data.csv
  
  # Single query mode
  python main.py --dataset data.csv --query "Plot the distribution of the age column"
  
  # Use sample data for testing
  python main.py --sample --query "Create a correlation heatmap"
"""
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to the dataset file (CSV, Excel, JSON, or Parquet)"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Analysis query (if not provided, runs in interactive mode)"
    )
    
    parser.add_argument(
        "--sample", "-s",
        action="store_true",
        help="Use sample_data.csv for testing"
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        default="cli",
        help="Session ID for tracing (default: cli)"
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.sample:
        dataset_path = Path(__file__).parent / "sample_data.csv"
        if not dataset_path.exists():
            print("Error: sample_data.csv not found in the project directory")
            sys.exit(1)
    elif args.dataset:
        dataset_path = args.dataset
    else:
        parser.print_help()
        print("\nError: Please provide --dataset or use --sample for testing")
        sys.exit(1)
    
    # Initialize telemetry
    print("\n" + "=" * 60)
    print("Initializing telemetry...")
    print("=" * 60)
    setup_telemetry()
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_dataset(str(dataset_path))
    print(f"✓ Loaded {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(dataset, args.query, str(dataset_path), args.session_id)
    else:
        interactive_mode(dataset, str(dataset_path), args.session_id)


if __name__ == "__main__":
    main()

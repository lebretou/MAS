"""
Test script for the multi-agent data analysis system.
This script tests the workflow without the web interface.
"""

import pandas as pd
import numpy as np
from backend.graph.workflow import run_analysis_workflow
from backend.telemetry.config import setup_telemetry
import os


def create_test_dataset():
    """Create a simple test dataset."""
    np.random.seed(42)
    n = 50
    
    return pd.DataFrame({
        'age': np.random.randint(20, 70, n),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })


def test_simple_query():
    """Test a simple query that should stay in interaction agent."""
    print("\n" + "=" * 60)
    print("TEST 1: Simple Query (Interaction Agent Only)")
    print("=" * 60)
    
    dataset = create_test_dataset()
    query = "What columns are in this dataset?"
    
    print(f"\nQuery: {query}")
    print("\nRunning workflow...")
    
    result = run_analysis_workflow(dataset, query, "test_dataset", "test_1")
    
    print(f"\nRelevance Decision: {result['relevance_decision']}")
    print(f"\nFinal Summary:\n{result['final_summary']}")
    
    assert result['relevance_decision'] == 'chat_only', "Expected chat_only decision"
    print("\n✓ Test 1 PASSED")


def test_analysis_query():
    """Test an analysis query that should trigger the full workflow."""
    print("\n" + "=" * 60)
    print("TEST 2: Analysis Query (Full Workflow)")
    print("=" * 60)
    
    dataset = create_test_dataset()
    query = "Create a histogram showing the distribution of the age column"
    
    print(f"\nQuery: {query}")
    print("\nRunning workflow...")
    
    result = run_analysis_workflow(dataset, query, "test_dataset", "test_2")
    
    print(f"\nRelevance Decision: {result['relevance_decision']}")
    print(f"\nGenerated Code:\n{result.get('generated_code', 'No code generated')[:200]}...")
    print(f"\nExecution Success: {result.get('execution_result', {}).get('success', False)}")
    print(f"\nPlots Generated: {result.get('execution_result', {}).get('plots', [])}")
    print(f"\nFinal Summary:\n{result['final_summary'][:300]}...")
    
    assert result['relevance_decision'] == 'relevant', "Expected relevant decision"
    assert result.get('generated_code'), "Expected code to be generated"
    print("\n✓ Test 2 PASSED")


def test_correlation_query():
    """Test a correlation analysis query."""
    print("\n" + "=" * 60)
    print("TEST 3: Correlation Analysis")
    print("=" * 60)
    
    dataset = create_test_dataset()
    query = "Plot the correlation between all numeric variables as a heatmap"
    
    print(f"\nQuery: {query}")
    print("\nRunning workflow...")
    
    result = run_analysis_workflow(dataset, query, "test_dataset", "test_3")
    
    print(f"\nRelevance Decision: {result['relevance_decision']}")
    print(f"\nExecution Success: {result.get('execution_result', {}).get('success', False)}")
    print(f"\nPlots Generated: {result.get('execution_result', {}).get('plots', [])}")
    print(f"\nFinal Summary:\n{result['final_summary'][:300]}...")
    
    assert result['relevance_decision'] == 'relevant', "Expected relevant decision"
    print("\n✓ Test 3 PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DATA ANALYSIS MULTI-AGENT SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # Setup telemetry
    print("\nSetting up telemetry...")
    setup_telemetry()
    
    # Check if environment variables are set
    print("\nChecking environment...")
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY not set. Tests will fail.")
        print("Please set up your .env file with required API keys.")
        return
    
    try:
        # Run tests
        test_simple_query()
        test_analysis_query()
        test_correlation_query()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
        print("\nNext Steps:")
        print("1. Check LangSmith dashboard for traces: https://smith.langchain.com/")
        print("2. Check Langfuse dashboard for telemetry: https://cloud.langfuse.com/")
        print("3. Start the web server: python -m backend.main")
        print("4. Open browser: http://localhost:8000/app/")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

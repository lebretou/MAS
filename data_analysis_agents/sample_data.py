"""
Script to generate sample datasets for testing the multi-agent system.
"""

import pandas as pd
import numpy as np

def create_sample_dataset_1():
    """Create a simple dataset with numeric and categorical variables."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'age': np.random.randint(20, 70, n),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'satisfaction': np.random.randint(1, 6, n)
    })
    
    df.to_csv('sample_data_1.csv', index=False)
    print("✓ Created sample_data_1.csv")
    return df


def create_sample_dataset_2():
    """Create a dataset suitable for regression analysis."""
    np.random.seed(123)
    n = 150
    
    # Create features with some correlation
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = x1 * 0.5 + np.random.normal(0, 0.5, n)  # Correlated with x1
    
    # Create target variable
    y = 2 * x1 + 3 * x2 - 1.5 * x3 + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,
        'target': y,
        'group': np.random.choice(['Group_A', 'Group_B', 'Group_C'], n)
    })
    
    df.to_csv('sample_data_2.csv', index=False)
    print("✓ Created sample_data_2.csv")
    return df


def create_sample_dataset_3():
    """Create a sales dataset."""
    np.random.seed(456)
    n = 200
    
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'sales': np.random.poisson(100, n) + np.sin(np.arange(n) * 2 * np.pi / 30) * 20,
        'customers': np.random.poisson(50, n),
        'revenue': np.random.normal(5000, 1000, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], n)
    })
    
    df.to_csv('sample_data_3.csv', index=False)
    print("✓ Created sample_data_3.csv")
    return df


if __name__ == "__main__":
    print("Generating sample datasets...")
    print("=" * 50)
    
    create_sample_dataset_1()
    create_sample_dataset_2()
    create_sample_dataset_3()
    
    print("=" * 50)
    print("Sample datasets created successfully!")
    print("\nTo use them:")
    print("1. Start the server: python -m backend.main")
    print("2. Open http://localhost:8000/app/")
    print("3. Upload one of the sample CSV files")
    print("4. Try example queries like:")
    print("   - 'Plot correlation between all numeric variables'")
    print("   - 'Show distribution of age with a histogram'")
    print("   - 'Run regression analysis on the features'")

"""
Simple diagnostic script to examine the structure of data files
"""

import pandas as pd
import sys
import os

def inspect_excel_file(file_path):
    """Inspect an Excel file and print its structure"""
    print(f"\n--- Examining Excel file: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return
        
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Basic info
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        
        # Sample data
        print("\nFirst 3 rows:")
        for i, row in df.head(3).iterrows():
            print(f"Row {i}:")
            for col in df.columns:
                print(f"  {col}: {row[col]}")
            print()
            
        # Check for common data issues
        null_counts = df.isnull().sum()
        print("\nNull value counts:")
        for col, count in null_counts.items():
            print(f"  {col}: {count}")
            
        # Check data types
        print("\nData types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
            
    except Exception as e:
        print(f"ERROR: Failed to read Excel file: {e}")
        import traceback
        print(traceback.format_exc())

def inspect_csv_file(file_path):
    """Inspect a CSV file and print its structure"""
    print(f"\n--- Examining CSV file: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return
        
    try:
        # Print first few lines of raw file
        print("\nRaw file preview:")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
                for i, line in enumerate(lines):
                    print(f"Line {i+1}: {line.strip()}")
        except Exception as e:
            print(f"Error reading raw file: {e}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic info
        print(f"\nNumber of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        
        # Sample data
        print("\nFirst 3 rows:")
        for i, row in df.head(3).iterrows():
            print(f"Row {i}:")
            for col in df.columns:
                print(f"  {col}: {row[col]}")
            print()
            
        # Check for common data issues
        null_counts = df.isnull().sum()
        print("\nNull value counts:")
        for col, count in null_counts.items():
            print(f"  {col}: {count}")
            
        # Check data types
        print("\nData types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
            
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    """Main function to run diagnostics"""
    print("=== Data File Diagnostics ===")
    
    # Check Python version and pandas version
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    
    # Files to check
    product_file = "pricing_test.xlsx"
    competitor_file = "competitor_prices.csv"
    
    # Inspect each file
    inspect_excel_file(product_file)
    inspect_csv_file(competitor_file)
    
    print("\nDiagnostic complete!")

if __name__ == "__main__":
    main()
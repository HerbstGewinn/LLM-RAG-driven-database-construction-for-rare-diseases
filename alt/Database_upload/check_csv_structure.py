import pandas as pd
import os
import numpy as np

def suggest_db_types(df):
    type_mapping = {}
    
    for column in df.columns:
        # Get sample of non-null values
        sample = df[column].dropna()
        
        if len(sample) == 0:
            type_mapping[column] = 'text'  # default to text if no data
            continue
            
        # Try to infer if it's numeric
        try:
            pd.to_numeric(sample)
            # Check if all numbers are integers
            if all(sample.astype(float).astype(int) == sample.astype(float)):
                type_mapping[column] = 'integer'
            else:
                type_mapping[column] = 'double precision'
            continue
        except:
            pass
        
        # Check if it's a boolean
        if set(sample.unique()) <= {'True', 'False', True, False}:
            type_mapping[column] = 'boolean'
            continue
            
        # Check if it's a date
        try:
            pd.to_datetime(sample)
            type_mapping[column] = 'timestamp'
            continue
        except:
            pass
            
        # Default to text
        # Check max length
        max_length = sample.astype(str).str.len().max()
        if max_length <= 255:
            type_mapping[column] = f'varchar({max_length})'
        else:
            type_mapping[column] = 'text'
    
    return type_mapping

def analyze_csv_structure():
    try:
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(os.path.dirname(current_dir), 'validation_data.csv')
        
        print(f"\nReading CSV file from: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        
        # Get suggested database types
        db_types = suggest_db_types(df)
        
        # Print column analysis
        print("\nSuggested database column types:")
        print("--------------------------------")
        for column, db_type in db_types.items():
            sample_values = df[column].dropna().head(3).tolist()
            print(f"\nColumn: {column}")
            print(f"Suggested type: {db_type}")
            print(f"Sample values: {sample_values}")
            print(f"Null values: {df[column].isna().sum()} out of {len(df)}")
            
        # Print SQL create table statement
        print("\nSQL Create Table Statement:")
        print("---------------------------")
        print("CREATE TABLE validation_data (")
        columns = []
        for column, db_type in db_types.items():
            nullable = "NULL" if df[column].isna().any() else "NOT NULL"
            columns.append(f"    {column.lower()} {db_type} {nullable}")
        print(",\n".join(columns))
        print(");")
            
    except Exception as e:
        print(f"Error analyzing CSV: {str(e)}")

if __name__ == "__main__":
    analyze_csv_structure()
import pandas as pd
from supabase_connection import get_supabase_client
import os
import numpy as np

def upload_validation_data():
    try:
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to validation_data.csv
        csv_path = os.path.join(os.path.dirname(current_dir), 'validation_data.csv')
        
        print(f"Looking for CSV file at: {csv_path}")
        
        # Read the CSV file with proper encoding and separator
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        
        # Replace NaN values with None
        df = df.replace({np.nan: None})
        
        # Print any remaining NaN values to debug
        for column in df.columns:
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                print(f"Column {column} has {nan_count} NaN values")
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert data into Supabase table
        data = supabase.table('validation_data').insert(records).execute()
        
        print(f"Successfully uploaded {len(records)} records to Supabase!")
        return data
        
    except Exception as e:
        print(f"Error uploading data to Supabase: {str(e)}")
        raise

if __name__ == "__main__":
    upload_validation_data() 
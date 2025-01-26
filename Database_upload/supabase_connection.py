from supabase import create_client
import os
from dotenv import load_dotenv, find_dotenv

def get_supabase_client():
    # Find the .env file
    env_path = find_dotenv(raise_error_if_not_found=True)
    print(f"Found .env file at: {env_path}")
    
    # Load environment variables
    load_dotenv(env_path)
    
    # Get Supabase credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    # Debug prints with repr to show exact string content
    print(f"Loaded URL: {supabase_url!r}")
    print(f"Loaded key exists: {bool(supabase_key)}")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Missing Supabase credentials in .env file")
    
    # Remove any whitespace
    supabase_url = supabase_url.strip()
    supabase_key = supabase_key.strip()
    
    return create_client(supabase_url, supabase_key) 
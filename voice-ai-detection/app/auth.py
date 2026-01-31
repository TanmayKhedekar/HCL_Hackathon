import os
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

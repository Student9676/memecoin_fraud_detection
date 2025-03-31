import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read API key from .env file
api_key = os.getenv("SOLSCAN_API_KEY")

url = "https://pro-api.solscan.io/v2.0/token/trending?limit=10"

headers = {"token": api_key}

response = requests.get(url, headers=headers)

print(response.text)
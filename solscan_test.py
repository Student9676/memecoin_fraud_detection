import requests
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("SOLSCAN_API_KEY")

url = "https://public-api.solscan.io/chaininfo"    

headers = {"token": api_key}
response = requests.get(url, headers=headers)
print(response.text)

url = "https://pro-api.solscan.io/v2.0/block/last?limit=10"    
response = requests.get(url, headers=headers)
print(response.text) # need to purchase a plan for any data other than the first query

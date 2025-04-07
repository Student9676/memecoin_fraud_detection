import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("SOLTRACKER_API_KEY")
headers = {"x-api-key": api_key}

url = "https://data.solanatracker.io/tokens/8yj7dQ1smUViXLYefLBpQCiHPuuj2RPEcbUhtxbiip7b"
response = requests.get(url, headers=headers)

print(response.text)

json_response = response.json()
with open("output.json", "w") as f:
    f.write(json.dumps(json_response, indent=4))
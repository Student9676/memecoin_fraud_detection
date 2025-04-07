import json
from playwright.sync_api import sync_playwright
import cookie_loader
import os
import re
from dotenv import load_dotenv
import requests
import sys
import time



"""
Nodes:
- Dev Attributes: id (api), num tokens created before (webscrape), num rugpull tokens created
- Wallet Attributes: id, balance (coin & sol -- before any investment and after final transaction?**), rugpull association, num transactions, num transfers, creation age (ALL WEBSCRAPE)
Edges:
- Dev-coin Attributes: id, creation time, starting balance/mc (WEBSCRAPE)
"""
load_dotenv()
SOLTRACKER_API_KEY = os.getenv("SOLTRACKER_API_KEY")
HEADERS = {"x-api-key": SOLTRACKER_API_KEY}

token_name = "r/pwease"
token_name = re.sub(r"[^\w\-]", "_", token_name)
token_address = "H8AyUYyjzVLxkkDHDShd671YrtziroNc8HjfXDnypump"
get_cookies = False
COOKIES_PATH = "cookies.json"
DEV_NODES_PATH = "dev_nodes"
TRANSACTION_DATA_PATH = "transaction_data"
BASE_SOLTRACKER_API_URL = "https://data.solanatracker.io"
BASE_SOLSCAN_URL = "https://solscan.io"


def load_cookies(cookie_file):
    with open(cookie_file, "r") as f:
        return json.load(f)

def save_transaction_data():
    idx = 0
    cursor = None  # no cursor for first request
    has_next = True
    retried = False
    while has_next:
        url = f"{BASE_SOLTRACKER_API_URL}/trades/{token_address}"
        url += f"?cursor={cursor}" if cursor else ""
        
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200 and not retried:
            retried = True
            print(f"Error fetching trades: {response.status_code}: {response.text}\n\nRetrying...")
            time.sleep(2)
            continue
        elif response.status_code != 200:
            raise Exception(f"Error fetching trades: {response.status_code}: {response.text}")
            
        trades = response.json().get("trades", [])
        has_next = response.json().get("hasNextPage", False)
        cursor = response.json().get("nextCursor", None)

        for tx_data in trades:
            folder = os.path.join(TRANSACTION_DATA_PATH, token_name)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"{token_name}_{idx}.json"), "w") as f:
                json.dump(tx_data, f, indent=4)
            idx += 1
        time.sleep(1.1)  # API rate limit
    print(f"Fetched {idx} transactions for {token_name}.")

def get_dev_data():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH)
        # load cookies from the saved file and set them in the context
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()

        page.goto(f"{BASE_SOLSCAN_URL}/token/{token_address}")
        page.wait_for_load_state("networkidle")
        page.locator("button[aria-controls='radix-:r1t:']").click()
        page.wait_for_selector("div[data-state='open']")
        dev_address = page.locator("div[data-state='open'] a.text-current").get_attribute("href")
        dev_address = dev_address.split("/")[-1]
        print(f"Dev Address: {dev_address}")
        
        page.goto(f"https://solscan.io/account/{dev_address}?activity_type=ACTIVITY_SPL_INIT_MINT#defiactivities")
        page.wait_for_load_state("networkidle")
        num_tokens_created = page.locator("div.gap-1.flex-row.items-center.justify-between.flex-wrap.flex.lg\\:hidden").inner_text()
        num_tokens_created = int(re.findall(r"[0-9]+", num_tokens_created)[0])
        browser.close()
        print(f"Num tokens created: {num_tokens_created}")

        dev_data = {
            "token_name": token_name,
            "token_address": token_address,
            "dev_address": dev_address,
            "num_tokens_created": num_tokens_created,
            "num_rugpull_tokens_created": 0,
        }
        return dev_data

if __name__ == "__main__":

        dev_data = get_dev_data()
        os.makedirs(DEV_NODES_PATH, exist_ok=True)
        with open(os.path.join(DEV_NODES_PATH, f"{token_name}.json"), "w") as f:
            json.dump(dev_data, f, indent=4)
        
        save_transaction_data()

        print("Finished!")
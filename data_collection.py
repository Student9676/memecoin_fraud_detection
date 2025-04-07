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
headless = True
COOKIES_PATH = "cookies.json"
DEV_NODES_PATH = "dev_nodes"
WALLET_NODES_PATH = "wallet_nodes"
TRANSACTION_DATA_PATH = "transaction_data"
BASE_SOLTRACKER_API_URL = "https://data.solanatracker.io"
BASE_SOLSCAN_URL = "https://solscan.io"

def extract_number(text: str) -> int:
    text = text.split("(s)")[0]
    text = text.replace(",", "")
    return int(re.findall(r"[0-9]+", text)[0])

def load_cookies(cookie_file):
    with open(cookie_file, "r") as f:
        return json.load(f)

def read_transaction_data():
    transactions = []
    for folder_name in os.listdir(TRANSACTION_DATA_PATH):
        folder_path = os.path.join(TRANSACTION_DATA_PATH, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                transactions.append(data)
    return transactions

def save_wallet_data():
    transactions = read_transaction_data()
    wallet_addresses = set()
    for transaction in transactions:
        wallet_addresses.add(transaction["wallet"])
    os.makedirs(WALLET_NODES_PATH, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH)
        # load cookies from the saved file and set them in the context
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()
        
        for wallet_address in wallet_addresses:
            url = f"{BASE_SOLSCAN_URL}/account/{wallet_address}"
            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.locator("button", has_text="Transfers").click()       
            page.wait_for_load_state("networkidle")
            num_transfers = page.locator("div.flex.gap-1.flex-row.items-center.justify-start.flex-wrap", has_text="transfer(s)").nth(0).inner_text()
            num_transfers = extract_number(num_transfers)
            page.locator("button", has_text="Defi Activities").click()
            num_defi_activities = page.locator("div.flex.gap-1.flex-row.items-center.justify-start.flex-wrap", has_text="activities(s)").nth(0).inner_text()
            num_defi_activities = extract_number(num_defi_activities)
            
            wallet_data = {
                "wallet_address": wallet_address,
                "num_transfers": num_transfers,
                "num_defi_activities": num_defi_activities,
                "rugpull_association": 0,
            }
            filename = f"{wallet_address[:4]}...{wallet_address[-4:]}"
            with open(os.path.join(WALLET_NODES_PATH, f"{filename}.json"), "w") as f:
                json.dump(wallet_data, f, indent=4)

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
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH)
        # load cookies from the saved file and set them in the context
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()

        token_url = f"{BASE_SOLSCAN_URL}/token/{token_address}"
        page.goto(token_url)
        page.wait_for_load_state("networkidle")
        page.locator("button[aria-controls='radix-:r1t:']").click()
        page.wait_for_selector("div[data-state='open']")
        dev_address = page.locator("div[data-state='open'] a.text-current").get_attribute("href")
        dev_address = dev_address.split("/")[-1]
        print(f"Dev Address: {dev_address}")
        
        dev_defi_url = f"{BASE_SOLSCAN_URL}/account/{dev_address}?activity_type=ACTIVITY_SPL_INIT_MINT#defiactivities"
        page.goto(dev_defi_url)
        page.wait_for_load_state("networkidle")
        num_tokens_created = page.locator("div.gap-1.flex-row.items-center.justify-between.flex-wrap.flex.lg\\:hidden").inner_text()
        num_tokens_created = extract_number(num_tokens_created)
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

        save_wallet_data()

        print("Finished!")
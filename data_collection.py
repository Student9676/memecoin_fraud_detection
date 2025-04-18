import json
from playwright.sync_api import sync_playwright
import cookie_loader
import os
import re
from dotenv import load_dotenv
import requests
import sys
import time
from tqdm import tqdm

"""
Nodes:
- Dev Attributes: id (api), num tokens created before (webscrape), num rugpull tokens created
- Wallet Attributes: id, balance (coin & sol -- before any investment and after final transaction?**), rugpull association, num transactions, num transfers, creation age (ALL WEBSCRAPE)
Edges:
- Dev-coin Attributes: id, creation time, starting balance/mc (WEBSCRAPE)
"""
load_dotenv()
SOLTRACKER_API_KEY = os.getenv("SOLTRACKER_API_KEY")
SOLTRACKER_API_KEY = "bf14fc87-65f6-4e6d-8e79-806e8d4b48f4"
HEADERS = {"x-api-key": SOLTRACKER_API_KEY}

token_name = "r/pwease"
token_name = re.sub(r"[^\w\-]", "_", token_name) # replace special characters with underscores
token_address = "H8AyUYyjzVLxkkDHDShd671YrtziroNc8HjfXDnypump"
get_cookies = False
headless = False
COOKIES_PATH = "cookies.json"
DEV_NODES_PATH = "dev_nodes"
WALLET_NODES_PATH = "wallet_nodes"
TRANSACTION_DATA_PATH = "transaction_data"
DEV_COIN_EDGES_PATH = "dev_coin_edges"
BASE_SOLTRACKER_API_URL = "https://data.solanatracker.io"
BASE_SOLSCAN_URL = "https://solscan.io"

def extract_number(text: str) -> int:
    """
    Extracts the first integer number found in the given text.

    Args:
        text (str): The input string of format "Total 1,457 transfer(s) Filters0".

    Returns:
        int: The first integer found in the text.
    """
    text = text.split("(s)")[0] # expected format
    text = text.replace(",", "")
    return int(re.findall(r"[0-9]+", text)[0])

def load_cookies(cookie_file):
    """
    This function reads a JSON file containing cookies and returns the 
    parsed data as a dict or list of dicts.

    Args:
        cookie_file (str): The path to the JSON file containing cookies.

    Returns:
        dict or list: The parsed cookies data from the JSON file.
    """
    with open(cookie_file, "r") as f:
        return json.load(f)

def read_transaction_data():
    """
    Reads transaction data from a nested directory structure and returns it as a list of dictionaries.
    The expected directory structure is:
    transaction_data/
        ├── token_name/
        │   ├── token_name_0.json
        │   ├── token_name_1.json
        │   └── ...
        └── ...
    
    Returns:
        list: A list of dictionaries, where each dictionary represents a transaction loaded 
              from a JSON file.
    """
    print("Reading transaction data...")
    transactions = []
    for folder_name in os.listdir(TRANSACTION_DATA_PATH): # for each coin
        folder_path = os.path.join(TRANSACTION_DATA_PATH, folder_name)
        for file_name in tqdm(os.listdir(folder_path), desc=f"Reading {folder_name} transactions..."): # for each transaction
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                transactions.append(data)
    print(f"Read {len(transactions)} transactions.")
    return transactions

def get_dev_coin_edge():
    """
    This function returns the developer-coin edge data for a given token address.
    It retrieves the developer's address, the token name, the token address, the 
    creation time, and the starting balance from transaction_data and dev_nodes. 
    NOTE: Make sure to run this function after saving the dev_nodes and transaction_data.
    
    Returns:
        dict: A dictionary containing the developer's address, token name, token address,
              creation time, and starting balance.
    """
    # get the last transaction JSON file for the token_name
    transaction_folder = os.path.join(TRANSACTION_DATA_PATH, token_name)
    transaction_files = sorted(
        os.listdir(transaction_folder),
        key = lambda x: extract_number(x) # sort by the number in the filename
    )
    dev_transaction_filename = transaction_files[-1]  # alphabetically the last file, i.e. the first tx is the dev funding the coin
    dev_transaction_path = os.path.join(transaction_folder, dev_transaction_filename)
    with open(dev_transaction_path, "r") as f:
        last_transaction_data = json.load(f)

    # get the dev wallet_address from data in dev_nodes
    dev_node_file = os.path.join(DEV_NODES_PATH, f"{token_name}.json")
    with open(dev_node_file, "r") as f:
        dev_data = json.load(f)
    
    assert last_transaction_data["wallet"] == dev_data["dev_address"], "The first transaction's wallet address is not the dev's address."
    assert last_transaction_data["type"] == "buy", "The first transaction is not a buy transaction."

    creation_time = last_transaction_data["time"] # unix timestamp
    starting_balance = last_transaction_data["volume"] # starting balance in USD
    edge_data = {
        "dev_address": dev_data["dev_address"],
        "token_name": token_name,
        "token_address": token_address,
        "creation_time":  creation_time,
        "starting_balance": starting_balance,
    }
    return edge_data

def save_wallet_data():
    """
    This function collects and saves wallet data from the Solscan website for each 
    wallet address associated with the transactions in the transaction data. It saves
    a JSON file for each wallet address in the WALLET_NODES_PATH directory containing 
    the wallet address, number of transfers, number of defi activities, and a rugpull 
    association boolean.
    """
    print("Getting wallet data...")
    # get all unique wallet addresses from the transaction data
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
        
        for wallet_address in tqdm(wallet_addresses):
            url = f"{BASE_SOLSCAN_URL}/account/{wallet_address}"
            page.goto(url)
            page.wait_for_load_state("networkidle")
            # get num of transfers
            page.locator("button", has_text="Transfers").click()      
            page.wait_for_load_state("networkidle")
            num_transfers = page.locator("div.flex.gap-1.flex-row.items-center.justify-start.flex-wrap", has_text="transfer(s)").nth(0).inner_text()
            num_transfers = extract_number(num_transfers)
            # get num of defi activities
            page.locator("button", has_text="Defi Activities").click()
            num_defi_activities = page.locator("div.flex.gap-1.flex-row.items-center.justify-start.flex-wrap", has_text="activities(s)").nth(0).inner_text()
            num_defi_activities = extract_number(num_defi_activities)
            
            wallet_data = {
                "wallet_address": wallet_address,
                "num_transfers": num_transfers,
                "num_defi_activities": num_defi_activities,
                "rugpull_association": 0,
            }
            with open(os.path.join(WALLET_NODES_PATH, f"{wallet_address}.json"), "w") as f:
                json.dump(wallet_data, f, indent=4)
        print("Saved all wallet data.")

def save_transaction_data():
    """
    Fetches transaction data from the SolanaTracker API and saves it to a JSON file.
    The transaction data is saved in the following directory structure:
    transaction_data/
        ├── token_name/
        │   ├── token_name_0.json
        │   ├── token_name_1.json
        │   └── ...
        └── ...
    """
    print("Getting transaction data...")
    idx = 0
    cursor = None  # no cursor for first request
    has_next = True
    retried = False
    while has_next: # following api pagination
        url = f"{BASE_SOLTRACKER_API_URL}/trades/{token_address}"
        url += f"?cursor={cursor}" if cursor else "" # for pagination
        
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200 and not retried: # retry once
            retried = True
            print(f"Error fetching trades: {response.status_code}: {response.text}\n\nRetrying...")
            time.sleep(2)
            continue
        elif response.status_code != 200:
            raise Exception(f"Error fetching trades: {response.status_code}: {response.text}")
            
        trades = response.json().get("trades", [])
        has_next = response.json().get("hasNextPage", False) # for pagination
        cursor = response.json().get("nextCursor", None) # for pagination

        for tx_data in tqdm(trades, desc="Getting batches of transactions..."):
            folder = os.path.join(TRANSACTION_DATA_PATH, token_name)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"{token_name}-{idx}.json"), "w") as f:
                json.dump(tx_data, f, indent=4)
            idx += 1
        time.sleep(1.1)  # API rate limit
    print(f"Fetched a total of {idx} transactions for {token_name}.")

def get_dev_data():
    """
    This function retrieves developer data for a given token address from the Solscan website.
    
    Returns:
        dict: A dictionary containing the developer's address, number of tokens created,
              and number of rugpull tokens created.
    """
    print("Getting dev data...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH)
        # load cookies from the saved file and set them in the context
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()

        # get dev address from specific div
        token_url = f"{BASE_SOLSCAN_URL}/token/{token_address}"
        page.goto(token_url)
        page.wait_for_load_state("networkidle")
        page.locator("button[aria-controls='radix-:r1t:']").click()
        page.wait_for_selector("div[data-state='open']")
        dev_address = page.locator("div[data-state='open'] a.text-current").get_attribute("href")
        dev_address = dev_address.split("/")[-1]
        
        # get num of tokens created from specific div
        dev_defi_url = f"{BASE_SOLSCAN_URL}/account/{dev_address}?activity_type=ACTIVITY_SPL_INIT_MINT#defiactivities"
        page.goto(dev_defi_url)
        page.wait_for_load_state("networkidle")
        num_tokens_created = page.locator("div.gap-1.flex-row.items-center.justify-between.flex-wrap.flex.lg\\:hidden").inner_text()
        num_tokens_created = extract_number(num_tokens_created)
        browser.close()

        dev_data = {
            "token_name": token_name, # remove?
            "token_address": token_address, # remove?
            "dev_address": dev_address,
            "num_tokens_created": num_tokens_created,
            "num_rugpull_tokens_created": 0,
        }
        print("Dev data returned.")
        return dev_data

def get_coin_data():
    """
    Retrieves coin attributes (coin id, market cap, rugpull flag, dev id) from Solana Tracker API or Solscan web scraping.
    Returns dictionary containing coin id, market cap, rugpull, and dev id
    """
    print("Getting coin data...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH)
        # Load cookies from saved file
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()
        
        # Go to token page on Solscan
        token_url = f"{BASE_SOLSCAN_URL}/token/{token_address}"
        page.goto(token_url)
        page.wait_for_load_state("networkidle")

        # Coin ID (use token address as ID)
        coin_id = token_address

        # Market Cap
        market_cap_selector = "div.grid.grid-cols-2.md\\:grid-cols-4 > div:nth-child(3) div.text-white"
        market_cap_text = page.locator(market_cap_selector).inner_text()
        market_cap_clean = market_cap_text.replace("$", "").replace(",", "").strip()
        market_cap = float(market_cap_clean)

        # Rugpull flag — infer based on heuristics or default to 0
        rugpull_flag = 0  # You could later add logic to label based on tx patterns, drops, or API indicators

        # Developer ID from previously defined logic
        page.locator("button[aria-controls='radix-:r1t:']").click()
        page.wait_for_selector("div[data-state='open']")
        dev_id = page.locator("div[data-state='open'] a.text-current").get_attribute("href").split("/")[-1]

        browser.close()

        coin_data = {
            "coin_id": coin_id,
            "market_cap": market_cap,
            "rugpull": rugpull_flag,
            "dev_id": dev_id,
        }

        print("Coin data retrieved.")
        return coin_data

def get_wallet_wallet_edges():
    """
    Derives wallet-wallet edges from transactions that include both sender and receiver wallets.

    Returns:
        list: A list of wallet-wallet edge dictionaries.
    """
    print("Getting wallet-wallet edges...")
    edges = []
    transactions = read_transaction_data()
    for tx in transactions:
        from_wallet = tx.get("from", None)
        to_wallet = tx.get("wallet", None)
        if from_wallet and to_wallet and from_wallet != to_wallet:
            edge = {
                "from_wallet": from_wallet,
                "to_wallet": to_wallet,
                "amount": tx["volume"],
                "timestamp": tx["time"],
            }
            edges.append(edge)
    print(f"Collected {len(edges)} wallet-wallet edges.")
    return edges

def get_wallet_dev_edges(dev_address):
    """
    Constructs wallet-dev edge data using transactions that interact with the dev.

    Args:
        dev_address (str): The developer's wallet address.

    Returns:
        list: A list of wallet-dev edge dictionaries.
    """
    print("Getting wallet-dev edges...")
    edges = []
    transactions = read_transaction_data()
    for tx in transactions:
        if tx["wallet"] != dev_address:
            edge = {
                "wallet_id": tx["wallet"],
                "dev_id": dev_address,
                "amount": tx["volume"],
                "timestamp": tx["time"],
            }
            edges.append(edge)
    print(f"Collected {len(edges)} wallet-dev edges.")
    return edges

def get_wallet_coin_edges():
    """
    Extracts wallet-coin edge data from transaction_data files.

    Returns:
        list: A list of wallet-coin edge dictionaries.
    """
    print("Getting wallet-coin edges...")
    edges = []
    transactions = read_transaction_data()
    for tx in transactions:
        edge = {
            "wallet_id": tx["wallet"],
            "coin_id": tx["tokenAddress"],
            "tx_type": tx["type"],
            "amount": tx["volume"],  # Could be SOL or token depending on tx type
            "timestamp": tx["time"],
            "mc_at_tx": tx.get("marketCap", None),  # Some txs may not have this
        }
        edges.append(edge)
    print(f"Collected {len(edges)} wallet-coin edges.")
    return edges

if __name__ == "__main__":

        # get dev data and save it as a JSON in DEV_NODES_PATH
        dev_data = get_dev_data()
        os.makedirs(DEV_NODES_PATH, exist_ok=True)
        with open(os.path.join(DEV_NODES_PATH, f"{token_name}.json"), "w") as f:
            json.dump(dev_data, f, indent=4)
        print(SOLTRACKER_API_KEY)
        SOLTRACKER_API_KEY = "bf14fc87-65f6-4e6d-8e79-806e8d4b48f4"
        
        save_transaction_data()
        save_wallet_data()

        coin_data = get_coin_data()
        os.makedirs(DEV_NODES_PATH, exist_ok=True)
        with open(os.path.join(DEV_NODES_PATH, f"{token_name}_coin.json"), "w") as f:
            json.dump(coin_data, f, indent=4)

        # get dev<->coin edge data and save it as a JSON in DEV_COIN_EDGES_PATH
        dev_coin_edge_data = get_dev_coin_edge()
        os.makedirs(DEV_COIN_EDGES_PATH, exist_ok=True)
        filename = f"{token_name}-{dev_coin_edge_data['dev_address']}.json"
        with open(os.path.join(DEV_COIN_EDGES_PATH, filename), "w") as f:
            json.dump(dev_coin_edge_data, f, indent=4)

       
        wallet_coin_edges = get_wallet_coin_edges()
        os.makedirs("wallet_coin_edges", exist_ok=True)
        with open(f"wallet_coin_edges/{token_name}.json", "w") as f:
            json.dump(wallet_coin_edges, f, indent=4)

        dev_address = dev_coin_edge_data["dev_address"]
        wallet_dev_edges = get_wallet_dev_edges(dev_address)
        os.makedirs("wallet_dev_edges", exist_ok=True)
        with open(f"wallet_dev_edges/{token_name}-{dev_address}.json", "w") as f:
            json.dump(wallet_dev_edges, f, indent=4)

    
        wallet_wallet_edges = get_wallet_wallet_edges()
        os.makedirs("wallet_wallet_edges", exist_ok=True)
        with open(f"wallet_wallet_edges/{token_name}.json", "w") as f:
            json.dump(wallet_wallet_edges, f, indent=4)

        print("Finished data collection!")
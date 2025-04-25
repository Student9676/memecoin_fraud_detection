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
from itertools import combinations

# API keys + URLs
load_dotenv()
SOLTRACKER_API_KEY = os.getenv("SOLTRACKER_API_KEY")
SOLTRACKER_HEADERS = {"x-api-key": SOLTRACKER_API_KEY}
SHYFT_API_KEY = os.getenv("SHYFT_API_KEY")
BASE_SOLTRACKER_API_URL = "https://data.solanatracker.io"
BASE_SOLSCAN_URL = "https://solscan.io"

# Tokens to collect data for
tokens = [
    ("r/pwease", "H8AyUYyjzVLxkkDHDShd671YrtziroNc8HjfXDnypump", 0), # 0
    ("brain", "6SzkDM3RKZWEVuGeNfBxJNNRbQCQC9WQVtqXmdzepump", 1), # 1 (rugpull)
    ("CK", "FuMeUE6XreUFzz8cTGAmgZp3E4qduehpzUJpcEsppump", 0), # 0
    ("KETAMINE", "7NsA8cMXi7U9CZ4wjif2J9g9YCKJkqy32PfBEtcVpump", 0), # 0
    ("MASSIVE", "ZGve8w1jgHwZLjd46kcbvFRep1TwpCngQsDx7Nmpump", 1), # 1 (rugpull)
    ("dragon", "2Vo6J4UVBYgky7rEfj6z2WVF94LLkAipkyRNuT37pump", 1), # 1 (slow rug)
    ("squeakaboo", "7Mf5cJdp3zpsA9GCBKdCK7So8BtzVK1w7qpVQcTvpump", 0), # 0
    ("sundog", "BhQHCi6AkngEyYJooswZ9sGXxAFBWESpUx9Je8WoMCRK", 1), # 1 (washtrading)
    ("popedog", "87eBqoJ6iwyFJcjYBGZivs7RA7nQ6JELoqp6jXfsj3mP", 1), # 1 (washtrading)
    ("trumppope", "S3CcCg1z2y5HZ3fsEHbAiCpmRXk2ApxzkrBeup6TiF2", 1), # 1 (washtrading)
    ("carlo", "7daesFB2skTZAEM9GmjeP9Nc3omsc7aWZdNWPrdupump", 0), # 0
    ("hierophant", "EehspZuVw3jcJ5ppYxrJTk3t62umPEsffPndnkR5pump", 0), # 0 
    ("btcd", "5dS7KMV8kmkLLSAvRsAqTTB3LvLNJyt4DhTXcnD7pump", 0), # 0
    ("manners", "9oN5M2gPity4QzxShdLk4suZQF7rLoB1oFFGGvr3pump", 0), # 0
    ("clickbait", "6Mntx18DBsk9em3a7KvaV4M5aiTDwrnNZgzR4RpQpump", 1), # 1
    ("ogtroll", "71XUCawm1BPXJ9JkEET2BR2hxVkLzLLMhzLWD7b6pump", 1), # 1
    ("quarter", "NmGYpXfwmc8Tt9yPtkDoo6tTyvjdZJfp3hAb7aWpump", 0), # 0
    ("fakeout", "D2BhLfj7749EnVDkF6WcKqDJDGp9mm2xdUjoZYDepump", 1), # 1
    ("dogs","8SJjRDpNgJLuSrwKvJ8jEu8uRBWTeDmNtadhrAzDpump", 1), # 1
    ("rpecoin","HZ97T6LzHrjXvnA3RCPsJhizebQ2isLQet2G6NU4pump", 0), # 0
    ("kevin","6Av5TuEQKtRVGrgQPNP6snTWQNHNjwqja574EQHGSWvB", 0), # 0
    ("clown","6YxDFVwMF5NaP5eMmu78ut8J8J8Qd83QgwBpm451pump", 0), # 0
    ("wave","HziLoXS5Tu1XyDrBpnN6T2ve2cg5ndCpNyiSiAeYpump", 1), # 1
    ("aizen","8DmNNJb6naVPZLaB4o64gf2jfKB73WqNKRCECmzepump", 1), # 1?
    ("bhc","7T4X8diCfk9Kkcm2n9ZayaCEmWmsAAjw4Z21zF7vpump", 0), # 0
    ("shtcoin","FZ9za81u9gLpWKjUzZ8h2xoeZxgfL826MtUhU6sQpump", 0), # 0
    ("bithuahua","6zxVEjmUVeLFPFRzD5h6fKHHvAEWNZ9caXN18GJNpump", 1), # 1
    ("trollingo","C8Yih2EauLzg3qQJFcAUVrZyrVuvjEGLot3YeE5cpump", 0), # 0
    ("dognald","GSmTGgbK5hmgqWaSk9xTUc8LUmPohaNqYj1SmBLSpump", 1), # 1
    ("petroll","MxJQbqUAGVG1X1xAEgbgWaERcccQSDpwCfFBjqCpump", 1), # 1
    ("skill", "Ar3CWwj4sxb5riXhMsszmLquQpd2SztFk5fjcp9Bpump", 0), # 0
    ("latinas", "7JG7HtTNTbrsTDqXPNDw2hbH7xoiRQFgL6BxN971pump", 1), # 1
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
    # ("", "", 0), # 0
]

# save token names and their labels
token_labels = {re.sub(r"[^\w\-]", "_", token) + ".pt": label for token, _, label in tokens}
with open(f"data/labels.json", "w") as f:
    json.dump(token_labels, f, indent=4)

tokens = tokens[16:]

# Token info (temp global variables)
token_name = ""
token_address = ""

# Collection algo args
get_cookies = True
headless = False

# File paths
COOKIES_PATH = "cookies.json"
DEV_NODES_PATH = "data/dev_nodes"
TOKEN_NODES_PATH = "data/token_nodes"
WALLET_NODES_PATH = "data/wallet_nodes"
TRANSACTION_DATA_PATH = "data/transaction_data"
WALLET_TOKEN_EDGES_PATH = "data/wallet_token_edges"
DEV_COIN_EDGES_PATH = "data/dev_coin_edges"
WALLET_WALLET_EDGES_PATH = "data/wallet_wallet_edges"
WALLET_DEV_EDGES_PATH = "data/wallet_dev_edges"

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

def read_transaction_data(token_name=None, sort=False):
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
    if not token_name:
        iterator = os.listdir(TRANSACTION_DATA_PATH)
    else:
        iterator = [token_name]
    
    all_transactions = {}
    for folder_name in iterator: # for each coin
        transactions = []

        # get all transactions for curr folder_name
        folder_path = os.path.join(TRANSACTION_DATA_PATH, folder_name)
        tx_files = sorted(os.listdir(folder_path), key=lambda x: extract_number(x)) if sort else os.listdir(folder_path)
        for file_name in tqdm(tx_files, desc=f"Reading {folder_name} transactions..."): # for each transaction
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                transactions.append(data)
        
        all_transactions[folder_name] = transactions

    print(f"Read transactions for {len(all_transactions.keys())} tokens.")
    
    if token_name:
        return transactions
    else:
        return all_transactions

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
    print("Getting dev-coin edge data...")
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
    print("Returned dev-coin edge data.")
    return edge_data

def save_wallet_data():
    """
    This function collects and saves wallet data from the Solscan website for each 
    wallet address associated with the transactions in the transaction data. It saves
    a JSON file for each wallet address in the WALLET_NODES_PATH directory containing 
    the wallet address, number of transfers, number of defi activities, and a rugpull 
    association boolean. This function also creates a logs/wallets_to_collect.log to 
    allow resuming after interruptions.
    """
    # if log file DNE or is empty, we start from scratch
    if not os.path.exists("logs/wallets_to_collect.log") or os.stat("logs/wallets_to_collect.log").st_size == 0:
        print("Getting wallet data...")
        # get all unique wallet addresses from the transaction data
        transactions = read_transaction_data(token_name)
        wallet_addresses = set()
        for transaction in transactions:
            wallet_addresses.add(transaction["wallet"])
        wallet_addresses = sorted(list(wallet_addresses))
        # save all wallet addresses in the log file
        os.makedirs("logs", exist_ok=True)
        with open("logs/wallets_to_collect.log", "w") as f:
            for wallet_address in wallet_addresses:
                f.write(f"{wallet_address}\n")
    # continue where we left off: from the first address in wallets_to_collect.log
    else:
        print("Continuing getting wallet data...")
        # read wallet addresses from log
        wallet_addresses = []
        with open("logs/wallets_to_collect.log", "r") as f:
            for line in f:
                wallet_addresses.append(line.strip())
        # get cookies again because interruption is likely due to human cookies expiring
        if not get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH, headless=headless)

    os.makedirs(WALLET_NODES_PATH, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH, headless=headless)
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

            # write the wallet data in its file
            with open(os.path.join(WALLET_NODES_PATH, f"{wallet_address}.json"), "w") as f:
                json.dump(wallet_data, f, indent=4)

            # remove the top most wallet address from the log since we have saved its data
            with open("logs/wallets_to_collect.log", "r") as f:
                lines = f.readlines()
                lines = lines[1:] # remove the first address
            with open("logs/wallets_to_collect.log", "w") as f:
                for line in lines:
                    f.write(line) # write all the addresses except for the first one
        browser.close()
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
        
        response = requests.get(url, headers=SOLTRACKER_HEADERS)
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
            cookie_loader.get_cookies(COOKIES_PATH, headless=headless)
        # load cookies from the saved file and set them in the context
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()

        # get dev address from specific div
        token_url = f"{BASE_SOLSCAN_URL}/token/{token_address}"
        page.goto(token_url)
        page.wait_for_load_state("networkidle")
        page.locator("div[class*='max-w-24/24']:has(div:text('Creator')) + div button").click()
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

def get_token_data():
    """
    Retrieves token attributes (id, lowest and highest price, rugpull flag, and dev id) from 
    Solana Tracker API.

    Returns:
        dict: A dictionary containing token_id, min/max_price, rugpull_flag, and dev_id
    """
    print("Getting coin data...")

    # Get min/max price from Solana Tracker API + transactions_data
    transaction_folder = os.path.join(TRANSACTION_DATA_PATH, token_name)
    transaction_files = sorted(
        os.listdir(transaction_folder),
        key = lambda x: extract_number(x) # sort by the number in the filename
    )
    first_tx_file = transaction_files[-1] # dev funding the token
    last_tx_file = transaction_files[0]
    
    first_tx_file_path = os.path.join(transaction_folder, first_tx_file)
    last_tx_file_path = os.path.join(transaction_folder, last_tx_file)

    with open(first_tx_file_path, "r") as f:
        first_tx_data = json.load(f)
    with open(last_tx_file_path, "r") as f:
        last_tx_data = json.load(f)

    first_tx_time = first_tx_data["time"]
    last_tx_time = last_tx_data["time"]
    assert last_tx_time > first_tx_time, "The timestamp of the last transaction must be after the first."

    params = {
        "token": token_address,
        "time_from": first_tx_time,
        "time_to": last_tx_time
    }
    
    response = requests.get(f"{BASE_SOLTRACKER_API_URL}/price/history/range", params=params, headers=SOLTRACKER_HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error fetching trades: {response.status_code}: {response.text}")
    
    data = response.json()
    min_price = data["price"]["lowest"]["price"]
    max_price = data["price"]["highest"]["price"]

    # Get dev address from dev_nodes
    dev_node_path = os.path.join(DEV_NODES_PATH, f"{token_name}.json")
    with open(dev_node_path, "r") as f:
        dev_data = json.load(f)
    dev_address = dev_data["dev_address"]

    token_data = {
        "token_address": token_address,
        "dev_address": dev_address,
        "min_price": min_price,
        "max_price": max_price,
        "rugpull": 0,
    }

    print("Token data retrieved.")
    return token_data

def get_transaction_signatures(wallet_address, rpc_url="https://api.mainnet-beta.solana.com", verbose=False):
    """
    Returns all transaction signatures for a Solana wallet.

    Args:
        wallet_address (str): Wallet address to fetch signatures for.
        rpc_url (str): Solana RPC URL (default: "https://api.mainnet-beta.solana.com").
        verbose (bool): If True, prints additional logs (default: False).

    Returns:
        set: Transaction signatures for the wallet, or None on failure.
    """
    os.makedirs(f"{WALLET_WALLET_EDGES_PATH}/{token_name}_transaction_signatures/", exist_ok=True)
    file_path = f"{WALLET_WALLET_EDGES_PATH}/{token_name}_transaction_signatures/{wallet_address}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            signature_set = set(line.strip() for line in f)
        return signature_set

    headers = {"Content-Type": "application/json"}

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [
            wallet_address,
        ]
    }

    response = requests.post(rpc_url, json=payload, headers=headers)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 2))+2
        if verbose:
            tqdm.write(f"\tRate limited (getSignaturesForAddress). Retrying after {retry_after}s...")
        time.sleep(retry_after)
        response = requests.post(rpc_url, json=payload, headers=headers)
        if response.status_code != 200:
            tqdm.write("SIGNATURES RETRY FAILED")
    elif response.status_code != 200:
        tqdm.write(f"\tERROR: Failed to fetch signatures for {wallet_address} (getSignaturesForAddress)")
        tqdm.write(f"\tError: {response.status_code}")
        tqdm.write(f"\t{response.headers}")
        tqdm.write(f"\t{response.content}")
        return

    signatures = response.json().get("result", [])
    signature_hashes = set()
    for signature in signatures:
        signature_hash = signature["signature"]
        signature_hashes.add(signature_hash)

    os.makedirs(f"{WALLET_WALLET_EDGES_PATH}/{token_name}_transaction_signatures/", exist_ok=True)
    with open(file_path, "w") as f:
        for signature in signature_hashes:
            f.write(f"{signature}\n")
    
    if verbose:
        tqdm.write(f"Got {len(signature_hashes)} tx_signatures for wallet {wallet_address[:5]}...")
    return signature_hashes

def parse_transaction(tx_data, signature, verbose=False):
    """
    A helper function that extracts and returns relevant data from the given transaction data.

    Args:
        tx_data (dict): The transaction data retrieved from the RPC.
        signature (str): The unique signature of the transaction.

    Returns:
        dict: A dictionary containing the parsed transaction details with the following keys:
            - "signature" (str): The transaction signature.
            - "from" (str or None): The sender's address, if identifiable.
            - "to" (str or None): The recipient's address, if identifiable.
            - "type" (str or None): The type of transaction (e.g., "transfer", "unknown").
            - "time" (int): The block time of the transaction (default is 0 if unavailable).
            - "amount" (float): The amount of tokens transferred (default is 0 if unavailable).
            - "token_address" (str or None): The address of the token involved in the transaction.
    """
    result = {
        "signature": signature,
        "from": None,
        "to": None,
        "type": None,
        "time": 0,
        "amount": 0,
        "token_address": None
    }

    # time
    result["time"] = tx_data.get("blockTime", 0)

    # tx type
    try:
        instructions = tx_data["transaction"]["message"]["instructions"]
        type = instructions[-1]["parsed"].get("type", None) if instructions else "unknown"
    except:
        type = "unknown"
    result["type"] = type

    # FROM and TO -- use pre/post token balances to determine value change
    try:
        pre_tokens = {b["owner"]: b for b in tx_data["meta"].get("preTokenBalances", [])}
        post_tokens = {b["owner"]: b for b in tx_data["meta"].get("postTokenBalances", [])}

        for owner in post_tokens:
            pre_amt = float(pre_tokens.get(owner, {}).get("uiTokenAmount", {}).get("uiAmount", 0))
            post_amt = float(post_tokens[owner]["uiTokenAmount"]["uiAmount"])

            if pre_amt > post_amt:
                result["from"] = owner
                result["token"] = post_tokens[owner]["mint"]
                result["amount"] = pre_amt - post_amt
            elif post_amt > pre_amt:
                result["to"] = owner
                result["token"] = post_tokens[owner]["mint"]
                result["amount"] = post_amt - pre_amt
    except Exception as e:
        if verbose:
            tqdm.write(f"ERROR parsing from, to, amount, and/or token for {signature}")
            tqdm.write(e.__str__())
        else:
            tqdm.write(f"ERROR parsing from, to, amount, and/or token for {signature[:5]}...")


    return result


def get_transfer_transactions(wallet1, wallet2, base_path, rpc_url="https://api.mainnet-beta.solana.com", verbose=False):
    """
    Returns a list of transfer transactions (as dicts) between two wallets.

    Args:
        wallet1 (str): The first wallet address.
        wallet2 (str): The second wallet address.
        base_path (str): The base path to save logs & data.
        rpc_url (str): The Solana RPC URL to use for fetching transaction data.
        verbose (bool): If True, print additional loggging information.
    
    Returns:
        list: A list of transfer transactions (as dicts) between the two wallets.
    """
    w1_signatures = get_transaction_signatures(wallet1, rpc_url, verbose=verbose)
    w2_signatures = get_transaction_signatures(wallet2, rpc_url, verbose=verbose)
    if not w1_signatures or not w2_signatures:
        return None
    
    common_signatures = w1_signatures.intersection(w2_signatures)
    if not common_signatures:
        return None

    if verbose:
        tqdm.write(f"Found {len(common_signatures)} common transaction signatures between wallets {wallet1[:5]}... and {wallet2[:5]}...")
    os.makedirs(f"{base_path}/{token_name}_raw/", exist_ok=True)
    with open(f"{base_path}/{token_name}_raw/common_signatures.txt", "a") as f:
        for signature in common_signatures:
            f.write(f"{signature}\n")

    transactions = []
    for signature in common_signatures:
        
        if os.path.exists(f"{base_path}/{token_name}_raw/{signature}.json"):
            continue
        
        if signature in open(f"{base_path}/{token_name}_raw/common_signatures.txt").read():
            continue

        # Get the transaction details
        headers = {"Content-Type": "application/json"}
        tx_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
        }

        response = requests.post(rpc_url, json=tx_payload, headers=headers)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2))+2
            if verbose:
                tqdm.write(f"\tRate limited (getTransaction). Retrying after {retry_after}s...")
            time.sleep(retry_after)
            response = requests.post(rpc_url, json=tx_payload, headers=headers)
            if response.status_code != 200:
                tqdm.write("TRANSACTION RETRY FAILED")
        elif response.status_code != 200:
            tqdm.write(f"\tERROR: Failed to fetch {signature} tx details (getTransaction)")
            tqdm.write(f"\tError: {response.status_code}")
            tqdm.write(f"\t{response.headers}")
            tqdm.write(f"\t{response.content}")
            continue
        
        tx_data = response.json().get("result")
        if not tx_data:
            tqdm.write(f"\tTransaction details for {signature} not found in data.")
            tqdm.write(f"\tError: {response.status_code}")
            tqdm.write(f"\t{response.headers}")
            tqdm.write(f"\t{response.content}")
            continue
        
        os.makedirs(f"{base_path}/{token_name}_raw/", exist_ok=True)
        with open(f"{base_path}/{token_name}_raw/{signature}.json", "w") as f:
            json.dump(tx_data, f, indent=4)

        if tx_data["meta"].get("preTokenBalances"): # if its in a valid format/ a token transfer (not sol transfer)
            tx_data = parse_transaction(tx_data, signature, verbose=verbose)
            transactions.append(tx_data)
            if verbose:
                tqdm.write(f"Returning transaction data for {signature[:5]}...")
        elif verbose:
            tqdm.write(f"SKIPPED transaction data for {signature[:5]}...")

    return transactions

def save_wallet_wallet_edges(base_path, verbose=False):
    """
    Derives wallet-wallet edges from transactions that include both sender and receiver wallets.

    Returns:
        list: A list of wallet-wallet edge dictionaries.
    """
    print("Getting wallet-wallet edges...")
    with open(os.path.join(DEV_NODES_PATH, f"{token_name}.json"), "r") as f:
        dev_address = json.load(f)["dev_address"]

    involved_wallets = set()
    transactions = read_transaction_data(token_name)
    print("num of transactions: ", len(transactions))
    for tx in transactions:
        if tx["wallet"] != dev_address:
            involved_wallets.add(tx["wallet"])
    print("num of involved wallets: ", len(involved_wallets))
    
    os.makedirs(f"{base_path}/{token_name}", exist_ok=True)
    
    # Generate all unique wallet pairs (no (A, A) and no (B, A) if (A, B) exists)
    wallet_pairs = list(combinations(involved_wallets, 2))
    print("num of wallet-wallet pairs: ", len(wallet_pairs))
    for w1, w2 in tqdm(wallet_pairs, desc="Getting wallet-wallet edges..."):
        transactions = get_transfer_transactions(w1, w2, base_path, verbose=verbose, rpc_url=f"https://rpc.shyft.to?api_key={SHYFT_API_KEY}")
        if transactions:
            for transaction in transactions:
                if not transaction["from"] or not transaction["to"]:
                    continue
                file_path = f"{base_path}/{token_name}/{transaction['from']}-{transaction['to']}.json"
                with open(file_path, "w") as f:
                    json.dump(transaction, f, indent=4)
                tqdm.write(f"Saved transaction data for {transaction['signature'][:5]}...")            
            
    print("Saved wallet-wallet edges.")

def save_wallet_dev_edges(base_path, verbose=False):
    """
    Constructs wallet-dev edge data using transactions that interact with the dev.

    Args:
        dev_address (str): The developer's wallet address.

    Returns:
        list: A list of wallet-dev edge dictionaries.
    """
    
    print("Getting wallet-dev edges...")
    # get dev address
    with open(os.path.join(DEV_NODES_PATH, f"{token_name}.json"), "r") as f:
        dev_address = json.load(f)["dev_address"]

    # get all buying and selling wallets from token transaction data
    involved_wallets = set()
    transactions = read_transaction_data(token_name)
    print("num of transactions: ", len(transactions))
    for tx in transactions:
        if tx["wallet"] != dev_address:
            involved_wallets.add(tx["wallet"])
    print("num of involved wallets: ", len(involved_wallets))
    
    os.makedirs(f"{base_path}/{token_name}", exist_ok=True)
    
    # Generate all wallet pairs with dev wallet
    wallet_pairs = [(dev_address, w) for w in involved_wallets]
    print("num of wallet-dev pairs: ", len(wallet_pairs))
    for w1, w2 in tqdm(wallet_pairs, desc="Getting wallet-wallet edges..."):
        # get all transactions between w1 and w2
        transactions = get_transfer_transactions(w1, w2, base_path, verbose=verbose, rpc_url=f"https://rpc.shyft.to?api_key={SHYFT_API_KEY}")
        # save all transactions between w1 and w2 (if any)
        if transactions:
            for transaction in transactions:
                if not transaction["from"] or not transaction["to"]:
                    continue
                file_path = f"{base_path}/{token_name}/{transaction['from']}-{transaction['to']}.json"
                with open(file_path, "w") as f:
                    json.dump(transaction, f, indent=4)
                tqdm.write(f"Saved transaction data for {transaction['signature'][:5]}...")            
            
    print("Saved wallet-dev edges.")

def get_wallet_token_edges():
    """
    Extracts wallet-coin edge data from transaction_data files.

    Returns:
        list: A list of wallet-coin edge dictionaries.
    """
    print("Getting wallet-coin edges...")
    with open(os.path.join(DEV_NODES_PATH, f"{token_name}.json"), "r") as f:
        dev_address = json.load(f)["dev_address"]

    edges = []
    transactions = read_transaction_data(token_name=token_name, sort=True)
    for tx in transactions[:-1]: # exclude the first transaction (dev funding the token)
        edge = {
            "tx_address": tx["tx"],
            "token_name": token_name,
            "token_address": token_address,
            "wallet_address": tx["wallet"],
            "type": tx["type"],
            "time": tx["time"],
            "amount": tx["amount"],
            "priceUsd": tx["priceUsd"],
            "volume": tx["volume"],
            "volumeSol": tx["volumeSol"],
        }
        edges.append(edge)
    print(f"Collected {len(edges)} wallet-coin edges.")
    return edges

def log_data_collection(data_type):
    """
    Logs the data collection status to a file.

    Args:
        data_type (str): The type of data that was collected.
    """
    with open("logs/completed_data_collection.log", "a") as f:
        f.write(f"{data_type}\n")

if __name__ == "__main__":

    # Save all tokens to a log file if it doesn't already exist
    os.makedirs("logs", exist_ok=True)
    tokens_log_path = "logs/tokens_to_collect.log"
    if not os.path.exists(tokens_log_path) or os.stat(tokens_log_path).st_size == 0:
        with open(tokens_log_path, "w") as f:
            for name, address, label in tokens:
                f.write(f"{name},{address},{label}\n")

    # Read tokens from the log file
    with open(tokens_log_path, "r") as f:
        tokens = [line.strip().split(",") for line in f]

    # Collect data for each token
    for name, address, label in tokens:
        print(f"Collecting data for {name} ({address[:5]}...)...")
        # Get the last data collected from the log file
        data_collection_log_path = "logs/completed_data_collection.log"
        if os.path.exists(data_collection_log_path):
            with open(data_collection_log_path, "r") as f:
                last_data_collected = [line.strip() for line in f.readlines()]
        else:
            last_data_collected = None

        token_name = re.sub(r"[^\w\-]", "_", name) # replace special characters with underscores
        token_address = address

        # get dev data and save it as a JSON in DEV_NODES_PATH
        if last_data_collected and "dev_nodes" in last_data_collected:
            print(f"Skipping dev_nodes collection since it was already collected.")
        else:
            dev_data = get_dev_data()
            os.makedirs(DEV_NODES_PATH, exist_ok=True)
            with open(os.path.join(DEV_NODES_PATH, f"{token_name}.json"), "w") as f:
                json.dump(dev_data, f, indent=4)
            log_data_collection("dev_nodes")
        
        # get transaction data and save it as a JSON files in TRANSACTION_DATA_PATH
        if last_data_collected and "transaction_data" in last_data_collected:
            print(f"Skipping transaction data collection since it was already collected.")
        else:
            save_transaction_data()
            log_data_collection("transaction_data")

        # try to save wallet data. if an error occurs, retry up to 15 times
        if last_data_collected and "wallet_nodes" in last_data_collected:
            print(f"Skipping wallet_nodes collection since it was already collected.")
        else:
            retries = 15
            for attempt in range(retries):
                try:
                    save_wallet_data()
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        print("Max retries reached. Exiting.")
                        raise
            log_data_collection("wallet_nodes")

        # get dev<->coin edge data and save it as a JSON in DEV_COIN_EDGES_PATH
        if last_data_collected and "dev_coin_edges" in last_data_collected:
            print(f"Skipping dev_coin_edges collection since it was already collected.")
        else:
            dev_coin_edge_data = get_dev_coin_edge()
            os.makedirs(DEV_COIN_EDGES_PATH, exist_ok=True)
            filename = f"{token_name}-{dev_coin_edge_data['dev_address']}.json"
            with open(os.path.join(DEV_COIN_EDGES_PATH, filename), "w") as f:
                json.dump(dev_coin_edge_data, f, indent=4)
            log_data_collection("dev_coin_edges")

        # get token data and save it as a JSON in TOKEN_NODES_PATH
        if last_data_collected and "token_nodes" in last_data_collected:
            print(f"Skipping token_nodes collection since it was already collected.")
        else:
            token_data = get_token_data()
            os.makedirs(TOKEN_NODES_PATH, exist_ok=True)
            with open(os.path.join(TOKEN_NODES_PATH, f"{token_name}.json"), "w") as f:
                json.dump(token_data, f, indent=4)
            log_data_collection("token_nodes")
            
        # get wallet<->token edges data and save them as JSON files in WALLET_TOKEN_EDGES_PATH/token_name/
        if last_data_collected and "wallet_token_edges" in last_data_collected:
            print(f"Skipping wallet_token_edges collection since it was already collected.")
        else:
            wallet_token_edges = get_wallet_token_edges()
            folder = os.path.join(WALLET_TOKEN_EDGES_PATH, token_name)
            os.makedirs(folder, exist_ok=True)
            for idx, edge in enumerate(wallet_token_edges):
                with open(os.path.join(folder, f"{edge['tx_address']}.json"), "w") as f:
                    json.dump(edge, f, indent=4)
            log_data_collection("wallet_token_edges")

        # get wallet<->wallet edges data and save them as JSON files in WALLET_WALLET_EDGES_PATH/token_name/
        # log files will be created in WALLET_DEV_EDGES_PATH/token_name/{token_name}_raw and {token_name}_transaction_signatures
        if last_data_collected and "wallet_wallet_edges" in last_data_collected:
            print(f"Skipping wallet_wallet_edges collection since it was already collected.")
        else:
            save_wallet_wallet_edges(verbose=False, base_path=WALLET_WALLET_EDGES_PATH)
            log_data_collection("wallet_wallet_edges")

        # get wallet<->dev edges data and save them as JSON files in WALLET_DEV_EDGES_PATH/token_name/
        # log files will be created in WALLET_DEV_EDGES_PATH/token_name/{token_name}_raw and {token_name}_transaction_signatures
        save_wallet_dev_edges(verbose=False, base_path=WALLET_DEV_EDGES_PATH)
        log_data_collection("wallet_dev_edges")

        # Remove the current token,address from the log file
        with open(tokens_log_path, "r") as f:
            lines = f.readlines()
        lines = [line for line in lines if address in line.strip()]
        with open(tokens_log_path, "w") as f:
            f.writelines(lines)
        
        # clear the log file for completed data collection
        with open(data_collection_log_path, "w") as f:
            pass

        print(f"Finished data collection for {token_name}!\n")
    print("Finished ALL data collection!")

# ERROR: POSSIBLE PROBLEM WITH API OF NOT RETURNING FIRST FEW TRANSACTION DATA i.e. the token creation/funding and some of the first few transactions
import requests
from datetime import datetime

def get_recent_token_tx(token_address, limit=50):
    url = f'https://public-api.solscan.io/token/txs?tokenAddress={token_address}&limit={limit}'
    headers = {'accept': 'application/json'}
    return requests.get(url, headers=headers).json()

def get_price_at_time(token_address, timestamp):
    url = f'https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}'
    res = requests.get(url).json()
    return float(res["pair"]["priceUsd"])

def scrape_coin_info(token_address):
    # Placeholder for actual Playwright scraping logic
    return {
        "coin_id": token_address,
        "rugpull_status": "Unknown",
        "developer_id": "DevAddress123"
    }

def build_graph(token_address):
    txs = get_recent_token_tx(token_address)
    coin_info = scrape_coin_info(token_address)

    graph = {
        "nodes": {
            "coins": [coin_info],
            "wallets": set(),
            "devs": [coin_info["developer_id"]]
        },
        "edges": []
    }

    for tx in txs:
        wallet = tx.get("owner")
        graph["nodes"]["wallets"].add(wallet)

        time_unix = tx.get("slotTime")
        timestamp = datetime.utcfromtimestamp(time_unix).isoformat()
        price = get_price_at_time(token_address, time_unix)
        amount = float(tx.get("changeAmount", 0))
        mc_at_tx = amount * price

        edge = {
            "from": wallet,
            "to": token_address,
            "type": "wallet-coin",
            "attributes": {
                "amount": amount,
                "tx_type": tx.get("changeType"),
                "time": timestamp,
                "mc_at_tx": mc_at_tx
            }
        }
        graph["edges"].append(edge)

        # Wallet-Dev edge (if buyer = dev)
        if wallet == coin_info["developer_id"]:
            dev_edge = {
                "from": wallet,
                "to": coin_info["developer_id"],
                "type": "wallet-dev",
                "attributes": {
                    "amount": amount
                }
            }
            graph["edges"].append(dev_edge)

    graph["nodes"]["wallets"] = list(graph["nodes"]["wallets"])
    return graph

# Run it
token_address = "YourTokenAddressHere"
graph = build_graph(token_address)

# Save to JSON
import json
with open("graph_data.json", "w") as f:
    json.dump(graph, f, indent=2)

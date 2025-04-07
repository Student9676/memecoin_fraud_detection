import requests
import json
from datetime import datetime

# BONK Token address
token_address = "DezXz1mHyvNQz8iFSAvyDb5XtfpZ7VGhvcE5ATdu5jCG"

# Fetch recent token transactions
def get_recent_token_tx(token_address, limit=50):
    url = f"https://public-api.solscan.io/token/holders?tokenAddress={token_address}&limit={limit}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    holders = response.json().get("data", [])

    txs = []
    for holder in holders:
        txs.append({
            "owner": holder["owner"],
            "changeAmount": holder.get("tokenAmount", {}).get("uiAmount", 0),
            "changeType": "TRANSFER",  # Solscan's holder API doesn't give type, so assume transfer
            "slotTime": int(datetime.now().timestamp())  # Mock timestamp (real logic would use tx history)
        })
    return txs

# Fetch current price (latest only)
def get_price_at_time(token_address, timestamp):
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}"
    response = requests.get(url)
    data = response.json()

    try:
        return float(data["pair"]["priceUsd"])
    except:
        return 0.0

# Scrape coin attributes (placeholder for Playwright)
def scrape_coin_info(token_address):
    # Normally you'd scrape sites like Rugcheck and Solscan here
    return {
        "coin_id": token_address,
        "rugpull_status": "Unknown",  # Placeholder
        "developer_id": "DevAddress123"  # Placeholder
    }

# Build the graph structure
def build_graph(token_address):
    graph = {
        "nodes": {
            "coins": [],
            "wallets": set(),
            "devs": set()
        },
        "edges": []
    }

    coin_info = scrape_coin_info(token_address)
    graph["nodes"]["coins"].append(coin_info)
    graph["nodes"]["devs"].add(coin_info["developer_id"])

    txs = get_recent_token_tx(token_address)

    for tx in txs:
        wallet = tx.get("owner")
        graph["nodes"]["wallets"].add(wallet)

        time_unix = tx.get("slotTime")
        timestamp = datetime.utcfromtimestamp(time_unix).isoformat()
        price = get_price_at_time(token_address, time_unix)
        amount = float(tx.get("changeAmount", 0))
        mc_at_tx = amount * price

        # Wallet-to-Coin edge
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

        # Wallet-to-Dev edge if dev is transacting
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

    # Convert sets to lists
    graph["nodes"]["wallets"] = list(graph["nodes"]["wallets"])
    graph["nodes"]["devs"] = list(graph["nodes"]["devs"])

    # Save to JSON file
    with open("graph_data.json", "w") as f:
        json.dump(graph, f, indent=2)

    return graph

# Run the build
if __name__ == "__main__":
    build_graph(token_address)

import requests
import json
import time

def get_transaction_signatures(wallet_address, rpc_url="https://api.mainnet-beta.solana.com"):
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
    if response.status_code != 200:
        print(f"Error: {response.content}")
        print(f"Failed to fetch signatures for {wallet_address}")
        return

    signatures = response.json().get("result", [])
    signature_hashes = set()
    for signature in signatures:
        signature_hash = signature["signature"]
        signature_hashes.add(signature_hash)

    print(f"Got {len(signature_hashes)} transactions for wallet {wallet_address[:5]}...")
    return signature_hashes

def parse_transaction(tx_data):
    result = {
        "from": None,
        "to": None,
        "type": None,
        "time": None,
        "amount": None,
        "token": None
    }

    # time
    result["time"] = tx_data["blockTime"]

    # tx type
    instructions = tx_data["transaction"]["message"]["instructions"]
    type = instructions[-1]["parsed"].get("type", None) if instructions else "unknown"
    result["type"] = type

    # FROM and TO -- use pre/post token balances to determine value change
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

    return result


def get_transfer_transactions(wallet1, wallet2, rpc_url="https://api.mainnet-beta.solana.com"):
    w1_signatures = get_transaction_signatures(wallet1, rpc_url)
    w2_signatures = get_transaction_signatures(wallet2, rpc_url)
    
    common_signatures = w1_signatures.intersection(w2_signatures)

    print(f"Found {len(common_signatures)} common transaction signatures between the two wallets.")

    for signature in common_signatures:
        
        # Get the transaction details
        headers = {"Content-Type": "application/json"}
        tx_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [signature, {"encoding": "jsonParsed"}]
        }

        response = requests.post(rpc_url, json=tx_payload, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {signature} details")
            continue
        
        tx_data = response.json().get("result")
        if not tx_data:
            print(f"Transaction {signature} not found in data.")
            continue
        
        tx_data = parse_transaction(tx_data)
        
        json.dump(tx_data, open(f"{signature}.json", "w"), indent=4)
        print(f"Transaction {signature[:5]}... details saved")

wallet1 = "DmbibsFFJp2SJ8VXtCRVBfyWYmSzQhCsH2mgDscUo6aF"
wallet2 = "G5PKPj8G23Gf8pHi93xVQ2zXxb8wGSosXVAjRZWXusVr"
get_transfer_transactions(wallet1, wallet2)
time.sleep(1)
get_transfer_transactions(wallet2, wallet1)

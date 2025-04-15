import json
import tqdm

def parse_transaction(tx_data, signature):
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
    except:
        tqdm.write(f"ERROR parsing from, to, amount, and/or token")

    return result

# Path to the JSON file
file_path = "/Users/raasikh/Documents/Coding/spring2025/cs485/final_project/data/wallet_wallet_edges/raw/1GCtyMZiFPDZF9UCJ9JSE1xP7UugWfVy1QvKAaSFbZdX9sMMK1e9NMyR7zEmaJPWZapQNV1RjZ4SRSgUEtLgtui.json"


# Read the JSON file

with open(file_path, 'r') as file:
    data = json.load(file)
    print("JSON data successfully loaded:")

print(parse_transaction(data, "1GCtyMZiFPDZF9UCJ9JSE1xP7UugWfVy1QvKAaSFbZdX9sMMK1e9NMyR7zEmaJPWZapQNV1RjZ4SRSgUEtLgtui"))
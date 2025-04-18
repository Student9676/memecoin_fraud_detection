import os
import json
from torch_geometric.data import HeteroData
import torch

def load_wallet_wallet_edges(path):
    edges = []
    weights = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx["from"] and tx["to"]:
                    edges.append((tx["from"], tx["to"]))
                    weights.append(tx["amount"])
    return edges, weights

def load_wallet_dev_edges(path):
    edges = []
    weights = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx["from"] and tx["to"]:
                    edges.append((tx["from"], tx["to"]))
                    weights.append(tx["amount"])
    return edges, weights

def load_wallet_token_edges(path):
    edges_buy = []
    edges_sell = []
    attrs_buy = []
    attrs_sell = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx["wallet_address"] and tx["token_address"]:
                    edge_type = tx["type"]
                    attrs = {
                        "amount": tx["amount"],
                        "priceUsd": tx["priceUsd"],
                        "volume": tx["volume"],
                        "volumeSol": tx["volumeSol"],
                        "time": tx["time"]
                    }
                    if edge_type == "buy":
                        edges_buy.append((tx["wallet_address"], tx["token_address"]))
                        attrs_buy.append(attrs)
                    elif edge_type == "sell":
                        edges_sell.append((tx["wallet_address"], tx["token_address"]))
                        attrs_sell.append(attrs)
    return edges_buy, attrs_buy, edges_sell, attrs_sell

def create_hetero_data(base_path, token_name):
    wallet_wallet_path = os.path.join(base_path, "wallet_wallet_edges", token_name)
    wallet_dev_path = os.path.join(base_path, "wallet_dev_edges", token_name)
    wallet_token_path = os.path.join(base_path, "wallet_token_edges", token_name)

    data = HeteroData()

    print("Loading wallet-wallet edges...")
    wallet_wallet_edges, wallet_wallet_weights = load_wallet_wallet_edges(wallet_wallet_path)

    print("Loading wallet-dev edges...")
    wallet_dev_edges, wallet_dev_weights = load_wallet_dev_edges(wallet_dev_path)

    print("Loading wallet-token edges...")
    buy_edges, buy_attrs, sell_edges, sell_attrs = load_wallet_token_edges(wallet_token_path)

    wallets = set()
    devs = set()
    tokens = set()

    for src, dst in wallet_wallet_edges:
        wallets.add(src)
        wallets.add(dst)
    for src, dst in wallet_dev_edges:
        wallets.add(src)
        devs.add(dst)
    for src, dst in buy_edges + sell_edges:
        wallets.add(src)
        tokens.add(dst)

    wallet_map = {k: i for i, k in enumerate(wallets)}
    dev_map = {k: i for i, k in enumerate(devs)}
    token_map = {k: i for i, k in enumerate(tokens)}

    data["wallet"].num_nodes = len(wallet_map)
    data["dev"].num_nodes = len(dev_map)
    data["token"].num_nodes = len(token_map)

    if wallet_wallet_edges:
        edge_index = torch.tensor([[wallet_map[src] for src, _ in wallet_wallet_edges],
                                   [wallet_map[dst] for _, dst in wallet_wallet_edges]], dtype=torch.long)
        data["wallet", "wallet_wallet", "wallet"].edge_index = edge_index
        data["wallet", "wallet_wallet", "wallet"].edge_weight = torch.tensor(wallet_wallet_weights, dtype=torch.float)

    if wallet_dev_edges:
        edge_index = torch.tensor([[wallet_map[src] for src, _ in wallet_dev_edges],
                                   [dev_map[dst] for _, dst in wallet_dev_edges]], dtype=torch.long)
        data["wallet", "wallet_dev", "dev"].edge_index = edge_index
        data["wallet", "wallet_dev", "dev"].edge_weight = torch.tensor(wallet_dev_weights, dtype=torch.float)

    if buy_edges:
        edge_index = torch.tensor([[wallet_map[src] for src, _ in buy_edges],
                                   [token_map[dst] for _, dst in buy_edges]], dtype=torch.long)
        data["wallet", "buys", "token"].edge_index = edge_index
        for key in buy_attrs[0]:
            data["wallet", "buys", "token"][key] = torch.tensor([attr[key] for attr in buy_attrs], dtype=torch.float)

    if sell_edges:
        edge_index = torch.tensor([[wallet_map[src] for src, _ in sell_edges],
                                   [token_map[dst] for _, dst in sell_edges]], dtype=torch.long)
        data["wallet", "sells", "token"].edge_index = edge_index
        for key in sell_attrs[0]:
            data["wallet", "sells", "token"][key] = torch.tensor([attr[key] for attr in sell_attrs], dtype=torch.float)

    return data

if __name__ == "__main__":
    base_path = r"/Users/drew/Desktop/data"
    token_name = "r_pwease"

    print(f"Creating HeteroData object for token: {token_name}")
    data = create_hetero_data(base_path, token_name)

    print(data)
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    

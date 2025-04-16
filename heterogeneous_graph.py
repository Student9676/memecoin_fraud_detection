import os
import json
import networkx as nx
from tqdm import tqdm

def load_wallet_wallet_edges(path):
    edges = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx["from"] and tx["to"]:
                    edges.append((tx["from"], tx["to"], {"type": "wallet-wallet", "weight": tx["amount"]}))
    return edges

def load_wallet_dev_edges(path):
    edges = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx["from"] and tx["to"]:
                    edges.append((tx["from"], tx["to"], {"type": "wallet-dev", "weight": tx["amount"]}))
    return edges

def load_wallet_token_edges(path):
    edges = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx["wallet_address"] and tx["token_address"]:
                    edge_type = f"wallet-token-{tx['type']}"  
                    edges.append((tx["wallet_address"], tx["token_address"], {
                        "type": edge_type,
                        "amount": tx["amount"],
                        "priceUsd": tx["priceUsd"],
                        "volume": tx["volume"],
                        "volumeSol": tx["volumeSol"],
                        "time": tx["time"]
                    }))
    return edges

def create_heterogeneous_graph(base_path, token_name):
    wallet_wallet_path = os.path.join(base_path, "wallet_wallet_edges", token_name)
    wallet_dev_path = os.path.join(base_path, "wallet_dev_edges", token_name)
    wallet_token_path = os.path.join(base_path, "wallet_token_edges", token_name)

    G = nx.DiGraph()

    print("Loading wallet-wallet edges...")
    wallet_wallet_edges = load_wallet_wallet_edges(wallet_wallet_path)
    G.add_edges_from(wallet_wallet_edges)
    print(f"Added {len(wallet_wallet_edges)} wallet-wallet edges.")

    print("Loading wallet-dev edges...")
    wallet_dev_edges = load_wallet_dev_edges(wallet_dev_path)
    G.add_edges_from(wallet_dev_edges)
    print(f"Added {len(wallet_dev_edges)} wallet-dev edges.")

    print("Loading wallet-token edges...")
    wallet_token_edges = load_wallet_token_edges(wallet_token_path)
    G.add_edges_from(wallet_token_edges)
    print(f"Added {len(wallet_token_edges)} wallet-token edges.")

    for node in G.nodes:
        if node.startswith("token_"):
            G.nodes[node]["type"] = "token"
        elif node.startswith("dev_"):
            G.nodes[node]["type"] = "dev"
        else:
            G.nodes[node]["type"] = "wallet"

    return G

def save_graph(G, base_path, token_name):
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, f"{token_name}_heterogeneous_graph.gpickle")
    nx.write_gpickle(G, path)
    print(f"Graph saved to {path}")

if __name__ == "__main__":
    base_path = "/Users/drew/Library/CloudStorage/OneDrive-Personal/cs485 final project" 
    token_name = "r_pwease"  

    print(f"Creating heterogeneous graph for token: {token_name}")
    G = create_heterogeneous_graph(base_path, token_name)
    save_graph(G, base_path, token_name)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

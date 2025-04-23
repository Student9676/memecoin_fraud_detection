import os
import json
import torch
from torch_geometric.data import HeteroData

def load_wallet_wallet_edges(path):
    edges = []
    weights = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx.get("from") and tx.get("to"):
                    edges.append((tx["from"], tx["to"]))
                    weights.append(tx.get("amount", 0.0))
    return edges, weights

def load_wallet_dev_edges(path):
    edges = []
    weights = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx.get("from") and tx.get("to"):
                    edges.append((tx["from"], tx["to"]))
                    weights.append(tx.get("amount", 0.0))
    return edges, weights

def load_wallet_token_edges(path):
    edges_buy, edges_sell = [], []
    attrs_buy, attrs_sell = [], []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx.get("wallet_address") and tx.get("token_address"):
                    attrs = {
                        "amount": tx.get("amount", 0.0),
                        "priceUsd": tx.get("priceUsd", 0.0),
                        "volume": tx.get("volume", 0.0),
                        "volumeSol": tx.get("volumeSol", 0.0),
                        "time": tx.get("time", 0.0)
                    }
                    if tx.get("type") == "buy":
                        edges_buy.append((tx["wallet_address"], tx["token_address"]))
                        attrs_buy.append(attrs)
                    elif tx.get("type") == "sell":
                        edges_sell.append((tx["wallet_address"], tx["token_address"]))
                        attrs_sell.append(attrs)
    return edges_buy, attrs_buy, edges_sell, attrs_sell

def load_dev_coin_edges(path):
    edges = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                dev = tx.get("dev_address")
                coin = tx.get("token_address") or tx.get("coin_id")
                if dev and coin:
                    edges.append((dev, coin))
    return edges

def load_dev_nodes(path):
    dev_ids = set()
    for fname in os.listdir(path):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname), "r") as f:
                dev = json.load(f)
                dev_ids.add(dev.get("dev_address"))
    return dev_ids

def create_hetero_data(base_path, token_name, save_path):
    # Paths
    wallet_wallet_path = os.path.join(base_path, "wallet_wallet_edges", token_name)
    wallet_dev_path = os.path.join(base_path, "wallet_dev_edges", token_name)
    wallet_token_path = os.path.join(base_path, "wallet_token_edges", token_name)
    dev_coin_path = os.path.join(base_path, "dev_coin_edges", token_name)
    dev_node_path = os.path.join(base_path, "dev_nodes")

    print(f"Creating HeteroData object for token: {token_name}")
    data = HeteroData()

    # Load all edges
    print("Loading wallet-wallet edges...")
    wallet_wallet_edges, wallet_wallet_weights = load_wallet_wallet_edges(wallet_wallet_path)

    print("Loading wallet-dev edges...")
    wallet_dev_edges, wallet_dev_weights = load_wallet_dev_edges(wallet_dev_path)

    print("Loading wallet-token edges...")
    buy_edges, buy_attrs, sell_edges, sell_attrs = load_wallet_token_edges(wallet_token_path)

    print("Loading dev-coin edges...")
    dev_coin_edges = load_dev_coin_edges(dev_coin_path)

    # Collect all unique nodes
    wallets, devs, tokens = set(), set(), set()

    for src, dst in wallet_wallet_edges:
        wallets.add(src)
        wallets.add(dst)
    for src, dst in wallet_dev_edges:
        wallets.add(src)
        devs.add(dst)
    for src, dst in buy_edges + sell_edges:
        wallets.add(src)
        tokens.add(dst)
    for src, dst in dev_coin_edges:
        devs.add(src)
        tokens.add(dst)

    # Include dev nodes from dev_nodes folder
    devs.update(load_dev_nodes(dev_node_path))

    # Map node IDs to indices
    wallet_map = {k: i for i, k in enumerate(wallets)}
    dev_map = {k: i for i, k in enumerate(devs)}
    token_map = {k: i for i, k in enumerate(tokens)}

    # Assign number of nodes
    data["wallet"].num_nodes = len(wallet_map)
    data["dev"].num_nodes = len(dev_map)
    data["token"].num_nodes = len(token_map)

    # Add edges to graph
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

    if dev_coin_edges:
        edge_index = torch.tensor([[dev_map[src] for src, _ in dev_coin_edges],
                                   [token_map[dst] for _, dst in dev_coin_edges]], dtype=torch.long)
        data["dev", "creates", "token"].edge_index = edge_index

    # Add placeholder features for all nodes that don't have them
    for node_type, node_data in data.items():
        if 'x' not in node_data:
            num_nodes = node_data.edge_index.shape[1] if 'edge_index' in node_data else 0
            # Placeholder feature: assign zeros or random values as features (size = num_nodes, feature_dim)
            data[node_type].x = torch.zeros(num_nodes, 64)  # 64 is an arbitrary feature size
            print(f"Warning: Node type '{node_type}' does not have 'x' feature. Assigning placeholder.")

    # Save to file
    filename = os.path.join(save_path, f"{token_name}_heterograph.pt")
    torch.save(data, filename)
    print(f"Heterogeneous graph saved to: {filename}")

    # Summary
    print(f"\nNode types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"Wallet nodes: {len(wallet_map)}, Dev nodes: {len(dev_map)}, Token nodes: {len(token_map)}")

    return data

def load_data_from_json(graph_path, labels_path):
    print(f"Loading graph from {graph_path}")  # Debug print to confirm the path
    try:
        graph = torch.load(graph_path)  # Try to load the graph
        print("Graph loaded successfully.")
    except Exception as e:
        print(f"Error loading graph: {e}")  # Print any error that occurs

    print(f"Loading labels from {labels_path}")  # Debug print for labels path
    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)  # Load labels from the JSON file
        print("Labels loaded successfully.")
    except Exception as e:
        print(f"Error loading labels: {e}")  # Print any error that occurs
    
    # Assuming you then need to convert the graph into a HeteroData object
    return graph

def main():
    base_path = "/Users/drew/Desktop/CS/CS 485/memecoin_fraud_detection/data"  # Adjust if needed
    save_path = "/Users/drew/Desktop/CS/CS 485/memecoin_fraud_detection/data/graphs"  # Folder to save the .pt file
    token_name = "r_pwease"  # Your target coin
    labels_path = '/Users/drew/Desktop/CS/CS 485/memecoin_fraud_detection/data/labels.json'  # Adjust this path as needed

    data = create_hetero_data(base_path, token_name, save_path)
    graph_path = "/Users/drew/Desktop/CS/CS 485/memecoin_fraud_detection/data/graphs"  # Update this path as needed
    loaded_graph = load_data_from_json(graph_path, labels_path)

if __name__ == "__main__":
    main()

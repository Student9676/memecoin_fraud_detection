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
                    weights.append([
                        tx.get("amount", 0.0),
                        tx.get("time", 0.0),
                    ])
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
                    weights.append(tx.get("amount", 0.0)) # no data for all tokens + add time here?
    return edges, weights

def load_wallet_token_edges(path, token_name):
    with open(os.path.join("data/dev_nodes", f"{token_name}.json"), "r") as f:
        dev_address = json.load(f)["dev_address"]
    edges_buy, edges_sell = [], []
    attrs_buy, attrs_sell = [], []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx.get("wallet_address") and tx.get("token_address"):
                    if tx.get("wallet_address") == dev_address: # dev buy/sell edges separate
                        continue
                    attrs = [
                        tx.get("amount", 0.0),
                        tx.get("priceUsd", 0.0),
                        tx.get("volume", 0.0),
                        tx.get("volumeSol", 0.0),
                        tx.get("time", 0.0)
                    ]
                    if tx.get("type") == "buy":
                        edges_buy.append((tx["wallet_address"], tx["token_address"]))
                        attrs_buy.append(attrs)
                    elif tx.get("type") == "sell":
                        edges_sell.append((tx["wallet_address"], tx["token_address"]))
                        attrs_sell.append(attrs)
    return edges_buy, attrs_buy, edges_sell, attrs_sell

def load_dev_buy_sell_edges(path, token_name):
    with open(os.path.join("data/dev_nodes", f"{token_name}.json"), "r") as f:
        dev_address = json.load(f)["dev_address"]
    edges_buy, edges_sell = [], []
    attrs_buy, attrs_sell = [], []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                if tx.get("wallet_address") and tx.get("token_address"):
                    if tx.get("wallet_address") != dev_address: # dev buy/sell edges separate
                        continue
                    attrs = [
                        tx.get("amount", 0.0),
                        tx.get("priceUsd", 0.0),
                        tx.get("volume", 0.0),
                        tx.get("volumeSol", 0.0),
                        tx.get("time", 0.0)
                    ]
                    if tx.get("type") == "buy":
                        edges_buy.append((tx["wallet_address"], tx["token_address"]))
                        attrs_buy.append(attrs)
                    elif tx.get("type") == "sell":
                        edges_sell.append((tx["wallet_address"], tx["token_address"]))
                        attrs_sell.append(attrs)
    return edges_buy, attrs_buy, edges_sell, attrs_sell

def load_dev_coin_edges(path, token_name):
    edges = []
    weights = []
    for filename in os.listdir(path):
        if filename.endswith(".json") and filename.startswith(token_name):
            with open(os.path.join(path, filename), "r") as f:
                tx = json.load(f)
                dev = tx.get("dev_address")
                coin = tx.get("token_address")
                edges.append((dev, coin))
                weights.append([
                    tx.get("creation_time", 0.0),
                    tx.get("starting_balance", 0.0),
                ])

    return edges, weights

def load_dev_nodes(path, token_name):
    dev_ids = set()
    for fname in os.listdir(path):
        if fname.endswith(".json") and fname.startswith(token_name):
            with open(os.path.join(path, fname), "r") as f:
                dev = json.load(f)
                dev_ids.add(dev.get("dev_address")) # num_tokens_created or num_rugpull_tokens_created?
    return dev_ids

# where are token nodes and wallet nodes????
# dev buys token, dev sells token???

def create_hetero_data(base_path, token_name, save_path):
    # Paths
    wallet_wallet_path = os.path.join(base_path, "wallet_wallet_edges", token_name)
    wallet_dev_path = os.path.join(base_path, "wallet_dev_edges", token_name)
    wallet_token_path = os.path.join(base_path, "wallet_token_edges", token_name)
    dev_coin_path = os.path.join(base_path, "dev_coin_edges")
    dev_node_path = os.path.join(base_path, "dev_nodes")

    print(f"Creating HeteroData object for token: {token_name}")
    data = HeteroData()

    # Load all edges
    print("Loading wallet-wallet edges...")
    wallet_wallet_edges, wallet_wallet_weights = load_wallet_wallet_edges(wallet_wallet_path)

    print("Loading wallet-dev edges...")
    wallet_dev_edges, wallet_dev_weights = load_wallet_dev_edges(wallet_dev_path)

    print("Loading wallet-token edges...")
    buy_edges, buy_attrs, sell_edges, sell_attrs = load_wallet_token_edges(wallet_token_path, token_name)

    print("Loading dev-coin edges...")
    dev_coin_edges, dev_coin_weights = load_dev_coin_edges(dev_coin_path, token_name)

    print("Loading dev buy/sell edges...")
    dev_buy_edges, dev_buy_attrs, dev_sell_edges, dev_sell_attrs = load_dev_buy_sell_edges(wallet_token_path, token_name)

    # Collect all unique nodes
    wallets, devs, tokens = set(), set(), set() # only one token and dev per graph

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
    devs.update(load_dev_nodes(dev_node_path, token_name))
    print(devs)

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
        data["wallet", "buys", "token"].edge_weight = torch.tensor(buy_attrs, dtype=torch.float)
    if sell_edges:
        edge_index = torch.tensor([[wallet_map[src] for src, _ in sell_edges],
                                   [token_map[dst] for _, dst in sell_edges]], dtype=torch.long)
        data["wallet", "sells", "token"].edge_index = edge_index
        data["wallet", "sells", "token"].edge_weight = torch.tensor(sell_attrs, dtype=torch.float)

    if dev_buy_edges:
        edge_index = torch.tensor([[dev_map[src] for src, _ in dev_buy_edges],
                                   [token_map[dst] for _, dst in dev_buy_edges]], dtype=torch.long)
        data["dev", "buys", "token"].edge_index = edge_index
        data["dev", "buys", "token"].edge_weight = torch.tensor(dev_buy_attrs, dtype=torch.float)
    if dev_sell_edges:
        edge_index = torch.tensor([[dev_map[src] for src, _ in dev_sell_edges],
                                    [token_map[dst] for _, dst in dev_sell_edges]], dtype=torch.long)
        data["dev", "sells", "token"].edge_index = edge_index
        data["dev", "sells", "token"].edge_weight = torch.tensor(dev_sell_attrs, dtype=torch.float)

    if dev_coin_edges:
        edge_index = torch.tensor([[dev_map[src] for src, _ in dev_coin_edges],
                                   [token_map[dst] for _, dst in dev_coin_edges]], dtype=torch.long)
        data["dev", "creates", "token"].edge_index = edge_index
        data["dev", "creates", "token"].edge_weight = torch.tensor(dev_coin_weights, dtype=torch.float)

    # dev node features
    with open(os.path.join("data/dev_nodes", f"{token_name}.json"), "r") as f:
        ft = json.load(f)["num_tokens_created"]
        data["dev"].x = torch.tensor([[ft]], dtype=torch.float)
    
    # wallet node features
    num_wallets = len(wallet_map)
    num_features = 2
    wallet_data = {}
    # get wallet features
    wallet_nodes_path = os.path.join(base_path, "wallet_nodes")
    for filename in os.listdir(wallet_nodes_path):
        with open(os.path.join(wallet_nodes_path, filename), "r") as f:
            wallet_info = json.load(f)
            wallet_address = wallet_info.get("wallet_address", None)
            if wallet_address:
                wallet_data[wallet_address] = [
                    wallet_info.get("num_transfers", 0.0), 
                    wallet_info.get("num_defi_activities", 0.0),
                ]
    # set wallet features
    wallet_fts = torch.zeros((num_wallets, num_features), dtype=torch.float)
    for wallet_id, idx in wallet_map.items():
        if wallet_id in wallet_data:
            wallet_fts[idx] = torch.tensor(wallet_data[wallet_id], dtype=torch.float)
        else:
            wallet_fts[idx] = torch.tensor([0.0] * num_features, dtype=torch.float) # handle missing data
    data["wallet"].x = wallet_fts

    # token node features
    token_nodes_path = os.path.join(base_path, "token_nodes", f"{token_name}.json")
    with open(token_nodes_path, "r") as f:
        token_data = json.load(f)
    token_fts = [
        token_data.get("min_price", 0.0),
        token_data.get("max_price", 0.0),
    ]
    data["token"].x = torch.tensor([token_fts], dtype=torch.float)

    # Save to file
    filename = os.path.join(save_path, f"{token_name}.pt")
    torch.save(data, filename)
    print(f"Heterogeneous graph saved to: {filename}")

    # Summary
    print(f"\nNode types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"Wallet nodes: {len(wallet_map)}, Dev nodes: {len(dev_map)}, Token nodes: {len(token_map)}")
    # Print the number of each type of edges
    print(f"Number of wallet-wallet edges: {len(wallet_wallet_edges)}")
    print(f"Number of wallet-dev edges: {len(wallet_dev_edges)}")
    print(f"Number of wallet-token buy edges: {len(buy_edges)}")
    print(f"Number of wallet-token sell edges: {len(sell_edges)}")
    print(f"Number of dev-coin edges: {len(dev_coin_edges)}")
    print(f"Number of dev buy edges: {len(dev_buy_edges)}")
    print(f"Number of dev sell edges: {len(dev_sell_edges)}")
    return data



def main():
    base_path = "data"  # Adjust if needed
    save_path = "heterographs"  # Folder to save the .pt file
    
    for token in os.listdir(os.path.join(base_path, "token_nodes")):
        token_name = token.split(".")[0]
        data = create_hetero_data(base_path, token_name, save_path)
    # # SHOW GRAPH AS HOMOGENEOUS GRAPH
    # import networkx as nx
    # from matplotlib import pyplot as plt
    # from torch_geometric.nn import to_hetero
    # import torch_geometric

    # g = torch_geometric.utils.to_networkx(data.to_homogeneous())
    # nx.draw(g, with_labels=True)
    # plt.show()

    # SHOW GRAPH AS HETEROGENEOUS GRAPH
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    graph = to_networkx(data, to_undirected=False)

    # Define colors for nodes and edges
    node_type_colors = {
        "dev": "#4599C3",
        "wallet": "#ED8546",
        "token": "#F2A900",
    }

    node_colors = []
    labels = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        color = node_type_colors[node_type]
        node_colors.append(color)
        if attrs["type"] == "dev":
            labels[node] = f"D{node}"
        elif attrs["type"] == "wallet":
            labels[node] = f"W{node}"
        elif attrs["type"] == "token":
            labels[node] = f"T{node}"

    # Define colors for the edges
    edge_type_colors = {
        ("wallet", "wallet_wallet", "wallet"): "#FF0000",  # Red
        ("wallet", "buys", "token"): "#D1FFBD",  # Very Light Green
        ("wallet", "sells", "token"): "#FFDADB",  # Very Light Red
        ("wallet", "wallet_dev", "dev"): "#8B0000",  # Dark Red
        ("dev", "creates", "token"): "#000000",  # Black
    }

    edge_colors = []
    edge_labels = {}
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        color = edge_type_colors[edge_type]

        graph.edges[from_node, to_node]["color"] = color
        edge_colors.append(color)

        # Add edge label
        edge_labels[(from_node, to_node)] = edge_type[1]

    # make the graph somewhat symmetrical
    pos = nx.spring_layout(graph, k=2)
    token_nodes = [node for node, attrs in graph.nodes(data=True) if attrs["type"] == "token"]
    for token_node in token_nodes:
        pos[token_node] = [0, 0]  # put token node at the center

    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=600,
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.show()

if __name__ == "__main__":
    main()

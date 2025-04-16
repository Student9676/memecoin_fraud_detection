import networkx as nx
import os
import json

def load_wallet_dev_edges(base_path, token_name):
    dev_edges_path = os.path.join(base_path, "wallet_dev_edges", token_name)
    dev_edges = []
    for file in os.listdir(dev_edges_path):
        with open(os.path.join(dev_edges_path, file), 'r') as f:
            transaction = json.load(f)
            dev_edges.append({
                'wallet_address': transaction['from'],
                'dev_address': transaction['to'],
                'amount': transaction['amount']
            })
    return dev_edges


def load_wallet_wallet_edges(base_path, token_name):
    wallet_wallet_edges_path = os.path.join(base_path, "wallet_wallet_edges", token_name)
    wallet_wallet_edges = []
    for file in os.listdir(wallet_wallet_edges_path):
        with open(os.path.join(wallet_wallet_edges_path, file), 'r') as f:
            transaction = json.load(f)
            wallet_wallet_edges.append({
                'wallet_1': transaction['from'],
                'wallet_2': transaction['to'],
                'amount': transaction['amount']
            })
    return wallet_wallet_edges


def load_wallet_token_edges(base_path, token_name):
    wallet_token_edges_path = os.path.join(base_path, "wallet_token_edges", token_name)
    wallet_token_edges = []
    for file in os.listdir(wallet_token_edges_path):
        with open(os.path.join(wallet_token_edges_path, file), 'r') as f:
            transaction = json.load(f)
            wallet_token_edges.append({
                'wallet_address': transaction['wallet'],
                'token_address': transaction['token'],
                'amount': transaction['amount']
            })
    return wallet_token_edges


def create_heterogeneous_graph(base_path, token_name):
    G = nx.DiGraph()
  
    wallet_dev_edges = load_wallet_dev_edges(base_path, token_name)
    wallet_wallet_edges = load_wallet_wallet_edges(base_path, token_name)
    wallet_token_edges = load_wallet_token_edges(base_path, token_name)

    for edge in wallet_dev_edges:
        G.add_edge(edge['wallet_address'], edge['dev_address'], type='wallet-dev', amount=edge['amount'])
    
    for edge in wallet_wallet_edges:
        G.add_edge(edge['wallet_1'], edge['wallet_2'], type='wallet-wallet', amount=edge['amount'])
    
    for edge in wallet_token_edges:
        G.add_edge(edge['wallet_address'], edge['token_address'], type='wallet-token', amount=edge['amount'])

    nx.write_gpickle(G, os.path.join(base_path, f"{token_name}_heterogeneous_graph.gpickle"))
    print(f"Graph with {len(G.nodes)} nodes and {len(G.edges)} edges created.")
  
    return G

import torch
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
import os
import json
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

labels_path = '/Users/drew/Desktop/CS/CS 485/memecoin_fraud_detection/data/labels.json'

class HGAT(torch.nn.Module):
    def __init__(self, metadata):
        super(HGAT, self).__init__()
    
        print("Initializing HGAT Model...")  # Debugging output
    
        # Modify the GATConv layers to handle edge features
        self.conv1 = HeteroConv({
            ('wallet', 'wallet_wallet', 'wallet'): GATConv(-1, 64, add_self_loops=False),
            ('wallet', 'buys', 'token'): GATConv(-1, 64, add_self_loops=False),
            ('wallet', 'sells', 'token'): GATConv(-1, 64, add_self_loops=False),
            ('dev', 'creates', 'token'): GATConv(-1, 64, add_self_loops=False),
        })
    
        self.conv2 = HeteroConv({
            ('wallet', 'wallet_wallet', 'wallet'): GATConv(64, 64, add_self_loops=False),
            ('wallet', 'buys', 'token'): GATConv(64, 64, add_self_loops=False),
            ('wallet', 'sells', 'token'): GATConv(64, 64, add_self_loops=False),
            ('dev', 'creates', 'token'): GATConv(64, 64, add_self_loops=False),
        })
    
        self.fc = torch.nn.Linear(64, 2)  # For binary classification (rugpull vs not)
    
        print("HGAT Model Initialized.")  # Debugging output

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        print("Running forward pass...")  # Debugging output
        # Passing edge features (if any) through the conv layers
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)  # First layer of message passing
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)  # Second layer of message passing
        
        token_features = x_dict['token']  # Aggregating the features of the 'token' node type for classification
        out = self.fc(token_features)  # Final classification layer
        return out

def load_data_from_json(graph_path, labels_path):
    print(f"Loading graph from {graph_path}...")  # Debugging output
    # Loading heterograph data
    data = HeteroData()

    # Load the graph
    try:
        graph = torch.load(graph_path)
        print(f"Graph loaded successfully from {graph_path}.")  # Debugging output
    except Exception as e:
        print(f"Error loading graph {graph_path}: {e}")  # Debugging output

    # Ensure 'x' exists for all node types and add default if missing
    for node_type, node_data in graph.items():
        print(f"Adding {node_type} data to the graph...")  # Debugging output
        
        # Check if 'x' (node features) exists, and add default if missing
        if 'x' in node_data:
            data[node_type].x = node_data['x']
            print(f"Node type '{node_type}' has 'x' feature.")  # Debugging output
        else:
            print(f"Warning: Node type '{node_type}' does not have 'x' feature. Assigning placeholder.")  # Debugging output
            num_nodes = node_data['edge_index'].shape[1] if 'edge_index' in node_data else 0
            data[node_type].x = torch.zeros(num_nodes, 64)  # Placeholder for node features (64 is arbitrary size, change as needed)

        # Add edge_index and edge_weight to the data
        data[node_type].edge_index = node_data.get('edge_index', None)
        data[node_type].edge_weight = node_data.get('edge_weight', None)

        # Handle additional edge features (e.g., amount, priceUsd, etc.)
        if 'amount' in node_data:
            data[node_type].amount = node_data['amount']
        if 'priceUsd' in node_data:
            data[node_type].priceUsd = node_data['priceUsd']
        if 'volume' in node_data:
            data[node_type].volume = node_data['volume']

    # Debugging: Check the contents of x_dict
    print(f"Available node types in x_dict: {data.x_dict.keys()}")  # Debugging output
    
    # Load the labels
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    print(f"Labels loaded from {labels_path}.")  # Debugging output
    
    # Convert labels to a tensor
    label_tensor = {}
    for token, label in labels.items():
        label_tensor[token] = torch.tensor(label, dtype=torch.long)
    
    print(f"Labels converted to tensors.")  # Debugging output

    # Add labels to the data object
    data.y_dict = label_tensor
    print(f"Labels added to data object.")  # Debugging output

    return data

# Training the model
def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    # Move data to the correct device
    data = data.to(device)

    # Forward pass with edge features (if available)
    out = model(data.x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
    
    # Calculate loss (cross-entropy for binary classification)
    loss = criterion(out, data.y_dict['token'])

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    return loss.item()

# Main function to train and evaluate the model
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")  # Debugging output

    # Create the models directory if it doesn't exist
    os.makedirs('solana_data/HGAT_models', exist_ok=True)

    # Load your heterograph data
    graphs_with_labels = []
    graph_dir = 'heterographs'
    labels_file = 'labels.json'
    
    print(f"Loading graphs from directory: {graph_dir}")  # Debugging output
    for graph_file in os.listdir(graph_dir):
        if graph_file.endswith('.pt'):
            graph_path = os.path.join(graph_dir, graph_file)
            data = load_data_from_json(graph_path, labels_path)
            graphs_with_labels.append((data, graph_file))

    print(f"Loaded {len(graphs_with_labels)} graphs.")  # Debugging output

    # Initialize the model, optimizer, and loss function
    if len(graphs_with_labels) > 0:
        print(f"Initializing the model...")  # Debugging output
        model = HGAT(graphs_with_labels[0][0].metadata()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(1, 101):
            total_loss = 0.0
            for data, _ in graphs_with_labels:
                loss = train(model, data, optimizer, criterion, device)
                total_loss += loss
            print(f"Epoch {epoch}, Loss: {total_loss}")

# Run the main function
if __name__ == "__main__":
    main()

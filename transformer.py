import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# === HGT Model Definition ===
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_heads=2):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()

        for node_type in metadata[0]:
            self.lin_dict[node_type] = torch.nn.Linear(-1, hidden_channels)

        self.conv1 = HGTConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             metadata=metadata, heads=num_heads)
        self.conv2 = HGTConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             metadata=metadata, heads=num_heads)

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x) if x.numel() > 0 else x
            for node_type, x in x_dict.items()
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.classifier(x_dict['token'])  # Predict on token nodes

# === Load graphs and labels ===
def load_graphs_with_labels(graph_dir, labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    graphs = []
    for fname in os.listdir(graph_dir):
        if fname.endswith("_heterograph.pt"):
            token_name = fname.split("_heterograph")[0]
            if token_name in labels:
                data = torch.load(os.path.join(graph_dir, fname))
                data.y = torch.tensor([labels[token_name]], dtype=torch.long)
                graphs.append(data)
    return graphs

# === Training & Evaluation ===
def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        # Mean logits over all token nodes (global prediction)
        pred = out.mean(dim=0, keepdim=True)
        loss = F.cross_entropy(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            pred = out.mean(dim=0, keepdim=True)
            predicted = pred.argmax(dim=1).item()
            all_preds.append(predicted)
            all_labels.append(batch.y.item())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1

# === Main Execution ===
def main():
    graph_dir = '/User/drew/Desktop/CS/CS 485/memecoin_fraud_detection/graphs'
    labels_path = '/User/drew/Desktop/data/labels.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs = load_graphs_with_labels(graph_dir, labels_path)
    print(f"Loaded {len(graphs)} graphs.")

    train_split = int(0.8 * len(graphs))
    train_graphs = graphs[:train_split]
    test_graphs = graphs[train_split:]

    train_loader = DataLoader(train_graphs, batch_size=1)
    test_loader = DataLoader(test_graphs, batch_size=1)

    metadata = train_graphs[0].metadata()
    model = HGT(hidden_channels=64, out_channels=2, metadata=metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    for epoch in range(1, 51):
        loss = train(model, optimizer, train_loader, device)
        acc, f1 = test(model, test_loader, device)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

if __name__ == '__main__':
    main()

import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, Linear, HeteroConv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set the correct data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Load node data
def load_json_folder(folder_path, key):
    data_list = []
    id_list = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.json'):
            with open(os.path.join(folder_path, fname)) as f:
                data = json.load(f)
                data_list.append(data)
                id_list.append(data.get(key))
    return data_list, id_list

wallet_nodes, wallet_ids = load_json_folder(os.path.join(data_dir, 'wallet_nodes'), 'wallet_address')
dev_nodes, dev_ids = load_json_folder(os.path.join(data_dir, 'dev_nodes'), 'dev_address')
token_nodes, token_ids = load_json_folder(os.path.join(data_dir, 'token_nodes'), 'token_address')

# Load labels
with open(os.path.join(data_dir, 'labels.json')) as f:
    token_labels = json.load(f)

data = HeteroData()

wallet_encoder = LabelEncoder()
dev_encoder = LabelEncoder()
token_encoder = LabelEncoder()

wallet_idx = wallet_encoder.fit_transform(wallet_ids)
dev_idx = dev_encoder.fit_transform(dev_ids)
token_idx = token_encoder.fit_transform(token_ids)

# Assign features
data['wallet'].x = torch.eye(len(wallet_ids))
data['dev'].x = torch.eye(len(dev_ids))
data['token'].x = torch.eye(len(token_ids))

# Assign labels to tokens
labels = [token_labels.get(tid, 0) for tid in token_ids]
data['token'].y = torch.tensor(labels)

# Train/test split
num_tokens = len(token_ids)
train_idx = torch.arange(0, int(0.8 * num_tokens))
test_idx = torch.arange(int(0.8 * num_tokens), num_tokens)
data['token'].train_mask = torch.zeros(num_tokens, dtype=torch.bool)
data['token'].train_mask[train_idx] = True
data['token'].test_mask = torch.zeros(num_tokens, dtype=torch.bool)
data['token'].test_mask[test_idx] = True

# Load edges
edge_mappings = [
    ('wallet', 'wallet_wallet', 'wallet', 'wallet_wallet_edges', 'from', 'to'),
    ('wallet', 'wallet_dev', 'dev', 'wallet_dev_edges', 'from', 'to'),
    ('wallet', 'buys', 'token', 'wallet_token_edges', 'wallet_address', 'token_address', 'buy'),
    ('wallet', 'sells', 'token', 'wallet_token_edges', 'wallet_address', 'token_address', 'sell'),
    ('dev', 'creates', 'token', 'dev_coin_edges', 'dev_address', 'token_address')
]

for edge_info in edge_mappings:
    src_type, rel_type, dst_type, folder, src_key, dst_key = edge_info[:6]
    type_filter = edge_info[6] if len(edge_info) > 6 else None
    edge_list = []
    edge_path = os.path.join(data_dir, folder)
    if not os.path.exists(edge_path):
        continue
    for fname in os.listdir(edge_path):
        if fname.endswith('.json'):
            with open(os.path.join(edge_path, fname)) as f:
                tx = json.load(f)
                if type_filter and tx.get("type") != type_filter:
                    continue
                src = tx.get(src_key)
                dst = tx.get(dst_key)
                if src and dst:
                    edge_list.append((src, dst))

    if not edge_list:
        continue

    if src_type == 'wallet':
        src_idx = wallet_encoder.transform([e[0] for e in edge_list])
    elif src_type == 'dev':
        src_idx = dev_encoder.transform([e[0] for e in edge_list])
    elif src_type == 'token':
        src_idx = token_encoder.transform([e[0] for e in edge_list])

    if dst_type == 'wallet':
        dst_idx = wallet_encoder.transform([e[1] for e in edge_list])
    elif dst_type == 'dev':
        dst_idx = dev_encoder.transform([e[1] for e in edge_list])
    elif dst_type == 'token':
        dst_idx = token_encoder.transform([e[1] for e in edge_list])

    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    data[(src_type, rel_type, dst_type)].edge_index = edge_index

# Define HGAT model
class HGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=2, heads=2):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(2):
            conv = HeteroConv({
                edge_type: GATConv((-1, -1), hidden_channels, heads=heads)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)

        self.classifiers = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.classifiers[node_type] = Linear(hidden_channels * heads, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_dict[k](v) for k, v in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        return {k: self.classifiers[k](v) for k, v in x_dict.items()}

# Train
model = HGAT(data.metadata()).to('cpu')
data = data.to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)['token']
    loss = F.cross_entropy(out[data['token'].train_mask], data['token'].y[data['token'].train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)['token']
            pred = out.argmax(dim=1)
            y_true = data['token'].y[data['token'].test_mask]
            y_pred = pred[data['token'].test_mask]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

# Save model
model_path = os.path.join(script_dir, 'hgat_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
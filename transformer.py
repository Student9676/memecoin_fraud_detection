import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, to_heterogeneous
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import networkx as nx

class HGTModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(HGTModel, self).__init__()
        self.conv1 = HGTConv(in_channels, hidden_channels, num_relations)
        self.conv2 = HGTConv(hidden_channels, out_channels, num_relations)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

def convert_to_pyg_data(G):
    data = HeteroData()
  
    for node in G.nodes:
        data[node].x = torch.randn((1, 16))  
    
    for edge in G.edges(data=True):
        src, dst, data = edge
        edge_type = data.get('type', 'wallet-wallet')  
        data_dict = {
            "edge_index": torch.tensor([[src], [dst]], dtype=torch.long)
        }
        data[edge_type].edge_index = torch.cat([data.get('edge_index', torch.tensor([])), data_dict["edge_index"]], dim=1)

    return data


def load_graph_and_convert(base_path, token_name):
    G = nx.read_gpickle(os.path.join(base_path, f"{token_name}_heterogeneous_graph.gpickle"))
    data = convert_to_pyg_data(G)
    return data

base_path = '/path/to/your/data'
token_name = 'your_token_name'
data = load_graph_and_convert(base_path, token_name)

in_channels = 16  
hidden_channels = 32
out_channels = 2  
num_relations = 3  
model = HGTModel(in_channels, hidden_channels, out_channels, num_relations)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
out = model(data)
print(out)

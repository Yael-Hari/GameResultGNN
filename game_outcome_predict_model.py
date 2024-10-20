import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import data_preperation
from gen_2nd_half_model import GraphGenerativeModel
import const

class GameOutcomePredictor(nn.Module):
    def __init__(
            self,
            gen_2nd_half_model: GraphGenerativeModel,
            in_channels=const.NUM_NODES_SINGLE_TEAM * 2, 
            hidden_channels=64, 
            num_classes=3
        ):
        super(GameOutcomePredictor, self).__init__()

        # gen_2nd_half_model
        self.gen_2nd_half_model = gen_2nd_half_model
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, data):
        
        # pass data through gen_2nd_half_model
        gnn_gen_model_output_data = self.gen_2nd_half_model(data)

        # concat the two graphs
        joined_data = data_preperation.join_graphs_data(
            gnn_gen_model_output_data=gnn_gen_model_output_data,
            init_data=data
        )
        x, edge_index, edge_weight, batch = \
            joined_data.x, joined_data.edge_index, joined_data.edge_attr, joined_data.batch

        # GCN layers with edge weights
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        
        # Global pooling to get graph-level representation
        x = self.global_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits for each class
        
        return x

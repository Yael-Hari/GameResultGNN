import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
import data_preperation
from gen_2nd_half_model import GraphGenerativeModel
import const

class GameOutcomePredictorV3(nn.Module):
    def __init__(
            self,
            gen_2nd_half_model: GraphGenerativeModel,
            in_channels=const.NUM_NODES_SINGLE_TEAM * 2, 
            hidden_channels=64, 
            num_classes=3,
            dropout_rate=0.5
        ):
        super(GameOutcomePredictorV3, self).__init__()

        # gen_2nd_half_model
        self.gen_2nd_half_model = gen_2nd_half_model
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
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

        # GraphSAGE layers
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        
        # Global pooling to get graph-level representation
        x = self.global_pool(x, batch)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits for each class
        
        return x
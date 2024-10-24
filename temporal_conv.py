import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import data_preperation
from gen_2nd_half_model import GraphGenerativeModel
import const

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Apply temporal convolution
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class GameOutcomePredictorV4(nn.Module):
    def __init__(
            self,
            gen_2nd_half_model: GraphGenerativeModel,
            in_channels=const.NUM_NODES_SINGLE_TEAM * 2,
            hidden_channels=64,
            temporal_channels=32,
            num_classes=3,
            kernel_size=3
        ):
        super(GameOutcomePredictorV4, self).__init__()

        # gen_2nd_half_model
        self.gen_2nd_half_model = gen_2nd_half_model
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Temporal convolution layers
        self.temporal_conv1 = TemporalConvLayer(hidden_channels, temporal_channels, kernel_size=kernel_size)
        self.temporal_conv2 = TemporalConvLayer(temporal_channels, temporal_channels, kernel_size=kernel_size)

        # Global pooling
        self.global_pool = global_mean_pool

        # Fully connected layers
        self.fc1 = nn.Linear(temporal_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # Pass data through gen_2nd_half_model
        gnn_gen_model_output_data = self.gen_2nd_half_model(data)

        # Concatenate the two graphs
        joined_data = data_preperation.join_graphs_data(
            gnn_gen_model_output_data=gnn_gen_model_output_data,
            init_data=data
        )
        
        # Extract graph data
        x, edge_index, edge_weight, batch = \
            joined_data.x, joined_data.edge_index, joined_data.edge_attr, joined_data.batch

        # GCN layers with edge weights
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        # Determine batch_size and number of nodes
        batch_size = batch.max().item() + 1
        num_nodes = x.shape[0]//batch_size  # Set to the correct number of nodes per graph
        
        # Reshape x for temporal convolution: (batch_size, num_nodes, hidden_channels)
        x = x.view(batch_size, num_nodes, -1).transpose(1, 2)  # Reshape to (batch, channels, nodes)

        # Temporal convolution layers
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)

        # Global pooling to get graph-level representation
        x = x.mean(dim=-1)  # Mean over the nodes (temporal dimension collapsed)

        # Make sure batch size and x dimensions align for global pooling
        x = self.global_pool(x, torch.arange(batch_size, device=x.device))  # Use correct batch indices

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits for each class

        return x


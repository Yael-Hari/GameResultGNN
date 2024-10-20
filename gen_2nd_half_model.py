# import pickle
# import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import const

"""
Next Steps:
Node Features: You may use various node features to represent the sectors, like sector ID, pass count, or pass intensity. If you don’t have features, a one-hot encoding for each node can act as initial embeddings.

Adding Temporal Aspects: Since you have the adjacency matrix split by time (e.g., every 5 minutes), you could:

Concatenate the layers as separate channels in the data.x feature.
Use a sequential model (like an LSTM or GRU) to process each layer/time interval individually and then pass these outputs to the GNN.
Edge Weighting: Use weighted adjacency matrices where the edge weights indicate the number of passes. You can add edge weights directly in GCNConv or pre-process the data to adjust for high-pass sectors.

Regularization: Apply dropout to fully connected layers to help with generalization.

Alternative Architectures: Test other architectures like Graph Attention Networks (GAT) to see if attention mechanisms can better capture sector importance.

"""

class GraphGenerativeModel(nn.Module):
    def __init__(
            self, 
            in_channels=const.NUM_NODES_SINGLE_TEAM * 2, 
            hidden_channels=64, 
            out_channels=32, 
            num_nodes=const.NUM_NODES_SINGLE_TEAM, 
            num_time_intervals=const.NUM_TIME_INTERVALS
        ):
        
        super(GraphGenerativeModel, self).__init__()
        
        # Define GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Compute the flattened size
        self.num_nodes_total = num_nodes * 2  # Total number of nodes (home and away)
        self.flattened_size = num_time_intervals * self.num_nodes_total * self.num_nodes_total  # 9 * 42 * 42
        
        # Fully connected layer to predict the flattened adjacency matrices
        self.fc = nn.Linear(out_channels, self.flattened_size)
        
        self.num_time_intervals = num_time_intervals
        
    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # GCN layers with edge weights
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        
        # Global pooling
        x = global_mean_pool(x, batch)  # Shape: [batch_size, out_channels]
        
        # Fully connected layer
        x = self.fc(x)  # Shape: [batch_size, flattened_size]
        
        # Output is already flattened
        adj_matrix_pred = x  # Shape: [batch_size, flattened_size]
        
        return adj_matrix_pred

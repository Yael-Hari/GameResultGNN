import pickle
import torch
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_time_intervals):
        super(GraphGenerativeModel, self).__init__()
        
        # Define GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Fully connected layer to predict the adjacency matrices
        self.fc = nn.Linear(out_channels, num_time_intervals * num_nodes * num_nodes * 2)
        
        self.num_nodes = num_nodes
        self.num_time_intervals = num_time_intervals
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)  # Shape: [batch_size, out_channels]
        
        # Fully connected layer
        x = self.fc(x)  # Shape: [batch_size, 9 * 21 * 42]
        
        # Reshape to [batch_size, 9, 21, 42]
        adj_matrix_pred = x.view(-1, self.num_time_intervals, self.num_nodes, self.num_nodes * 2)
        return adj_matrix_pred

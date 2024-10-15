import pickle
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import const

# def combined_adjacency_to_data(home_adj, away_adj, num_nodes):
#     """
#     Convert home and away adjacency matrices into a stacked PyTorch Geometric Data object with last-axis stacking.
#     num_nodes: Number of nodes in each of the graphs (home and away)
#     """
#     # Combine home and away adjacency matrices
#     stacked_adj = np.concatenate([home_adj, away_adj], axis=-1)  # Shape: (9, 42, 84)
    
#     edge_index_list = []
#     edge_weight_list = []
    
#     for time_layer in range(stacked_adj.shape[0]):
#         adj_layer = stacked_adj[time_layer]
#         row, col = np.nonzero(adj_layer)

#         # Ensure edges are within bounds by creating edge_index based on combined node count
#         edge_index = torch.tensor([row, col], dtype=torch.long)
#         edge_weight = torch.tensor(adj_layer[row, col], dtype=torch.float)
        
#         edge_index_list.append(edge_index)
#         edge_weight_list.append(edge_weight)
    
#     edge_index = torch.cat(edge_index_list, dim=1)
#     edge_weight = torch.cat(edge_weight_list)
    
#     """
#     Examples of Meaningful Node Features for Football Pass Analysis:
#     Pass Counts: For each sector (node), the total number of passes starting or ending in that sector could serve as a feature, giving a sense of sector activity.
#     Average Pass Distance: Nodes could carry information about the average distance of passes involving them, capturing spatial dynamics.
#     Possession Time: If you have data on how long each sector was occupied by a team, it could be helpful as a feature.
#     Heatmap Data: For each node, you could use normalized values indicating how often that sector is involved in gameplay.
#     Team-Specific Features: If home and away sectors differ significantly, adding team-specific indicators to distinguish between teams might be useful.
#     """

#     # Flatten the stacked adjacency matrices
#     adj_flat = stacked_adj.reshape(-1)
#     adj_tensor = torch.tensor(adj_flat, dtype=torch.float)
    
#     # Node features (adjust as needed)
#     x = torch.eye(num_nodes * 2, dtype=torch.float)  # Shape: [84, 84]

#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes*2)
#     data.adj = adj_tensor  # Store the target adjacency tensor


#     return data


def combined_adjacency_to_data(home_adj, away_adj, num_nodes):
    # Stack home and away adjacency matrices along the last dimension
    stacked_adj = np.concatenate([home_adj, away_adj], axis=-1)  # Shape: (9, 21, 42)
    
    # Flatten the stacked adjacency matrices
    adj_flat = stacked_adj.reshape(1, -1)  # Add batch dimension
    adj_tensor = torch.tensor(adj_flat, dtype=torch.float)  # Shape: [1, 9*21*42]
    
    # Node features
    x = torch.eye(num_nodes * 2, dtype=torch.float)  # Shape: [42, 42]
    
    data = Data(x=x, num_nodes=num_nodes * 2)
    data.adj = adj_tensor  # Store the adjacency tensor
    return data



# Prepare training dataset
def prepare_data(pkl_path, num_nodes, batch_size=4):
    # Load data
    with open(pkl_path, 'rb') as f:
        games_data = pickle.load(f)

    dataset = []
    for game in games_data:
        input_data = combined_adjacency_to_data(
            home_adj=game[0]['home_1st_half'], 
            away_adj=game[0]['away_1st_half'], 
            num_nodes=num_nodes)
        target_data = combined_adjacency_to_data(
            home_adj=game[0]['home_2nd_half'], 
            away_adj=game[0]['away_2nd_half'], 
            num_nodes=num_nodes)
        
        input_data.adj = target_data.adj  # Assign target adjacency tensor
        dataset.append(input_data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

if __name__ == '__main__':
    pkl_path = f'passes_data_{const.NUM_TIME_INTERVALS}_time_intervals.pkl'

    # Assuming games_data is the list of games as described
    num_nodes = const.NUM_SECTORS  # Adjust to your setting
    data_loader = prepare_data(pkl_path, num_nodes)


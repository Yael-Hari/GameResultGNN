import pickle
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import const


def combined_adjacency_to_data(home_adj, away_adj, num_nodes):
    num_nodes_total = num_nodes * 2  # Total number of nodes (home and away)

    # Initialize a zero adjacency matrix for each time interval
    stacked_adj = np.zeros((const.NUM_TIME_INTERVALS, num_nodes_total, num_nodes_total))

    # Place home_adj into the top-left block
    stacked_adj[:, :num_nodes, :num_nodes] = home_adj  # Shape: [9, 21, 21]

    # Place away_adj into the bottom-right block
    stacked_adj[:, num_nodes:, num_nodes:] = away_adj  # Shape: [9, 21, 21]

    # Flatten the stacked adjacency matrices for the target
    adj_flat = stacked_adj.reshape(-1)  # Shape: [flattened_size]
    adj_tensor = torch.tensor(adj_flat, dtype=torch.float)
    
    # Generate edge_index and edge_weight lists
    edge_index_list = []
    edge_weight_list = []
    
    for time_layer in range(stacked_adj.shape[0]):
        adj_layer = stacked_adj[time_layer]
        row, col = np.nonzero(adj_layer)
        edge_indices = torch.tensor([row, col], dtype=torch.long)
        edge_weights = torch.tensor(adj_layer[row, col], dtype=torch.float)
        
        edge_index_list.append(edge_indices)
        edge_weight_list.append(edge_weights)
    
    # Concatenate edge_index and edge_weight across all time layers
    edge_index = torch.cat(edge_index_list, dim=1)  # Shape: [2, total_num_edges]
    edge_weight = torch.cat(edge_weight_list)       # Shape: [total_num_edges]
    
    # Node features
    x = torch.eye(num_nodes_total, dtype=torch.float)  # Shape: [42, 42]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes_total)
    data.adj = adj_tensor  # Shape: [flattened_size]
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


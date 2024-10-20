import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import const


def combined_adjacency_to_data_for_gen_2nd_half_model(home_adj, away_adj, num_nodes, batch_size):
    if batch_size == 1 and len(home_adj.shape) == 3:
        assert len(away_adj.shape) == 3
        home_adj = torch.tensor(home_adj, dtype=torch.float).unsqueeze(0)
        away_adj = torch.tensor(away_adj, dtype=torch.float).unsqueeze(0)
    else:
        assert len(home_adj.shape) == 4
        assert len(away_adj.shape) == 4
        assert home_adj.shape[0] == away_adj.shape[0] == batch_size
        home_adj = torch.tensor(home_adj, dtype=torch.float)
        away_adj = torch.tensor(away_adj, dtype=torch.float)

    num_nodes_total = num_nodes * 2  # Total number of nodes (home and away)

    # Initialize a zero adjacency matrix for each time interval
    stacked_adj = np.zeros((batch_size, const.NUM_TIME_INTERVALS, num_nodes_total, num_nodes_total))

    # Place home_adj into the top-left block
    stacked_adj[:, :, :num_nodes, :num_nodes] = home_adj  # Shape: [4, 9, 21, 21]

    # Place away_adj into the bottom-right block
    stacked_adj[:, :, num_nodes:, num_nodes:] = away_adj  # Shape: [4, 9, 21, 21]

    # Flatten the stacked adjacency matrices for the target
    adj_flat = stacked_adj.reshape(-1)  # Shape: [flattened_size]
    adj_tensor = torch.tensor(adj_flat, dtype=torch.float)
    
    # Generate edge_index and edge_weight lists
    edge_index_list = []
    edge_weight_list = []
    
    # Iterate through each time layer in the stacked adjacency matrix
    for time_layer in range(stacked_adj.shape[1]):
        adj_layer = stacked_adj[:, time_layer]  # Shape: [batch_size, 42, 42]

        for i in range(batch_size):
            # Find non-zero elements in the adjacency matrix
            row, col = np.nonzero(adj_layer[i])
            # Create edge indices tensor from non-zero element positions
            edge_indices = torch.tensor([row, col], dtype=torch.long)
            # Create edge weights tensor from non-zero element values
            edge_weights = torch.tensor(adj_layer[i][row, col], dtype=torch.float)
        
            # Append edge indices and weights for this time layer
            edge_index_list.append(edge_indices)
            edge_weight_list.append(edge_weights)
    
    # Concatenate edge_index and edge_weight across all time layers
    # This creates a single edge_index tensor for all time layers
    edge_index = torch.cat(edge_index_list, dim=1)  # Shape: [2, total_num_edges]
    # creates a single edge_weight tensor for all time layers
    edge_weight = torch.cat(edge_weight_list)       # Shape: [total_num_edges]
    
    # Node features
    x_list = []
    for _ in range(batch_size):
        x = torch.eye(num_nodes_total, dtype=torch.float)  # Shape: [42, 42]
        x_list.append(x)
    x = torch.cat(x_list, dim=0)  # Shape: [batch_size * 42, 42]
    # x = x.view(-1, num_nodes_total)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes_total)
    data.adj = adj_tensor  # Shape: [flattened_size]
    return data


# Prepare training dataset
def prepare_data_for_gen_2nd_half_model(
        pkl_path, 
        num_nodes, 
        batch_size=4,
        train_test_split=0.8
    ):
    # Load data
    with open(pkl_path, 'rb') as f:
        games_data = pickle.load(f)

    dataset = []
    for game in games_data:
        input_data = combined_adjacency_to_data_for_gen_2nd_half_model(
            home_adj=game[0]['home_1st_half'], 
            away_adj=game[0]['away_1st_half'], 
            num_nodes=num_nodes,
            batch_size=1
        )
        target_data = combined_adjacency_to_data_for_gen_2nd_half_model(
            home_adj=game[0]['home_2nd_half'], 
            away_adj=game[0]['away_2nd_half'], 
            num_nodes=num_nodes,
            batch_size=1
        )
        game_outcome = game[0]['game_outcome']  # 0 for draw, 1 for home win, 2 for away win
        input_data.game_outcome = game_outcome  # Assign game outcome
        input_data.y = np.zeros(3)
        input_data.y[game_outcome] = 1
        input_data.y = torch.tensor(input_data.y, dtype=torch.float)
        input_data.adj = target_data.adj  # Assign target adjacency tensor
        dataset.append(input_data)

    # Split the dataset into train and test sets
    train_size = int(len(dataset) * train_test_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders for train and test sets
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader

def prepare_data_for_outcome_predict_model(first_half_adj, second_half_pred_adj, num_nodes, label):
    num_nodes_total = num_nodes * 2  # Total number of nodes (home and away)
    
    # Combine first half and predicted second half adjacency matrices along the time dimension
    combined_adj = torch.cat((first_half_adj, second_half_pred_adj), dim=0)  # Shape: [num_time_intervals * 2, num_nodes_total, num_nodes_total]
    
    # Generate edge_index and edge_weight lists
    edge_index_list = []
    edge_weight_list = []
    for time_layer in range(combined_adj.shape[0]):
        adj_layer = combined_adj[time_layer]
        row, col = torch.nonzero(adj_layer, as_tuple=True)
        edge_indices = torch.stack([row, col], dim=0)
        edge_weights = adj_layer[row, col]
        
        edge_index_list.append(edge_indices)
        edge_weight_list.append(edge_weights)
    
    # Concatenate edge_index and edge_weight across all time layers
    edge_index = torch.cat(edge_index_list, dim=1)  # Shape: [2, total_num_edges]
    edge_weight = torch.cat(edge_weight_list)       # Shape: [total_num_edges]
    
    # Node features (adjust as needed)
    x = torch.eye(num_nodes_total, dtype=torch.float)  # Shape: [num_nodes_total, num_nodes_total]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    data.y = torch.tensor([label], dtype=torch.long)  # Game outcome label
    return data


def reshape_model_1_output(graph_data_1):
    """
    @param graph_data_1: the output data from the gen GNN model. its a vector with Shape: [batch_size, flattened_size]
    @return: the reshaped graph data
    """

    graph_data_1_reshaped = graph_data_1.view(
        -1,  # batch_size
        const.NUM_TIME_INTERVALS, 
        const.NUM_NODES_SINGLE_TEAM * 2, 
        const.NUM_NODES_SINGLE_TEAM * 2
    )
    home_2nd_half = graph_data_1_reshaped[:, :, :const.NUM_NODES_SINGLE_TEAM, :const.NUM_NODES_SINGLE_TEAM]
    away_2nd_half = graph_data_1_reshaped[:, :, const.NUM_NODES_SINGLE_TEAM:, const.NUM_NODES_SINGLE_TEAM:]
    generated_2nd_half_data = combined_adjacency_to_data_for_gen_2nd_half_model(
        home_adj=home_2nd_half,
        away_adj=away_2nd_half,
        num_nodes=const.NUM_NODES_SINGLE_TEAM,
        batch_size=graph_data_1_reshaped.shape[0]
    )
    return generated_2nd_half_data


def join_graphs_data(init_data, gnn_gen_model_output_data):
    """
    @param init_data: the original data
    @param gnn_gen_model_output_data: the output data from the gen GNN model. its a vector
    @return: the joined graph data
    """

    x_1, edge_index_1, edge_weight_1, batch_1 = \
        init_data.x, \
        init_data.edge_index, \
        init_data.edge_attr, \
        init_data.batch

    # reshape the gnn_gen_model_output_data
    gnn_gen_model_output_data_reshaped = reshape_model_1_output(gnn_gen_model_output_data)

    x_2, edge_index_2, edge_weight_2 = \
        gnn_gen_model_output_data_reshaped.x, \
        gnn_gen_model_output_data_reshaped.edge_index, \
        gnn_gen_model_output_data_reshaped.edge_attr, \

    # concat the two graphs
    x = (x_1 + x_2) / 2
    edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
    edge_weight = torch.cat([edge_weight_1, edge_weight_2])
    batch = torch.cat([batch_1])

    # create the new data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, batch=batch)

    # assign the label
    data.game_outcome = init_data.game_outcome
    data.y = init_data.y

    return data

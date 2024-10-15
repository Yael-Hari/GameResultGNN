from tqdm import tqdm
import torch
import torch.nn.functional as F
from gen_2nd_half_model import GraphGenerativeModel
from data_preperation import prepare_data
import const


# Training function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        adj_pred = model(data)  # Shape: [batch_size, 9, 21, 42]
        
        # Reshape adj_target to match adj_pred
        adj_target = data.adj.view(adj_pred.size())
        
        loss = F.mse_loss(adj_pred, adj_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)


if __name__ == '__main__':
    num_nodes = const.NUM_SECTORS
    pkl_path = f'passes_data_{const.NUM_TIME_INTERVALS}_time_intervals.pkl'
    data_loader = prepare_data(pkl_path, num_nodes)

    # Initialize the model, optimizer, and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphGenerativeModel(
        in_channels=num_nodes * 2, 
        hidden_channels=64, 
        out_channels=32, 
        num_nodes=num_nodes,
        num_time_intervals=const.NUM_TIME_INTERVALS
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Full training loop
    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, data_loader, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

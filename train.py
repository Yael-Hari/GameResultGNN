import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from gen_2nd_half_model import GraphGenerativeModel
from data_preperation import prepare_data_for_gen_2nd_half_model
from game_outcome_predict_model import GameOutcomePredictor
from temporal_conv import GameOutcomePredictorV4
from outcome_model_for_compare1 import GameOutcomePredictorV2
from outcome_model_for_compare2 import GameOutcomePredictorV3
import const


# Training epoch function
def train_epoch_model1(model, data_loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        adj_pred = model(data)  # Shape: [batch_size, flattened_size]
        adj_target = data.adj.to(device)  # Shape: [batch_size, flattened_size]
        
        # Reshape adj_target to match adj_pred
        adj_target = data.adj.view(adj_pred.size())
        
        loss = criterion(adj_pred, adj_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(data_loader)

def train_epoch_model2(model, data_loader, optimizer, device, criterion):
    model.train()  # Sets the model to training mode, activating dropout and batch normalization
    total_loss = 0  # Initializes the total loss for the epoch
    correct = 0  # Initializes the count of correct predictions

    for data in tqdm(data_loader, desc='Training', total=len(data_loader)): 
        data = data.to(device)  # Moves the data to the specified device (CPU or GPU)
        optimizer.zero_grad()  # Resets the gradients of the optimizer
        
        out = model(data)  # Forward pass: computes the model's output
        target = data.y.view(-1, 3)
        loss = criterion(out, target.to(device))  # Calculates the loss between predictions and true labels
        loss.backward()  # Computes gradients of the loss with respect to model parameters
        optimizer.step()  # Updates model parameters based on computed gradients
        
        total_loss += loss.item()  # Accumulates the loss for this batch
        pred = out.argmax(dim=1)  # Gets the predicted class (highest probability)
        correct += pred.eq(data.game_outcome).sum().item()  # Counts correct predictions

    mean_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)  # Calculates overall accuracy for the epoch
    
    return mean_loss, accuracy  # Returns average loss and accuracy

def test_epoch_model2(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Eval on Test Set', total=len(data_loader)):
            data = data.to(device)
            out = model(data)
            target = data.y.view(-1, 3)
            loss = criterion(out, target.to(device))  # Calculates the loss between predictions and true labels
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(data.game_outcome).sum().item()
    
    mean_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    
    return mean_loss, accuracy


def train_gnn_gen_model(
        data_loader, 
        model_1_save_path,
        num_epochs=20,
        learning_rate=0.001
    ):

    # Initialize device, model, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphGenerativeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train model 1
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch_model1(
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            device=device,
            criterion=criterion
        )
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')
    
    # Save model 1
    torch.save(model.state_dict(), model_1_save_path)
    return model

def load_model_1(model_1_save_path, data_pkl_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1 = GraphGenerativeModel(
        in_channels=const.NUM_NODES_SINGLE_TEAM * 2, 
        hidden_channels=64, 
        out_channels=32, 
        num_nodes=const.NUM_NODES_SINGLE_TEAM,
        num_time_intervals=const.NUM_TIME_INTERVALS
        ).to(device)
    
    model_1.load_state_dict(torch.load(model_1_save_path))
    return model_1

def train_model(
        model_1: GraphGenerativeModel, 
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        model_2_save_path: str,
        num_epochs: int = 20,
        learning_rate: float = 0.001
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_2 = GameOutcomePredictorV3(gen_2nd_half_model=model_1).to(device)
    optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model 2
    test_accuracies = []
    losses = []
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch_model2(
            model=model_2, 
            data_loader=train_data_loader, 
            optimizer=optimizer, 
            device=device,
            criterion=criterion)
        
        test_loss, test_accuracy = test_epoch_model2(
            model=model_2, 
            data_loader=test_data_loader, 
            device=device,
            criterion=criterion)
        
        test_accuracies.append(test_accuracy)
        losses.append(train_loss)
        
        print(
            '*****************************************\n'
            f'Epoch {epoch + 1}/{num_epochs} ||| '
            f'Train Loss: {train_loss:.4f} | '
            f'Test Loss: {test_loss:.4f} ||| '
            f'Train Accuracy: {train_accuracy:.4f} | '
            f'Test Accuracy: {test_accuracy:.4f}'
            '\n*****************************************'
        )
    
    # Save model 2
    torch.save(model_2.state_dict(), model_2_save_path)
    return model_2, test_accuracies, losses

def main():

    # train model 1 separately (if not already trained)
    model_1_save_path = 'model1_state_dict.pth'
    model_2_save_path = 'model2_state_dict.pth'
    data_pkl_path = f'passes_data_{const.NUM_TIME_INTERVALS}_time_intervals.pkl'

    # more params
    learning_rate = 0.001

    train_data_loader, test_data_loader = prepare_data_for_gen_2nd_half_model(
        pkl_path=data_pkl_path,
        num_nodes=const.NUM_NODES_SINGLE_TEAM,
        batch_size=const.BATCH_SIZE,
        train_test_split=0.8
    )

    if not os.path.exists(model_1_save_path):
        # train model 1 separately
        model_1 = train_gnn_gen_model(
            data_loader=train_data_loader,
            model_1_save_path=model_1_save_path,
            num_epochs=50,
            learning_rate=learning_rate
        )
    else:
        # load model 1 from model_1_save_path
        model_1 = load_model_1(model_1_save_path, data_pkl_path)

    # train the whole model using model 1
    model_2, test_accuracies, losses = train_model(
        model_1=model_1, 
        model_2_save_path=model_2_save_path,
        train_data_loader=train_data_loader, 
        test_data_loader=test_data_loader,
        learning_rate=learning_rate
    )

    with open('test_accuracies_of_SAGE.txt', 'w') as f:
        for accuracy in test_accuracies:
            f.write(f'{accuracy:.4f}\n')  # Write each accuracy on a new line
            
    with open('train_loss_of_SAGE.txt', 'w') as f:
        for loss in losses:
            f.write(f'{loss:.4f}\n')  # Write each accuracy on a new line

if __name__ == '__main__':
    main()
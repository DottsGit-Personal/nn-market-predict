import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

def train_model(model, X_train, y_train, batch_size=32, num_epochs=200, learning_rate=0.001):
    # Convert data to numpy arrays if they're DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values
    
    # Ensure y_train is 2D
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    prev_lr = learning_rate
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        
        # Print average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Adjust learning rate
        scheduler.step(avg_loss)
        
        # Print current learning rate if it has changed
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or current_lr != prev_lr:
            print(f'Learning rate adjusted to: {current_lr}')
        prev_lr = current_lr
    
    return model

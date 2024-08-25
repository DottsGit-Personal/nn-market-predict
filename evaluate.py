import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  # Add this import

def evaluate_model(model, X_test, y_test):
    # Determine the device of the model
    device = next(model.parameters()).device
    
    # Convert input data to PyTorch tensors and move to the same device as the model
    if isinstance(X_test, pd.DataFrame):
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    elif isinstance(X_test, np.ndarray):
        X_test_tensor = torch.FloatTensor(X_test).to(device)
    else:
        X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
        y_test_tensor = torch.FloatTensor(y_test.values).to(device)
    elif isinstance(y_test, np.ndarray):
        y_test_tensor = torch.FloatTensor(y_test).to(device)
    else:
        y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Ensure y_test_tensor is 2D
    if y_test_tensor.dim() == 1:
        y_test_tensor = y_test_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    model.eval()

    # Perform the forward pass
    with torch.no_grad():
        y_pred = model(X_test_tensor)

    # Move predictions back to CPU for numpy operations
    y_pred = y_pred.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2

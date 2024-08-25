import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # Convert DataFrame to numpy array if it's not already
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred = model(X_test_tensor).numpy()
        
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    
    return mse, rmse, r2
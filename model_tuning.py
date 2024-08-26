import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from model import create_model

def objective(trial, X, y, num_epochs=100):
    # Define hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_sizes = [trial.suggest_int(f'hidden_size_{i}', 32, 256) for i in range(n_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Create model
    input_size = X.shape[1]
    output_size = 1
    model = create_model(input_size, hidden_sizes, output_size, dropout_rate)

    # Define loss and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cv_scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Scale the data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            cv_scores.append(val_loss.item())

    return np.mean(cv_scores)

def tune_hyperparameters(X, y, n_trials=100):
    import optuna

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return trial.params

# Usage example:
# best_params = tune_hyperparameters(X, y)
# best_model = create_model(input_size, best_params['hidden_sizes'], output_size, best_params['dropout_rate'])

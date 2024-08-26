import yfinance as yf
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
from data_fetcher import get_data_for_model
from data_loader import load_data
from model import create_model
from train import train_model
from evaluate import evaluate_model
from model_tuning import tune_hyperparameters

def main(ticker='QQQ', start_date='2010-01-01', end_date=None, lookback_days=20, hidden_sizes=[128, 64, 32], n_trials=100):
    # If end_date is not provided, use yesterday's date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    # Fetch and prepare data
    data, current_price = get_data_for_model(ticker, start_date, end_date, lookback_days)
    
    if data is None or data.empty:
        print(f"No data available for {ticker} in the specified date range.")
        return
    
    data.to_csv(f'{ticker}_data.csv', index=False)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_data(f'{ticker}_data.csv', 'Target')
    
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure X_train and y_train are numpy arrays
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) else y_train

    # Tune hyperparameters
    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(X_train_array, y_train_array, n_trials=n_trials)
    print("Best hyperparameters:", best_params)

    # Extract hidden sizes from best_params
    n_layers = best_params['n_layers']
    hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]

    # Create model with best parameters and move it to the appropriate device
    input_size = X_train.shape[1]
    output_size = 1  # Predicting a single value (next day's percentage change)
    model = create_model(input_size, hidden_sizes, output_size, best_params['dropout_rate']).to(device)
    
    # Train model
    trained_model = train_model(model, X_train, y_train, num_epochs=300, 
                                batch_size=best_params['batch_size'], 
                                learning_rate=best_params['learning_rate'], 
                                device=device)
    
    # Evaluate model
    mse, rmse, r2 = evaluate_model(trained_model, X_test, y_test)
    
    # Make a prediction for the next day
    last_known_data = torch.FloatTensor(X_test.iloc[-1].values).unsqueeze(0).to(device)
    with torch.no_grad():
        next_day_change_scaled = trained_model(last_known_data).cpu().item()
    
    # Inverse transform the prediction
    next_day_change = scaler_y.inverse_transform([[next_day_change_scaled]])[0][0]
    
    # Calculate the predicted price
    predicted_price = current_price * (1 + next_day_change)
    
    print(f"\nLast known price of {ticker}: ${current_price:.2f}")
    print(f"Predicted percentage change: {next_day_change*100:.2f}%")
    print(f"Prediction for the next trading day's closing price of {ticker}: ${predicted_price:.2f}")
    
    # Fetch the latest price
    latest_data = yf.Ticker(ticker).history(period="1d")
    if not latest_data.empty:
        latest_price = latest_data['Close'].iloc[-1]
        print(f"Latest available price: ${latest_price:.2f}")
    else:
        print("Could not fetch the latest price.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network model to predict stock prices.")
    parser.add_argument("--ticker", type=str, default='QQQ', help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default='2010-01-01', help="Start date for data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date for data (YYYY-MM-DD). Defaults to yesterday's date if not provided.")
    parser.add_argument("--lookback_days", type=int, default=20, help="Number of past days to use for prediction")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 64, 32], help="Sizes of hidden layers")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter tuning")
    
    args = parser.parse_args()
    
    main(args.ticker, args.start_date, args.end_date, args.lookback_days, args.hidden_sizes, args.n_trials)

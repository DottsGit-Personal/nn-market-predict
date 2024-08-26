import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
from data_fetcher import get_data_for_model
from model_tuning import tune_hyperparameters
from data_loader import load_data
from model import create_model
from train import train_model

def backtest(model, data, scaler_X, scaler_y, start_date, initial_amount, feature_names):
    device = next(model.parameters()).device  # Get the device of the model
    model.eval()  # Set the model to evaluation mode
    
    portfolio = initial_amount
    shares = 0
    transactions = []
    
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    data.index = data.index.tz_convert('UTC')
    
    print("Starting backtest...")
    print(f"Initial portfolio value: ${portfolio:.2f}")
    
    with torch.no_grad():
        for i in range(len(data)):
            date = data.index[i]
            if date < start_date:
                continue
            
            # Get current price
            current_price = data.iloc[i]['Close']
            
            # Prepare input data for the model
            input_data = data.iloc[i][feature_names].to_frame().T
            input_data_scaled = scaler_X.transform(input_data)
            input_tensor = torch.FloatTensor(input_data_scaled).to(device)
            
            # Get model prediction
            prediction_scaled = model(input_tensor).cpu().item()
            prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
            
            print(f"Date: {date}, Current Price: ${current_price:.2f}, Raw Prediction: {prediction_scaled:.6f}, Scaled Prediction: {prediction:.6f}")
            
            # Simulate buying
            if prediction > 0.0005 and portfolio > current_price:  # Lowered threshold to 0.05%
                shares_to_buy = int(portfolio * 0.25 // current_price)  # Use 25% of available cash
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    shares += shares_to_buy
                    portfolio -= cost
                    transactions.append((date, 'BUY', shares_to_buy, current_price, cost, portfolio, shares * current_price + portfolio))
                    print(f"Bought {shares_to_buy} shares at ${current_price:.2f}")
            
            # Simulate selling
            elif prediction < -0.0005 and shares > 0:  # Lowered threshold to -0.05%
                shares_to_sell = shares // 2  # Sell half of the shares
                if shares_to_sell > 0:
                    revenue = shares_to_sell * current_price
                    portfolio += revenue
                    shares -= shares_to_sell
                    transactions.append((date, 'SELL', shares_to_sell, current_price, revenue, portfolio, shares * current_price + portfolio))
                    print(f"Sold {shares_to_sell} shares at ${current_price:.2f}")
    
    # Sell any remaining shares at the end
    if shares > 0:
        final_price = data.iloc[-1]['Close']
        revenue = shares * final_price
        portfolio += revenue
        transactions.append((data.index[-1], 'SELL', shares, final_price, revenue, portfolio, portfolio))
        print(f"Final sell: {shares} shares at ${final_price:.2f}")
    
    if not transactions:
        initial_date = data.index[data.index >= start_date][0]
        final_date = data.index[-1]
        transactions.append((initial_date, 'INITIAL', 0, data.loc[initial_date, 'Close'], 0, initial_amount, initial_amount))
        transactions.append((final_date, 'FINAL', 0, data.loc[final_date, 'Close'], 0, portfolio, portfolio))
    
    print(f"Final portfolio value: ${portfolio:.2f}")
    
    return pd.DataFrame(transactions, columns=['Date', 'Action', 'Shares', 'Price', 'Amount', 'Cash', 'Total Value'])

def main(ticker, start_date, end_date, initial_amount, lookback_days=20, hidden_sizes=[128, 64, 32], n_trials=100):
    # Fetch data from the earliest available date to the end_date
    earliest_date = '1990-01-01'  # Adjust this if needed
    data, current_price = get_data_for_model(ticker, earliest_date, end_date, lookback_days)
    
    if data is None or data.empty:
        print(f"No data available for {ticker} in the specified date range.")
        return

    # Split the data into training (up to start_date) and backtesting periods
    train_data = data[data.index < start_date]
    backtest_data = data[data.index >= start_date]

    # Prepare data for the model
    X = train_data.drop('Target', axis=1)
    y = train_data['Target']
    
    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # After scaling the data
    print("Target variable statistics:")
    print(f"Original y range: [{y.min()}, {y.max()}]")
    print(f"Scaled y range: [{y_scaled.min()}, {y_scaled.max()}]")
    
    # Tune hyperparameters
    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(X_scaled, y_scaled, n_trials=n_trials)
    print("Best hyperparameters:", best_params)

    # Extract hidden sizes from best_params
    n_layers = best_params['n_layers']
    hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]

    # Create and train the model with best parameters
    input_size = X_scaled.shape[1]
    output_size = 1
    model = create_model(input_size, hidden_sizes, output_size, best_params['dropout_rate'])
    trained_model = train_model(model, X_scaled, y_scaled, num_epochs=300, 
                                batch_size=best_params['batch_size'], 
                                learning_rate=best_params['learning_rate'], 
                                device=device)

    # Ensure the model is on the same device as in training
    trained_model = trained_model.to(device)

    # After training the model
    print("\nModel output diagnostics:")
    with torch.no_grad():
        test_output = trained_model((torch.FloatTensor(X_scaled).to(device)).cpu().numpy())
    print(f"Raw model output range: [{test_output.min()}, {test_output.max()}]")
    print(f"Scaled model output range: [{scaler_y.inverse_transform(test_output).min()}, {scaler_y.inverse_transform(test_output).max()}]")
    
    feature_names = X.columns.tolist()
    backtest_results = backtest(trained_model, backtest_data, scaler_X, scaler_y, start_date, initial_amount, feature_names)

    # Print results
    print(backtest_results)
    
    if not backtest_results.empty:
        initial_value = backtest_results.iloc[0]['Total Value']
        final_value = backtest_results.iloc[-1]['Total Value']
        total_return = (final_value - initial_value) / initial_value * 100

        print(f"\nInitial investment: ${initial_amount:.2f}")
        print(f"Final value: ${final_value:.2f}")
        print(f"Total return: {total_return:.2f}%")
    else:
        print("No transactions were made during the backtest period.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the stock prediction model.")
    parser.add_argument("--ticker", type=str, default='QQQ', help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date for backtesting (YYYY-MM-DD). Defaults to yesterday's date if not provided.")
    parser.add_argument("--initial_amount", type=float, required=True, help="Initial investment amount")
    parser.add_argument("--lookback_days", type=int, default=20, help="Number of past days to use for prediction")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 64, 32], help="Sizes of hidden layers")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter tuning")

    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    main(args.ticker, args.start_date, args.end_date, args.initial_amount, args.lookback_days, args.hidden_sizes, args.n_trials)
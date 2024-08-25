import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def prepare_data_for_prediction(data, lookback_days=20):
    df = data[['Close', 'Volume']].copy()
    
    # Calculate percentage change for closing price
    df['Close_pct_change'] = df['Close'].pct_change()
    
    # Create features from past days
    for i in range(1, lookback_days + 1):
        df[f'Close_pct_change_{i}d_ago'] = df['Close_pct_change'].shift(i)
        df[f'Volume_{i}d_ago'] = df['Volume'].shift(i)
    
    # Create target variable (next day's percentage change)
    df['Target'] = df['Close_pct_change'].shift(-1)
    
    # Add technical indicators
    df['MA5'] = df['Close'].rolling(window=5).mean().pct_change()
    df['MA20'] = df['Close'].rolling(window=20).mean().pct_change()
    
    # Relative Strength Index (RSI)
    delta = df['Close_pct_change']
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close_pct_change'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close_pct_change'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close_pct_change'].rolling(window=20).std()
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def get_data_for_model(ticker, start_date, end_date, lookback_days=20):
    raw_data = fetch_stock_data(ticker, start_date, end_date)
    prepared_data = prepare_data_for_prediction(raw_data, lookback_days)
    return prepared_data, raw_data['Close'].iloc[-1]

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def prepare_data_for_prediction(data, lookback_days=20):
    df = data[['Close', 'Volume', 'Open', 'High', 'Low']].copy()
    
    # Price-based features
    df['Close_pct_change'] = df['Close'].pct_change()
    df['High_Low_diff'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_diff'] = (df['Open'] - df['Close']) / df['Close']
    
    # Volume features
    df['Volume_pct_change'] = df['Volume'].pct_change()
    
    # Moving averages
    sma_indicator = SMAIndicator(close=df['Close'], window=20)
    ema_indicator = EMAIndicator(close=df['Close'], window=20)
    df['SMA20'] = sma_indicator.sma_indicator()
    df['EMA20'] = ema_indicator.ema_indicator()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # RSI
    rsi_indicator = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi_indicator.rsi()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['Close'])
    df['Bollinger_hband'] = bollinger.bollinger_hband()
    df['Bollinger_lband'] = bollinger.bollinger_lband()
    df['Bollinger_mavg'] = bollinger.bollinger_mavg()
    
    # Lagged features
    for i in range(1, lookback_days + 1):
        df[f'Close_pct_change_{i}d_ago'] = df['Close_pct_change'].shift(i)
        df[f'Volume_pct_change_{i}d_ago'] = df['Volume_pct_change'].shift(i)
    
    # Target variable (next day's percentage change)
    df['Target'] = df['Close_pct_change'].shift(-1)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def get_data_for_model(ticker, start_date, end_date, lookback_days=20):
    raw_data = fetch_stock_data(ticker, start_date, end_date)
    prepared_data = prepare_data_for_prediction(raw_data, lookback_days)
    return prepared_data, raw_data['Close'].iloc[-1]

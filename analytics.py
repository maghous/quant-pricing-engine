import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
import requests
import scipy.stats as si

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_session():
    """Create a browser-like session to bypass blockings."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def generate_mock_data(ticker, start, end):
    """Realistic fallback if both API and Cache fail."""
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    price = 150 + np.cumsum(np.random.randn(n) * 2) 
    df = pd.DataFrame({
        'Open': price * 0.99, 'High': price * 1.01,
        'Low': price * 0.98, 'Close': price,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    df.is_mock = True
    df.is_cached = False
    return df

def get_stock_data(ticker_symbol, start_date, end_date):
    """
    Intelligent data fetcher: Cache -> API -> Mock.
    Stores results in CSV to avoid future API calls.
    """
    csv_path = os.path.join(DATA_DIR, f"{ticker_symbol}_history.csv")
    
    # 1. Try to load from Cache first
    if os.path.exists(csv_path):
        try:
            cached_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            # Filter for requested dates
            mask = (cached_df.index >= pd.Timestamp(start_date)) & (cached_df.index <= pd.Timestamp(end_date))
            result_df = cached_df.loc[mask]
            
            # If we have enough data (at least 30 days of what we need)
            if len(result_df) > 30:
                result_df.is_mock = False
                result_df.is_cached = True
                return result_df
        except Exception:
            pass # Fail silently and try API

    # 2. Try to fetch from API
    try:
        session = get_session()
        ticker = yf.Ticker(ticker_symbol, session=session)
        
        # Fetch a bit more to ensure SMA calculation is possible
        api_start = pd.Timestamp(start_date) - timedelta(days=200)
        
        # Ensure dates are strings in YYYY-MM-DD format to bypass timezone parsing errors
        start_str = api_start.strftime('%Y-%m-%d')
        end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
        
        df = ticker.history(start=start_str, end=end_str, interval="1d", auto_adjust=True)
        
        if not df.empty:
            # Save to Cache for next time
            df.to_csv(csv_path)
            
            # Filter to requested range for return
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            result_df = df.loc[mask]
            result_df.is_mock = False
            result_df.is_cached = False
            return result_df
    except Exception:
        pass

    # 3. Last resort: Mock data
    return generate_mock_data(ticker_symbol, start_date, end_date)

def pre_fetch_major_stocks():
    """Function to download top stocks in advance."""
    majors = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    end = datetime.now()
    start = end - timedelta(days=365*3) # 3 years
    results = []
    for ticker in majors:
        get_stock_data(ticker, start, end)
        results.append(ticker)
    return results

def calculate_rsi(df, window=14):
    if df.empty or len(df) < window: return df
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_moving_averages(df, short_window=20, long_window=50):
    if df.empty: return df
    df[f'SMA_{short_window}'] = df['Close'].rolling(window=short_window).mean()
    df[f'SMA_{long_window}'] = df['Close'].rolling(window=long_window).mean()
    return df

def calculate_bollinger_bands(df, window=20):
    if df.empty or len(df) < window: return df
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['Upper_Band'] = sma + (std * 2)
    df['Lower_Band'] = sma - (std * 2)
    return df

def basic_prediction(df, days_to_predict=30):
    if df.empty or len(df) < 10: return pd.DataFrame()
    temp_df = df.reset_index()
    if 'Date' not in temp_df.columns:
        temp_df = temp_df.rename(columns={temp_df.columns[0]: 'Date'})
    x = np.arange(len(temp_df)).reshape(-1, 1)
    y = temp_df['Close'].values.reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x, y)
    future_x = np.arange(len(temp_df), len(temp_df) + days_to_predict).reshape(-1, 1)
    future_preds = model.predict(future_x)
    last_date = temp_df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_preds.flatten()})

# --- Quantitative Finance Methods ---

def calculate_var(df, confidence_level=0.95):
    """Calculates Historical Value at Risk (VaR) on daily returns."""
    if df.empty or len(df) < 2: return None
    returns = df['Close'].pct_change().dropna()
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def monte_carlo_simulation(df, days_to_predict=30, simulations=100):
    """Generates future price paths using Geometric Brownian Motion (GBM)."""
    if df.empty or len(df) < 2: return pd.DataFrame()
    
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    if 'Date' in df.columns:
        last_date = df['Date'].iloc[-1]
        
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    
    # Pre-allocate array for simulations
    sim_paths = np.zeros((days_to_predict, simulations))
    
    for i in range(simulations):
        # Calculate daily shock
        shocks = np.random.normal(loc=mu, scale=sigma, size=days_to_predict)
        # Calculate path
        price_path = last_price * np.exp(np.cumsum(shocks))
        sim_paths[:, i] = price_path
        
    result_df = pd.DataFrame(sim_paths, index=future_dates, columns=[f'Sim_{i}' for i in range(simulations)])
    return result_df

def black_scholes_call(S, K, T, r, sigma):
    """
    Theoretical pricing of an European Call Option.
    S: Spot Price
    K: Strike Price
    T: Time to maturity (in years)
    r: Risk-free rate (annual)
    sigma: Volatility (annualized)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

"""
data/market_client.py
Wrappers for yfinance with Streamlit caching for optimized data ingestion.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from typing import List, Dict
from core.state import AssetPosition

@st.cache_data(ttl=3600)
def fetch_historical_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    r"""
    Fetches historical adjusted closing prices for a list of tickers.
    
    Args:
        tickers (List[str]): List of ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): Time period string compatible with yfinance. Defaults to "1y".
        
    Returns:
        pd.DataFrame: A DataFrame where columns correspond to ticker symbols, 
                      the index is the Date, and the values are the Adjusted Close prices.
    """
    if not tickers:
        return pd.DataFrame()
        
    # Removed group_by='ticker' for a more predictable multi-index structure
    data = yf.download(tickers, period=period, interval="1d", progress=False)
    
    # yfinance sometimes changes between 'Adj Close' and 'Close' depending on the version
    # and auto_adjust settings. We check dynamically to prevent KeyErrors.
    if 'Adj Close' in data:
        closes = data['Adj Close']
    elif 'Close' in data:
        closes = data['Close']
    else:
        raise KeyError(f"Could not locate price columns. Available columns: {data.columns}")
    
    # If only one ticker is requested, yfinance returns a Series. 
    # We must cast it to a DataFrame to maintain the expected contract.
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers)
        
    return closes.dropna()


@st.cache_data(ttl=60)
def get_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetches the most recent market price for a list of assets to ensure the UI 
    reflects near real-time valuations.
    
    Utilizes yfinance's `fast_info` property to bypass heavy metadata scraping 
    and minimize API latency.
    
    Args:
        tickers (List[str]): List of ticker symbols to query.
        
    Returns:
        Dict[str, float]: A dictionary mapping each ticker symbol to its latest 
                          market price (e.g., {'AAPL': 175.50, 'MSFT': 420.25}).
    """
    prices = {}
    for ticker in tickers:
        t = yf.Ticker(ticker)
        prices[ticker] = t.fast_info.last_price
    return prices


def hydrate_portfolio(ticker_shares: Dict[str, int]) -> List[AssetPosition]:
    """
    Transforms a raw mapping of ticker symbols and share counts into a list 
    of strictly validated domain models.
    
    This acts as the ingestion boundary between raw user input (or database storage) 
    and the application's internal state management, fetching current prices automatically
    to calculate the total value of each position.
    
    Args:
        ticker_shares (Dict[str, int]): A dictionary mapping ticker symbols to the 
                                        current integer number of shares held 
                                        (e.g., {'AAPL': 50, 'MSFT': 100}).
                                        
    Returns:
        List[AssetPosition]: A list of populated and Pydantic-validated AssetPosition 
                             models ready for injection into the PortfolioState.
    """
    tickers = list(ticker_shares.keys())
    prices = get_latest_prices(tickers)
    
    return [
        AssetPosition(
            symbol=ticker, 
            shares=shares, 
            price=prices.get(ticker, 0.0)
        ) 
        for ticker, shares in ticker_shares.items()
    ]
r"""
math_engine/moments.py
Financial mathematics engine for computing portfolio moments and culling asset universes.

This module is responsible for converting raw historical price matrices into the 
Expected Returns (\mu) and Covariance (\Sigma) matrices required for mean-variance optimization.
It includes critical dimensionality reduction logic to protect the local exact eigensolver 
from exponential memory scaling.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def cull_asset_universe(
    prices: pd.DataFrame, 
    max_assets: int, 
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> Tuple[List[str], pd.DataFrame]:
    r"""
    Reduces the dimensionality of the asset universe by selecting the top N assets 
    based on their annualized Sharpe ratio.
    
    This is a required safety valve for the exact eigensolver. By capping `max_assets`, 
    we ensure the downstream QUBO formulation does not exceed available RAM when 
    constructing the $2^N$ matrix.
    
    Args:
        prices (pd.DataFrame): Historical adjusted closing prices. Columns are tickers.
        max_assets (int): The maximum number of assets to retain.
        risk_free_rate (float): The baseline risk-free rate for Sharpe calculation. Defaults to 0.0.
        trading_days (int): Number of trading days in a year for annualization. Defaults to 252.
        
    Returns:
        Tuple[List[str], pd.DataFrame]: 
            - A list of the retained ticker symbols.
            - A filtered DataFrame containing only the price history of the retained assets.
    """
    if prices.empty or len(prices.columns) <= max_assets:
        return list(prices.columns), prices

    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Annualize metrics
    annualized_returns = daily_returns.mean() * trading_days
    annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
    
    # Calculate Sharpe Ratio, safely handling zero volatility
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe_ratios = (annualized_returns - risk_free_rate) / annualized_volatility
        sharpe_ratios = sharpe_ratios.fillna(0)  # Handle any division by zero
        
    # Sort tickers by Sharpe ratio descending and slice top N
    top_tickers = sharpe_ratios.sort_values(ascending=False).head(max_assets).index.tolist()
    
    return top_tickers, prices[top_tickers]


def compute_moments(
    prices: pd.DataFrame, 
    trading_days: int = 252
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Computes the annualized Expected Returns (\mu) vector and Covariance (\Sigma) matrix.
    
    Args:
        prices (pd.DataFrame): Historical adjusted closing prices. Must be pre-filtered 
                               if bounding the eigensolver matrix size.
        trading_days (int): Number of trading days in a year for annualization. Defaults to 252.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - Expected returns vector (\mu) as a 1D NumPy array.
            - Covariance matrix (\Sigma) as a 2D NumPy array.
            - Ordered list of ticker symbols corresponding to the array indices.
    """
    if prices.empty:
        return np.array([]), np.array([[]]), []

    daily_returns = prices.pct_change().dropna()
    
    # Calculate annualized expected returns (\mu)
    mu_series = daily_returns.mean() * trading_days
    mu_array = mu_series.to_numpy()
    
    # Calculate annualized covariance matrix (\Sigma)
    sigma_df = daily_returns.cov() * trading_days
    sigma_array = sigma_df.to_numpy()
    
    # Extract ordering to maintain parity with the matrices
    ordered_tickers = list(prices.columns)
    
    return mu_array, sigma_array, ordered_tickers
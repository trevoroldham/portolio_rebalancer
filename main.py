"""
main.py
Headless execution script to validate the backend portfolio optimization pipeline.
"""

import warnings
import pandas as pd
from core.state import PortfolioState
from data.market_client import hydrate_portfolio, fetch_historical_data
from math_engine.moments import cull_asset_universe, compute_moments
from math_engine.qubo_mapper import construct_portfolio_qubo
from math_engine.exact_solver import compute_optimal_allocation
from logic.rebalancer import generate_trade_actions

# Suppress Streamlit caching warnings when running outside of Streamlit
warnings.filterwarnings("ignore", module="streamlit")

def run_pipeline():
    print("\n--- 1. Initializing Current State ---")
    # Mock current holdings and cash
    raw_holdings = {"AAPL": 10, "MSFT": 5}
    cash_balance = 5000.0
    
    print(f"Hydrating holdings: {raw_holdings}...")
    positions = hydrate_portfolio(raw_holdings)
    
    # Construct strictly typed state
    state = PortfolioState(
        cash_balance=cash_balance,
        positions={p.symbol: p for p in positions}
    )
    print(f"Total Portfolio Equity: ${state.total_equity:,.2f}")

    print("\n--- 2. Fetching & Culling Market Data ---")
    # Define a broader watchlist to pull data for
    watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA"]
    
    print(f"Fetching 1-year historical data for {len(watchlist)} assets...")
    historical_prices = fetch_historical_data(watchlist, period="1y")
    
    print("Culling universe by Sharpe Ratio to protect exact solver memory...")
    # STRICT MEMORY LIMIT: max_assets=4 with 3-bit depth = 12 qubits (Safe for local RAM)
    top_tickers, filtered_prices = cull_asset_universe(historical_prices, max_assets=4)
    print(f"Retained Assets: {top_tickers}")

    print("\n--- 3. Computing Mathematical Moments ---")
    mu, sigma, ordered_tickers = compute_moments(filtered_prices)
    print(f"Expected Returns (\u03bc) shape: {mu.shape}")
    print(f"Covariance (\u03a3) shape: {sigma.shape}")

    print("\n--- 4. Formulating QUBO ---")
    # Get latest prices for the retained assets to set the exact budget constraint
    # We pull this from the last row of our historical data for consistency
    current_prices = filtered_prices.iloc[-1].to_numpy()
    
    qubo, qp = construct_portfolio_qubo(
        mu=mu,
        sigma=sigma,
        prices=current_prices,
        tickers=ordered_tickers,
        budget=state.total_equity,
        risk_aversion=0.5,
        bit_depth=3,
        penalty_multiplier=1e6
    )
    print(f"QUBO Variables (Qubits): {qubo.get_num_vars()}")

    print("\n--- 5. Executing Exact NumPy Eigensolver ---")
    print("Computing exact lowest eigenvalue... (This may take a moment)")
    optimal_allocation = compute_optimal_allocation(qubo, ordered_tickers)
    
    print("\nOptimal Integer Allocation:")
    for ticker, shares in optimal_allocation.items():
        print(f"  {ticker}: {shares} shares")

    print("\n--- 6. Running Threshold Rebalancer ---")
    trades = generate_trade_actions(
        current_state=state,
        optimal_allocation=optimal_allocation,
        drift_threshold_pct=5.0
    )

    if not trades:
        print("No trades recommended. Portfolio is within optimal drift threshold.")
    else:
        print("Recommended Trades (>5% drift):")
        for trade in trades:
            print(f"  [{trade.action}] {trade.symbol}: {trade.current_shares} -> {trade.target_shares} "
                  f"(Drift: {trade.drift_percentage:.1f}%, Est. Value: ${abs(trade.trade_value):,.2f})")

if __name__ == "__main__":
    run_pipeline()
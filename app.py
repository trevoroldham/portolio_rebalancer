"""
app.py
Main Streamlit entry point orchestrating the Portfolio Rebalancer pipeline.
"""

import streamlit as st
import pandas as pd
import time

# Import domain models and business logic
from core.state import PortfolioState
from data.market_client import hydrate_portfolio, fetch_historical_data
from math_engine.moments import cull_asset_universe, compute_moments
from math_engine.qubo_mapper import construct_portfolio_qubo
from math_engine.exact_solver import compute_optimal_allocation
from logic.rebalancer import generate_trade_actions
from ui.widgets import render_portfolio_input, render_optimization_params

# Configure the Streamlit page
st.set_page_config(
    page_title="Quantum-Inspired Portfolio Rebalancer",
    page_icon="⚛️",
    layout="wide"
)

def main():
    st.title("⚛️ Discrete Portfolio Rebalancer")
    st.markdown("""
    This engine maps Markowitz mean-variance optimization to a Quadratic Unconstrained Binary Optimization (QUBO) problem. 
    It utilizes Qiskit's exact classical eigensolver to output discrete, strictly integer share allocations.
    """)
    st.divider()

    # --- 1. UI Input Section ---
    col_input, col_params = st.columns(2, gap="large")
    
    with col_input:
        raw_holdings = render_portfolio_input()
        
    with col_params:
        cash_balance, risk_aversion, bit_depth, drift_threshold, lookback_period = render_optimization_params()

    st.divider()

    # --- 2. Execution Gate ---
    # We only run the heavy pipeline when the user explicitly clicks the button.
    if st.button("🚀 Run Exact Optimization", type="primary", use_container_width=True):
        
        if not raw_holdings:
            st.error("Please enter at least one holding to begin.")
            return

        with st.spinner("Executing optimization pipeline..."):
            start_time = time.time()
            
            # Step A: Hydrate State
            positions = hydrate_portfolio(raw_holdings)
            current_state = PortfolioState(
                cash_balance=cash_balance,
                positions={p.symbol: p for p in positions}
            )
            
            # Step B: Fetch Data
            tickers = list(raw_holdings.keys())
            prices_df = fetch_historical_data(tickers, period=lookback_period)
            
            # Step C: Safety Culling (OOM Protection)
            # We enforce a strict maximum of 14 qubits total to prevent the Streamlit app from freezing.
            # Max Assets = Floor(14 / bit_depth). If bit_depth is 3, max_assets is 4.
            safe_max_assets = max(1, 14 // bit_depth)
            
            if len(tickers) > safe_max_assets:
                st.warning(f"⚠️ Dimensionality limit reached for exact solving. Culling universe from {len(tickers)} to top {safe_max_assets} assets by Sharpe Ratio.")
                
            top_tickers, filtered_prices = cull_asset_universe(prices_df, max_assets=safe_max_assets)
            
            # Step D: Mathematical Moments
            mu, sigma, ordered_tickers = compute_moments(filtered_prices)
            current_prices = filtered_prices.iloc[-1].to_numpy()
            
            # Step E: QUBO Mapping
            qubo, _ = construct_portfolio_qubo(
                mu=mu,
                sigma=sigma,
                prices=current_prices,
                tickers=ordered_tickers,
                budget=current_state.total_equity,
                risk_aversion=risk_aversion,
                bit_depth=bit_depth,
                penalty_multiplier=1e5
            )
            
            # Step F: Execution
            optimal_allocation = compute_optimal_allocation(
                qubo=qubo, 
                tickers=ordered_tickers,
                prices=current_prices,
                budget=current_state.total_equity,
                bit_depth=bit_depth
            )
            
            # Step G: Rebalancing Logic
            trades = generate_trade_actions(
                current_state=current_state,
                optimal_allocation=optimal_allocation,
                drift_threshold_pct=drift_threshold
            )
            
            # Store results in session state for rendering
            st.session_state.optimization_results = {
                "current_state": current_state,
                "optimal_allocation": optimal_allocation,
                "trades": trades,
                "solve_time": time.time() - start_time,
                "qubits": qubo.get_num_vars()
            }

    # --- 3. Results Dashboard ---
    if "optimization_results" in st.session_state:
        results = st.session_state.optimization_results
        state = results["current_state"]
        trades = results["trades"]
        
        st.subheader("📊 Rebalancing Actions")
        st.caption(f"Solved exactly across {results['qubits']} qubits in {results['solve_time']:.2f} seconds.")
        
        # Macro Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Equity", f"${state.total_equity:,.2f}")
        m2.metric("Uninvested Cash", f"${state.cash_balance:,.2f}")
        m3.metric("Trade Actions Required", len(trades))
        
        if not trades:
            st.success(f"No trades required. All assets are within the {drift_threshold}% drift threshold.")
        else:
            # Format trades for display
            trade_data = []
            for t in trades:
                trade_data.append({
                    "Action": t.action,
                    "Asset": t.symbol,
                    "Current Shares": t.current_shares,
                    "Target Shares": t.target_shares,
                    "Delta": f"{'+' if t.share_delta > 0 else ''}{t.share_delta}",
                    "Est. Value ($)": round(t.trade_value, 2),
                    "Drift (%)": round(t.drift_percentage, 2)
                })
                
            st.dataframe(
                pd.DataFrame(trade_data),
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()
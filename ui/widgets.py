"""
ui/widgets.py
Granular, reusable UI components for the portfolio rebalancer.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Tuple

def render_portfolio_input() -> Dict[str, int]:
    """
    Renders an interactive data grid for the user to input their current holdings.
    Uses Streamlit's data_editor to allow dynamic row additions and deletions.
    
    Returns:
        Dict[str, int]: A cleaned dictionary mapping tickers to integer share counts.
    """
    st.subheader("💼 Current Holdings")
    
    # Initialize a default dataframe in session state if it doesn't exist
    if "raw_holdings_df" not in st.session_state:
        st.session_state.raw_holdings_df = pd.DataFrame([
            {"Ticker": "AAPL", "Shares": 10},
            {"Ticker": "MSFT", "Shares": 5}
        ])

    # Render the interactive editor
    edited_df = st.data_editor(
        st.session_state.raw_holdings_df,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker Symbol", 
                help="Enter standard stock ticker (e.g., AAPL)", 
                max_chars=6, 
                required=True
            ),
            "Shares": st.column_config.NumberColumn(
                "Shares Held", 
                help="Must be a whole number", 
                min_value=0, 
                step=1, 
                required=True
            )
        },
        use_container_width=True,
        hide_index=True
    )

    # Clean and parse the dataframe back into a strictly typed dictionary
    # We drop any rows where the user left the Ticker blank
    cleaned_dict = {}
    for _, row in edited_df.dropna(subset=['Ticker']).iterrows():
        ticker = str(row['Ticker']).strip().upper()
        if ticker:
            cleaned_dict[ticker] = int(row['Shares'])
            
    return cleaned_dict


def render_optimization_params() -> Tuple[float, float, int, float, str]:
    """
    Renders the control panel for mathematical and operational constraints.
    
    Returns:
        Tuple containing:
            - cash_balance (float)
            - risk_aversion (float)
            - bit_depth (int)
            - drift_threshold (float)
            - lookback_period (str)
    """
    st.subheader("⚙️ Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cash_balance = st.number_input(
            "Available Cash ($)", 
            min_value=0.0, 
            value=5000.0, 
            step=100.0,
            help="Uninvested capital available for allocation."
        )
        
        risk_aversion = st.slider(
            "Risk Aversion (q)", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.5, 
            step=0.1,
            help="Higher values penalize portfolio variance more aggressively."
        )
        
        # NEW: Timeframe selector
        lookback_period = st.selectbox(
            "Data Lookback Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,  # Defaults to "1y"
            help="The historical window used to calculate expected returns and covariance."
        )
        
    with col2:
        bit_depth = st.slider(
            "Bit-Depth per Asset", 
            min_value=1, 
            max_value=5, 
            value=3, 
            step=1,
            help="⚠️ WARNING: Scales eigensolver memory exponentially at O(2^N). 3 bits = max 7 shares per tranche. 4 bits = max 15."
        )
        
        drift_threshold = st.number_input(
            "Drift Threshold (%)", 
            min_value=0.0, 
            value=5.0, 
            step=0.5,
            help="Minimum allocation drift required to trigger a trade recommendation."
        )
        
    return cash_balance, risk_aversion, bit_depth, drift_threshold, lookback_period
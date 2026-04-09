"""
ui/dashboards.py
Visualization components for the portfolio rebalancer.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict
from core.state import PortfolioState

def render_weight_comparison(
    current_state: PortfolioState, 
    optimal_allocation: Dict[str, int], 
    prices: Dict[str, float]
):
    """
    Renders side-by-side donut charts comparing current vs. target portfolio weights.
    """
    st.subheader("⚖️ Portfolio Composition Shift")
    
    # --- Data Prep: Current State ---
    current_data = []
    for symbol, position in current_state.positions.items():
        if position.shares > 0:
            # Manually calculate value to avoid Pydantic strictness errors
            position_value = position.shares * position.price
            current_data.append({"Asset": symbol, "Value": position_value})
            
    if current_state.cash_balance > 0:
        current_data.append({"Asset": "CASH", "Value": current_state.cash_balance})
        
    # THIS is the line that was missing!
    df_current = pd.DataFrame(current_data)
    
    # --- Data Prep: Target State ---
    target_data = []
    total_invested = 0.0
    for symbol, shares in optimal_allocation.items():
        if shares > 0:
            val = shares * prices.get(symbol, 0.0)
            target_data.append({"Asset": symbol, "Value": val})
            total_invested += val
            
    leftover_cash = current_state.total_equity - total_invested
    if leftover_cash > 0:
        target_data.append({"Asset": "CASH", "Value": leftover_cash})
        
    df_target = pd.DataFrame(target_data)

    # --- Chart Rendering ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Allocation**")
        if not df_current.empty:
            fig_curr = px.pie(
                df_current, values='Value', names='Asset', hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_curr.update_traces(textposition='inside', textinfo='percent+label')
            fig_curr.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_curr, use_container_width=True)

    with col2:
        st.markdown("**Optimal Eigensolver Allocation**")
        if not df_target.empty:
            fig_targ = px.pie(
                df_target, values='Value', names='Asset', hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_targ.update_traces(textposition='inside', textinfo='percent+label')
            fig_targ.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_targ, use_container_width=True)
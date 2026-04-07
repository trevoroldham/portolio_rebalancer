r"""
math_engine/qubo_mapper.py
Translates continuous financial moments and capital constraints into a discrete QUBO using capital buckets.
"""

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import IntegerToBinary, LinearInequalityToPenalty
import numpy as np
from typing import List, Tuple

def construct_portfolio_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    prices: np.ndarray,
    tickers: List[str],
    budget: float,
    risk_aversion: float = 0.5,
    bit_depth: int = 3,
    penalty_multiplier: float = 1e5
) -> Tuple[QuadraticProgram, QuadraticProgram]:
    
    qp = QuadraticProgram(name="Discrete_Markowitz")
    max_units = (2 ** bit_depth) - 1
    
    # 1. Variable Definition (x_i = number of capital buckets)
    for ticker in tickers:
        qp.integer_var(name=ticker, lowerbound=0, upperbound=max_units)
        
    # 2. Objective Function Construction
    # Weight w_i = x_i / max_units
    linear_terms = {
        tickers[i]: -(mu[i] / max_units) for i in range(len(tickers))
    }
    
    quadratic_terms = {}
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            coef = risk_aversion * sigma[i, j] / (max_units ** 2)
            quadratic_terms[(tickers[i], tickers[j])] = coef
            
    qp.minimize(linear=linear_terms, quadratic=quadratic_terms)
    
    # 3. Capital Budget Constraint
    # The sum of all assigned buckets must be <= max_units.
    # Note: If you want to force the portfolio to be 100% invested (no cash), 
    # change sense="<=" to sense="=="
    bucket_dict = {ticker: 1 for ticker in tickers}
    qp.linear_constraint(
        linear=bucket_dict,
        sense="<=", 
        rhs=max_units,
        name="capital_budget"
    )
    
    # 4. Conversion Pipeline
    ineq2pen = LinearInequalityToPenalty(penalty=penalty_multiplier)
    qp_unconstrained = ineq2pen.convert(qp)
    
    int2bin = IntegerToBinary()
    qubo = int2bin.convert(qp_unconstrained)
    
    return qubo, qp
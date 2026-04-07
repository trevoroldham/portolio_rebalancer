r"""
math_engine/qubo_mapper.py
Translates continuous financial moments and capital constraints into a discrete QUBO.
Applies penalty multipliers (\lambda) for budget constraints.
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
    r"""
    Constructs the Discrete Markowitz problem and converts it to a QUBO.
    """
    qp = QuadraticProgram(name="Discrete_Markowitz")
    
    max_units = (2 ** bit_depth) - 1
    
    # 1. Variable Definition
    for ticker in tickers:
        qp.integer_var(name=ticker, lowerbound=0, upperbound=max_units)
        
    # 2. Objective Function Construction
    linear_terms = {
        tickers[i]: -(mu[i] * prices[i] / budget) for i in range(len(tickers))
    }
    
    quadratic_terms = {}
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            coef = risk_aversion * (prices[i] * prices[j] / (budget ** 2)) * sigma[i, j]
            quadratic_terms[(tickers[i], tickers[j])] = coef
            
    qp.minimize(linear=linear_terms, quadratic=quadratic_terms)
    
    # 3. Capital Budget Constraint (DYNAMIC SCALING FIX)
    # We dynamically scale the budget down to a maximum of ~1000 units.
    # This restricts the Qiskit slack variable to ~10 qubits, preventing OOM errors 
    # on the classical exact eigensolver while maintaining constraint fidelity.
    scale_factor = min(1.0, 1000.0 / budget) if budget > 0 else 1.0
    
    integer_budget = int(np.round(budget * scale_factor))
    
    # Ensure prices scale down proportionally, but never drop to 0 (which would make them "free")
    integer_prices = {
        tickers[i]: max(1, int(np.round(prices[i] * scale_factor))) for i in range(len(tickers))
    }
    
    qp.linear_constraint(
        linear=integer_prices,
        sense="<=",
        rhs=integer_budget,
        name="capital_budget"
    )
    
    # 4. Conversion Pipeline
    ineq2pen = LinearInequalityToPenalty(penalty=penalty_multiplier)
    qp_unconstrained = ineq2pen.convert(qp)
    
    int2bin = IntegerToBinary()
    qubo = int2bin.convert(qp_unconstrained)
    
    return qubo, qp
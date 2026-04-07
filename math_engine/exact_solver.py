"""
math_engine/exact_solver.py
Execution wrapper for NumPyMinimumEigensolver.
"""

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
import numpy as np
from typing import Dict, List

def compute_optimal_allocation(
    qubo: QuadraticProgram, 
    tickers: List[str],
    prices: np.ndarray,
    budget: float,
    bit_depth: int
) -> Dict[str, int]:
    
    exact_mes = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(exact_mes)
    result = optimizer.solve(qubo)
    
    # Step 1: Decode binary string into Bucket counts
    optimal_buckets = {ticker: 0 for ticker in tickers}
    for var_name, binary_value in result.variables_dict.items():
        if binary_value > 0.5:
            try:
                ticker, bit_index = var_name.split('@')
                if ticker in optimal_buckets:
                    optimal_buckets[ticker] += 2 ** int(bit_index)
            except ValueError:
                pass
                
    # Step 2: Translate Buckets to Physical Shares
    max_units = (2 ** bit_depth) - 1
    capital_per_bucket = budget / max_units if max_units > 0 else 0
    
    optimal_shares = {}
    for i, ticker in enumerate(tickers):
        allocated_capital = optimal_buckets[ticker] * capital_per_bucket
        # Floor division to ensure we don't exceed the allocated capital
        optimal_shares[ticker] = int(allocated_capital // prices[i])
                
    return optimal_shares
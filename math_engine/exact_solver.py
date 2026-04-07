"""
math_engine/exact_solver.py
Execution wrapper for NumPyMinimumEigensolver.

Provides the execution environment to compute the exact ground state of the 
portfolio QUBO. By running locally, this bypasses the statistical noise and 
shot-count limitations inherent to physical QPU executions.
"""

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from typing import Dict, List


def compute_optimal_allocation(
    qubo: QuadraticProgram, 
    tickers: List[str]
) -> Dict[str, int]:
    """
    Executes the NumPy exact eigensolver against the formulated QUBO to find the 
    global minimum of the objective function.
    
    This computes the lowest eigenvalue algebraically. The resulting binary bit-string 
    is then deterministically decoded back into integer share allocations based on 
    the bit-depth expansion.
    
    Args:
        qubo (QuadraticProgram): The fully constrained binary objective function.
        tickers (List[str]): The ordered list of asset symbols used in the problem.
        
    Returns:
        Dict[str, int]: A mapping of ticker symbols to their optimal integer 
                        share allocations.
    """
    # Initialize the exact classical eigensolver
    exact_mes = NumPyMinimumEigensolver()
    
    # Wrap the eigensolver in Qiskit's optimization interface
    optimizer = MinimumEigenOptimizer(exact_mes)
    
    # Compute the exact ground state (global minimum)
    result = optimizer.solve(qubo)
    
    # Initialize the allocation dictionary with zero bounds
    optimal_allocation = {ticker: 0 for ticker in tickers}
    
    # Decode the binary bit-string back into integer shares.
    # We parse the Qiskit '@' separator to reconstruct the base-10 integers.
    for var_name, binary_value in result.variables_dict.items():
        if binary_value > 0.5:  # Tolerance threshold for floating point 1.0
            try:
                ticker, bit_index = var_name.split('@')
                if ticker in optimal_allocation:
                    optimal_allocation[ticker] += 2 ** int(bit_index)
            except ValueError:
                # Failsafe: Handles variables not strictly conforming to the bit-expansion
                pass
                
    return optimal_allocation
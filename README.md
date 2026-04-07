# Portfolio Rebalancer

A production-grade classical web application designed to bridge the gap between quantum-inspired optimization and real-world portfolio management. This tool takes the exact QUBO (Quadratic Unconstrained Binary Optimization) mapping and mathematical constraints from a quantum portfolio project and executes them locally using Qiskit's `NumPyMinimumEigensolver`.

---

## Project Overview

The objective of this project is to provide a deterministic, high-fidelity rebalancing engine. By moving away from noisy QPU hardware (like the IBM Heron r2) to an exact classical eigensolver, we isolate the mathematical behavior of discrete asset allocation while maintaining the rigorous constraint mapping required for quantum execution.

### The Pipeline
1.  **Ingestion:** Pulls real-time and historical market data via `yfinance`.
2.  **Culling:** Automatically reduces the asset universe based on the annualized Sharpe Ratio to prevent the $2^N$ exponential memory explosion of the exact solver.
3.  **QUBO Mapping:** Translates the Markowitz mean-variance objective into a discrete **Capital Bucket** problem:
    $$\min_{x} \left( q \cdot \frac{x^T \Sigma x}{K^2} - \frac{\mu^T x}{K} \right)$$
    *Where $x$ is a vector of discrete capital units (buckets), $K$ is the maximum units available ($2^{bit\_depth} - 1$), and $q$ is the risk aversion.*
4.  **Exact Solving:** Algebraic computation of the global minimum using the `NumPyMinimumEigensolver`. The solver assigns $K$ total units across the asset universe.
5.  **Translation & Rebalancing:** Converts the optimal "buckets" back into physical share counts based on current market prices. It then compares this to the `PortfolioState` and generates `TradeRecommendations` for any assets exceeding the drift threshold.

## Tech Stack
* **Frontend:** Streamlit (Reactive State Management)
* **Optimization:** Qiskit Optimization & Qiskit Algorithms
* **Data Science:** NumPy, Pandas, yfinance
* **Data Integrity:** Pydantic V2 (Strict Domain Modeling)

---

## Roadmap

### Phase 1: Local Foundation (Current)
- [x] Architected the Pydantic-based state management.
- [x] Built the `yfinance` ingestion layer with Streamlit caching.
- [x] Implemented dynamic budget scaling to prevent memory OOM during slack variable generation.
- [x] Validated the end-to-end backend pipeline via `main.py`.

### Phase 2: UI & UX
- [x] Implement the Streamlit dashboard for "Current vs. Target" trade suggestions.
- [x] Build interactive parameter tuning (Risk Aversion, Bit-Depth, Drift Threshold).
- [ ] Add performance backtesting visuals for the "Optimal" suggested state.

### Phase 3: Advanced Logic
- [ ] Integrate transaction cost modeling into the QUBO penalty terms.
- [ ] Implement multi-period optimization.
- [ ] Support for varied bit-depths per asset based on liquidity/volatility.

---

## Getting Started

### Prerequisites
* Python 3.12+
* Virtual Environment (Recommended)

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the backend test:
    ```bash
    python3 main.py
    ```
4.  Launch the web app:
    ```bash
    streamlit run app.py
    ```
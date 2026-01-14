"""
Quantum Elasticity Solver - FastAPI Backend
Implements HHL, VQLS, Adiabatic quantum methods and classical solver
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from scipy.linalg import expm, eigh, solve, norm
from scipy.optimize import minimize
import time

app = FastAPI(
    title="Quantum Elasticity Solver",
    description="Solve elasticity problems using quantum and classical methods",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---
class ElasticityProblem(BaseModel):
    """Input problem definition"""
    matrix_size: int = Field(default=4, ge=2, le=10, description="Size of the stiffness matrix")
    force_position: int = Field(default=-1, description="Position of applied force (-1 for last node)")
    force_value: float = Field(default=1.0, description="Magnitude of applied force")
    custom_matrix: Optional[List[List[float]]] = Field(default=None, description="Custom stiffness matrix")
    custom_force: Optional[List[float]] = Field(default=None, description="Custom force vector")


class SolverConfig(BaseModel):
    """Solver configuration"""
    methods: List[str] = Field(
        default=["hhl", "vqls", "adiabatic", "classical"],
        description="Methods to run"
    )
    vqls_max_iter: int = Field(default=200, ge=10, le=500)
    adiabatic_steps: int = Field(default=1000, ge=100, le=100000)
    adiabatic_time: float = Field(default=300000.0, ge=1000.0, le=10000000.0)


class SolveRequest(BaseModel):
    """Combined request body for /solve endpoint"""
    # Problem definition
    matrix_size: int = Field(default=4, ge=2, le=10)
    force_position: int = Field(default=-1)
    force_value: float = Field(default=1.0)
    custom_matrix: Optional[List[List[float]]] = None
    custom_force: Optional[List[float]] = None
    # Solver configuration
    methods: List[str] = Field(default=["hhl", "vqls", "adiabatic", "classical"])
    vqls_max_iter: int = Field(default=200, ge=10, le=500)
    adiabatic_steps: int = Field(default=1000, ge=100, le=100000)
    adiabatic_time: float = Field(default=300000.0, ge=1000.0, le=10000000.0)


class MethodResult(BaseModel):
    """Result from a single method"""
    name: str
    solution: List[float]
    normalized_solution: List[float]
    cost_history: List[float]
    cost_history_x: List[int]  # Actual iteration numbers for x-axis
    final_cost: float
    execution_time: float
    iterations: int


class SolverResponse(BaseModel):
    """Complete solver response"""
    problem_size: int
    classical_reference: List[float]
    results: List[MethodResult]
    comparison_data: dict


# --- Core Algorithms ---
def generate_stiffness_matrix(n: int) -> np.ndarray:
    """Generate tridiagonal stiffness matrix for 1D elasticity"""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2 if i < n - 1 else 1
        if i > 0:
            A[i, i-1] = -1
        if i < n - 1:
            A[i, i+1] = -1
    return A


def qlsp_cost(psi: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """QLSP cost function: 1 - |<ψ|A|b>|² / ||Aψ||²"""
    psi = psi / norm(psi)
    Ab = A @ b
    overlap = np.abs(np.vdot(psi, Ab))
    denom = np.vdot(psi, A @ (A @ psi))
    if denom <= 1e-12:
        return 1.0
    return float(1 - (overlap ** 2) / denom)


def solve_hhl(A: np.ndarray, b: np.ndarray) -> MethodResult:
    """HHL Algorithm (spectral decomposition simulation)"""
    start_time = time.time()
    n = len(b)
    
    eigenvalues, eigenvectors = eigh(A)
    
    cost_history = []
    for n_ev in range(1, len(eigenvalues) + 1):
        x_hhl = np.zeros(n)
        for i in range(n_ev):
            v_i = eigenvectors[:, i]
            if abs(eigenvalues[i]) > 1e-12:
                x_hhl += (np.dot(v_i, b) / eigenvalues[i]) * v_i
        if norm(x_hhl) > 1e-12:
            x_hhl_norm = x_hhl / norm(x_hhl)
            cost_history.append(qlsp_cost(x_hhl_norm, A, b))
        else:
            cost_history.append(1.0)
    
    # Final solution
    x_final = np.zeros(n)
    for i in range(len(eigenvalues)):
        v_i = eigenvectors[:, i]
        if abs(eigenvalues[i]) > 1e-12:
            x_final += (np.dot(v_i, b) / eigenvalues[i]) * v_i
    
    x_normalized = x_final / norm(x_final) if norm(x_final) > 1e-12 else x_final
    
    # x-axis is eigenvalue index (1 to n)
    cost_history_x = list(range(1, len(cost_history) + 1))
    
    return MethodResult(
        name="HHL (Spectral)",
        solution=x_final.tolist(),
        normalized_solution=x_normalized.tolist(),
        cost_history=cost_history,
        cost_history_x=cost_history_x,
        final_cost=cost_history[-1] if cost_history else 1.0,
        execution_time=time.time() - start_time,
        iterations=len(eigenvalues)
    )


def solve_vqls(A: np.ndarray, b: np.ndarray, max_iter: int = 200) -> MethodResult:
    """VQLS - Variational Quantum Linear Solver"""
    start_time = time.time()
    n = len(b)
    
    # Create fixed basis for ansatz
    np.random.seed(42)
    basis = np.linalg.qr(np.random.randn(n, n))[0]
    
    def ansatz(params):
        x = sum(params[i] * basis[:, i] for i in range(n))
        n_x = norm(x)
        return x / n_x if n_x > 1e-12 else x
    
    cost_history = []
    
    def callback(params):
        x = ansatz(params)
        cost_history.append(qlsp_cost(x, A, b))
    
    initial = np.random.randn(n) * 0.1
    
    result = minimize(
        lambda p: qlsp_cost(ansatz(p), A, b),
        initial,
        method='Powell',
        callback=callback,
        options={'maxiter': max_iter}
    )
    
    x_final = ansatz(result.x)
    
    # x-axis is iteration number (1 to n)
    cost_history_x = list(range(1, len(cost_history) + 1))
    
    return MethodResult(
        name="VQLS (Variational)",
        solution=x_final.tolist(),
        normalized_solution=x_final.tolist(),
        cost_history=cost_history,
        cost_history_x=cost_history_x,
        final_cost=cost_history[-1] if cost_history else qlsp_cost(x_final, A, b),
        execution_time=time.time() - start_time,
        iterations=len(cost_history)
    )


def solve_adiabatic(
    A: np.ndarray, 
    b: np.ndarray, 
    steps: int = 1000, 
    T: float = 300000.0
) -> MethodResult:
    """Adiabatic quantum algorithm (Subaşı et al. inspired)
    
    Exactly as per original code:
    - psi starts as b (état initial |b⟩)
    - P_perp = I - |b><b|
    - Linear schedule: s = (i + 1) / steps
    - As = (1-s)*I + s*A
    - Hs = As @ P_perp @ As
    - U = expm(-1j * Hs * dt)
    """
    start_time = time.time()
    n = len(b)
    
    # État initial |b⟩
    psi = b.copy().astype(complex)
    P_perp = np.eye(n) - np.outer(b, b)
    
    cost_history = [qlsp_cost(np.real(psi), A, b)]
    cost_history_x = [0]  # Start at step 0
    
    # Calculate sampling interval to get ~500 points for visualization
    # This doesn't change the algorithm, just how often we record for the chart
    sample_interval = max(1, steps // 500)
    
    for i in range(steps):
        s = (i + 1) / steps  # schedule linéaire
        As = (1 - s) * np.eye(n) + s * A
        Hs = As @ P_perp @ As
        dt = T / steps
        U = expm(-1j * Hs * dt)
        psi = U @ psi
        
        # Record cost at sampled intervals for efficient visualization
        if i % sample_interval == 0 or i == steps - 1:
            cost_history.append(qlsp_cost(np.real(psi), A, b))
            cost_history_x.append(i + 1)  # Actual step number (1-indexed)
    
    x_final = np.real(psi)
    x_normalized = x_final / norm(x_final) if norm(x_final) > 1e-12 else x_final
    
    return MethodResult(
        name="Adiabatic (Subaşı)",
        solution=x_final.tolist(),
        normalized_solution=x_normalized.tolist(),
        cost_history=cost_history,
        cost_history_x=cost_history_x,
        final_cost=cost_history[-1] if cost_history else 1.0,
        execution_time=time.time() - start_time,
        iterations=steps
    )


def solve_classical(A: np.ndarray, b: np.ndarray) -> MethodResult:
    """Classical direct solver (reference)"""
    start_time = time.time()
    
    x_classical = solve(A, b)
    x_normalized = x_classical / norm(x_classical)
    
    cost = qlsp_cost(x_normalized, A, b)
    
    return MethodResult(
        name="Classical (Direct)",
        solution=x_classical.tolist(),
        normalized_solution=x_normalized.tolist(),
        cost_history=[cost],
        cost_history_x=[1],
        final_cost=cost,
        execution_time=time.time() - start_time,
        iterations=1
    )


# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "Quantum Elasticity Solver API",
        "version": "1.0.0",
        "endpoints": {
            "solve": "/solve",
            "methods": "/methods",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "quantum-elasticity-solver"}


@app.get("/methods")
async def get_methods():
    """Get available solving methods"""
    return {
        "methods": [
            {
                "id": "hhl",
                "name": "HHL Algorithm",
                "type": "quantum",
                "description": "Harrow-Hassidim-Lloyd algorithm using spectral decomposition"
            },
            {
                "id": "vqls",
                "name": "VQLS",
                "type": "quantum",
                "description": "Variational Quantum Linear Solver with classical optimization"
            },
            {
                "id": "adiabatic",
                "name": "Adiabatic",
                "type": "quantum",
                "description": "Adiabatic quantum computing approach (Subaşı et al.)"
            },
            {
                "id": "classical",
                "name": "Classical",
                "type": "classical",
                "description": "Direct matrix solver (reference solution)"
            }
        ]
    }


@app.post("/solve", response_model=SolverResponse)
async def solve_problem(request: SolveRequest):
    """Solve the elasticity problem with selected methods"""
    
    # Build matrix and force vector
    if request.custom_matrix is not None:
        A = np.array(request.custom_matrix, dtype=float)
        n = A.shape[0]
    else:
        n = request.matrix_size
        A = generate_stiffness_matrix(n)
    
    if request.custom_force is not None:
        b = np.array(request.custom_force, dtype=float)
    else:
        b = np.zeros(n)
        force_pos = request.force_position if request.force_position >= 0 else n - 1
        b[force_pos] = request.force_value
    
    # Normalize b
    b = b / norm(b) if norm(b) > 1e-12 else b
    
    # Classical reference
    x_classical = solve(A, b)
    x_classical_norm = x_classical / norm(x_classical)
    
    results = []
    
    # Run selected methods
    method_map = {
        "hhl": lambda: solve_hhl(A, b),
        "vqls": lambda: solve_vqls(A, b, request.vqls_max_iter),
        "adiabatic": lambda: solve_adiabatic(A, b, request.adiabatic_steps, request.adiabatic_time),
        "classical": lambda: solve_classical(A, b)
    }
    
    for method in request.methods:
        if method in method_map:
            try:
                result = method_map[method]()
                results.append(result)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error running {method}: {str(e)}"
                )
    
    # Build comparison data
    comparison = {
        "final_costs": {r.name: r.final_cost for r in results},
        "execution_times": {r.name: r.execution_time for r in results},
        "convergence_summary": {
            r.name: {
                "initial_cost": r.cost_history[0] if r.cost_history else None,
                "final_cost": r.final_cost,
                "improvement_factor": (
                    r.cost_history[0] / r.final_cost 
                    if r.cost_history and r.final_cost > 1e-12 
                    else None
                )
            }
            for r in results
        }
    }
    
    return SolverResponse(
        problem_size=n,
        classical_reference=x_classical_norm.tolist(),
        results=results,
        comparison_data=comparison
    )


@app.get("/demo")
async def run_demo():
    """Run a demo with default 4x4 elasticity problem"""
    problem = ElasticityProblem(matrix_size=4, force_position=-1, force_value=1.0)
    config = SolverConfig(
        methods=["hhl", "vqls", "adiabatic", "classical"],
        adiabatic_steps=300
    )
    return await solve_problem(problem, config)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

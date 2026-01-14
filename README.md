# Quantum Elasticity Solver

A polished, minimalistic web application for solving elasticity problems using quantum and classical methods. Compare **HHL**, **VQLS**, **Adiabatic** quantum algorithms against classical solvers in real-time.

![Quantum Elasticity Solver](https://img.shields.io/badge/Quantum-Solver-6366f1?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## 🌟 Features

- **4 Solving Methods**:
  - **HHL Algorithm** - Harrow-Hassidim-Lloyd spectral decomposition
  - **VQLS** - Variational Quantum Linear Solver
  - **Adiabatic** - Subaşı et al. inspired quantum adiabatic approach
  - **Classical** - Direct matrix solver (LU decomposition)

- **Real-time Visualization**:
  - Interactive stiffness matrix display
  - Convergence comparison charts
  - Solution vector comparison
  - Performance metrics dashboard

- **Modern UI**:
  - Glassmorphism design
  - Dark theme with animated background
  - Responsive layout
  - Premium aesthetics

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone or navigate to the project
cd VQLS_ADIABATIQUE

# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

### Local Development

#### Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend

# Serve with any static server, e.g.:
python -m http.server 3000

# Or use VS Code Live Server extension
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/methods` | GET | List available methods |
| `/solve` | POST | Solve elasticity problem |
| `/demo` | GET | Run demo with default problem |

### Example Request

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "matrix_size": 4,
    "force_position": -1,
    "force_value": 1.0,
    "methods": ["hhl", "vqls", "adiabatic", "classical"],
    "vqls_max_iter": 200,
    "adiabatic_steps": 300
  }'
```

## 🧮 The Problem

We solve the linear system **Ax = b** where:

- **A** is a tridiagonal stiffness matrix (1D elasticity)
- **b** is the force vector
- **x** is the displacement solution

### Stiffness Matrix (n=4)
```
A = | 2  -1   0   0 |
    |-1   2  -1   0 |
    | 0  -1   2  -1 |
    | 0   0  -1   1 |
```

## 📊 Methods Comparison

| Method | Type | Best For | Complexity |
|--------|------|----------|------------|
| HHL | Quantum | Well-conditioned systems | O(log N) |
| VQLS | Quantum | NISQ devices | Variational |
| Adiabatic | Quantum | Ground state preparation | O(1/gap²) |
| Classical | Classical | Reference solution | O(N³) |

## 🏗️ Project Structure

```
VQLS_ADIABATIQUE/
├── docker-compose.yml
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py          # FastAPI + algorithms
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    ├── index.html
    ├── styles.css
    └── app.js
```

## 🔬 Algorithm Details

### HHL (Harrow-Hassidim-Lloyd)
Uses quantum phase estimation and eigenvalue inversion to solve linear systems with exponential speedup for sparse, well-conditioned matrices.

### VQLS (Variational Quantum Linear Solver)
Combines parameterized quantum circuits with classical optimization. Suitable for near-term quantum devices (NISQ era).

### Adiabatic Quantum Computing
Based on Subaşı et al., evolves the system from an easy-to-prepare initial state to the solution state by slowly varying the Hamiltonian.

## 📝 License

MIT License - Feel free to use and modify.

## 🙏 Acknowledgments

- Qiskit team for quantum computing tools
- Subaşı et al. for the adiabatic QLSP approach
- FastAPI for the excellent Python web framework
